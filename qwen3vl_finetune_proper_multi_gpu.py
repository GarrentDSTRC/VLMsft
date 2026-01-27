#!/usr/bin/env python
"""
使用Qwen3-VL模型进行多GPU微调 - 支持4x4090设置
"""
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, TrainingArguments, Trainer
from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
import os
import argparse
import warnings
import yaml

def load_config(config_path="./config/multi_gpu_config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataset(config):
    """准备训练数据集 - 适配Llama Factory格式，包含图像信息"""
    # 读取转换后的数据
    with open(config['dataset_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 转换为HuggingFace Dataset格式
    formatted_data = []
    for item in data:
        # 处理Llama Factory格式数据
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', {})
        images = item.get('images', [])  # 获取图像路径

        # 构建完整的对话
        # 将instruction和input组合成用户输入
        user_input = instruction
        if input_text:
            user_input += "\n\n" + input_text

        # 处理输出 - output可能是字典格式
        if isinstance(output, dict):
            if 'reasoning' in output and 'action' in output:
                # 按照JSON格式构建答案
                assistant_response = json.dumps(output, ensure_ascii=False, indent=2)
            else:
                # 如果只有部分字段，尝试构建答案
                response_parts = []
                for key, value in output.items():
                    if isinstance(value, list):
                        response_parts.append(f"{key}: {str(value)}")
                    else:
                        response_parts.append(f"{key}: {value}")
                assistant_response = "\n".join(response_parts)
        else:
            assistant_response = str(output)

        # 构建对话格式
        conversation_str = f"<|user|>\n{user_input}\n<|assistant|>\n{assistant_response}"

        formatted_data.append({
            "text": conversation_str,
            "images": images  # 保存图像路径信息
        })

    # 创建Dataset对象
    dataset = Dataset.from_list(formatted_data)
    return dataset


class DataCollatorForQwen3VL:
    def __init__(self, processor, config):
        self.processor = processor
        self.config = config

    def __call__(self, features):
        # 提取文本和图像
        texts = []
        images_list = []

        for feature in features:
            texts.append(feature['text'])
            # 如果有图像路径，加载图像
            if 'images' in feature and feature['images']:
                import os
                from PIL import Image
                import base64
                import io

                image_path = feature['images'][0]  # 假设只处理第一个图像
                if image_path.startswith('data:image'):
                    # 处理base64图像
                    base64_str = image_path.split(',')[1]
                    image_bytes = base64.b64decode(base64_str)
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    images_list.append(pil_image)
                else:
                    # 处理本地文件路径
                    possible_paths = [
                        image_path,  # 原始路径
                        os.path.join(os.path.dirname(self.config['dataset_path']), image_path),  # 相对于数据集文件
                        os.path.join('.', image_path),  # 相对当前目录
                        os.path.join('..', image_path),  # 上级目录
                    ]

                    pil_image = None
                    for img_path in possible_paths:
                        if os.path.exists(img_path):
                            pil_image = Image.open(img_path).convert('RGB')
                            break

                    if pil_image is not None:
                        images_list.append(pil_image)
                    else:
                        # 如果找不到图像，只处理文本
                        images_list.append(None)
            else:
                # 没有图像，只处理文本
                images_list.append(None)

        # 分别收集有图像和无图像的样本
        texts_with_images = []
        images_for_processing = []
        texts_without_images = []

        for text, img in zip(texts, images_list):
            if img is not None:
                texts_with_images.append(text)
                images_for_processing.append(img)
            else:
                texts_without_images.append(text)

        # 处理有图像的样本
        batch_with_images = {}
        if texts_with_images:
            # 为有图像的样本构建消息格式
            messages_list = []
            for text in texts_with_images:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text.split('\n<|assistant|>\n')[0].replace('<|user|>\n', '')}
                        ]
                    }
                ]
                messages_list.extend(messages)  # 为每个样本单独处理

            # 应用对话模板并处理文本和图像
            formatted_texts = []
            for msg in messages_list:
                formatted_text = self.processor.apply_chat_template(
                    [msg],  # 每次处理一个消息
                    tokenize=False,
                    add_generation_prompt=False
                )
                formatted_texts.append(formatted_text)

            # 批量处理文本和图像
            batch_with_images = self.processor(
                text=formatted_texts,
                images=images_for_processing,
                padding=True,  # 现在启用padding，让processor处理
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            )

        # 处理无图像的样本
        batch_without_images = {}
        if texts_without_images:
            batch_without_images = self.processor(
                text=texts_without_images,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            )

        # 合并两个批次
        if batch_with_images and batch_without_images:
            print("figggggggg+text")
            # 合并两个批次的数据
            batch = {}
            all_keys = set(list(batch_with_images.keys()) + list(batch_without_images.keys()))

            for key in all_keys:
                if key in batch_with_images and key in batch_without_images:
                    # 两个批次都有这个键，合并它们
                    tensor1 = batch_with_images[key]
                    tensor2 = batch_without_images[key]

                    # 确保两个张量维度兼容
                    if tensor1.shape[1:] == tensor2.shape[1:]:
                        # 如果除了批次维度外其他维度相同，直接拼接
                        batch[key] = torch.cat([tensor1, tensor2], dim=0)
                    else:
                        # 如果维度不同，需要进行填充
                        max_seq_len = max(tensor1.shape[1] if len(tensor1.shape) > 1 else 0,
                                         tensor2.shape[1] if len(tensor2.shape) > 1 else 0)

                        # 对于input_ids, attention_mask等序列数据，进行填充
                        if key in ['input_ids', 'attention_mask', 'labels']:
                            # 确保张量是2维的
                            if tensor1.dim() == 1:
                                tensor1 = tensor1.unsqueeze(0)
                            if tensor2.dim() == 1:
                                tensor2 = tensor2.unsqueeze(0)

                            # 计算需要填充的长度
                            pad_len1 = max_seq_len - tensor1.shape[1]
                            pad_len2 = max_seq_len - tensor2.shape[1]

                            if pad_len1 > 0:
                                pad_token_id = getattr(self.processor.tokenizer, 'pad_token_id', 0)
                                tensor1 = torch.nn.functional.pad(tensor1, (0, pad_len1), value=pad_token_id)
                            if pad_len2 > 0:
                                pad_token_id = getattr(self.processor.tokenizer, 'pad_token_id', 0)
                                tensor2 = torch.nn.functional.pad(tensor2, (0, pad_len2), value=pad_token_id)

                            batch[key] = torch.cat([tensor1, tensor2], dim=0)
                        else:
                            # 对于其他类型的数据，尝试直接拼接
                            batch[key] = torch.cat([tensor1, tensor2], dim=0)
                elif key in batch_with_images:
                    batch[key] = batch_with_images[key]
                elif key in batch_without_images:
                    batch[key] = batch_without_images[key]
        elif batch_with_images:
            batch = batch_with_images
            print("figggggggg")
        elif batch_without_images:
            batch = batch_without_images
            print("textttttttt")
        else:
            # 如果都没有，处理纯文本
            batch = self.processor(
                text=texts,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            )

        # 不要在collate_fn中处理CUDA设备分配，让Trainer自动处理
        # 避免在DataLoader worker进程中引发CUDA reinitialization错误

        # 设置标签
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids'].clone()
        return batch


class SafeSaveCallback(TrainerCallback):
    def __init__(self, finetune_method='lora'):
        super().__init__()
        self.finetune_method = finetune_method

    def on_save(self, args, state, control, **kwargs):
        # 在DDP环境中，只让rank 0处理保存
        if dist.is_initialized() and dist.get_rank() != 0:
            return control

        model = kwargs['model']

        # 处理DDP包装
        if hasattr(model, 'module'):
            model = model.module

        # 确保保存时处于评估模式
        training_mode = model.training
        model.eval()

        # 保存检查点
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # 根据微调方法选择保存策略
        if self.finetune_method == 'lora':
            # 仅保存LoRA适配器
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(checkpoint_path)
                print(f"✓ LoRA适配器保存至: {checkpoint_path}")
        elif self.finetune_method == 'full':
            # 全量训练：保存完整模型（注意内存！）
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(
                    checkpoint_path,
                    safe_serialization=True,  # 推荐使用safetensors
                    max_shard_size="5GB"      # 避免单文件过大
                )
                print(f"✓ 全量模型保存至: {checkpoint_path}")

        # 通用：保存processor和训练状态
        processor = kwargs.get('tokenizer') or kwargs.get('processor')
        if processor is not None:
            processor.save_pretrained(checkpoint_path)

        # 保存训练状态
        torch.save({
            'global_step': state.global_step,
            'epoch': state.epoch,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }, os.path.join(checkpoint_path, "training_state.pt"))

        # 恢复训练模式
        if training_mode:
            model.train()

        return control

    def on_train_begin(self, args, state, control, **kwargs):
        # 处理从检查点恢复
        if state.is_world_process_zero and hasattr(state, 'global_step') and state.global_step > 0:
            print(f"从检查点恢复训练，当前步数: {state.global_step}")
        return control


def setup_distributed_training():
    """设置分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        return False

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    return True


def fine_tune(config_path="./config/multi_gpu_config.yaml"):
    """开始微调过程"""
    # 加载配置
    config = load_config(config_path)
    print(f"使用配置文件: {config_path}")

    # 获取GPU数量
    n_gpu = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 初始化分布式训练
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        rank = 0  # 设定默认rank为0

    # 检查是否存在检查点，如果有则从中恢复训练
    output_dir = config['output_dir']
    resume_from_checkpoint = None

    # 只在rank 0检测检查点
    if rank == 0:
        if os.path.exists(output_dir):
            # 查找最新的PEFT格式检查点 (区分LoRA/全量)
            checkpoints = []
            for d in os.listdir(output_dir):
                checkpoint_path = os.path.join(output_dir, d)
                if os.path.isdir(checkpoint_path) and "checkpoint-" in d:
                    # LoRA检查点特征
                    is_lora = (os.path.exists(os.path.join(checkpoint_path, "adapter_model.bin")) or
                              os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors"))) and \
                              os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))

                    # 全量检查点特征
                    is_full = (os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")) or
                              os.path.exists(os.path.join(checkpoint_path, "model.safetensors"))) and \
                              os.path.exists(os.path.join(checkpoint_path, "config.json"))

                    if is_lora or is_full:
                        try:
                            step = int(d.split("-")[1])
                            cp_type = 'lora' if is_lora else 'full'
                            checkpoints.append((step, checkpoint_path, cp_type))
                        except (ValueError, IndexError):
                            # 如果无法解析步数，跳过此目录
                            continue

            if checkpoints:
                # 按步数排序，获取最新的
                checkpoints.sort(key=lambda x: x[0])
                resume_from_checkpoint, checkpoint_type = checkpoints[-1][1], checkpoints[-1][2]
                print(f"✓ 检测到{checkpoint_type}检查点: {resume_from_checkpoint}")
            else:
                print(f"输出目录 {output_dir} 已存在但没有有效的PEFT检查点，将从头开始训练")
        else:
            print(f"输出目录 {output_dir} 不存在，将创建新目录并从头开始训练")

    # 同步检查点信息到所有rank
    if dist.is_initialized():
        # 创建一个包含检查点路径存在状态的张量
        if rank == 0:
            resume_flag = torch.tensor(1 if resume_from_checkpoint else 0, device=f"cuda:{local_rank}")
        else:
            resume_flag = torch.tensor(0, device=f"cuda:{local_rank}")

        dist.broadcast(resume_flag, src=0)

        # 广播检查点路径
        if resume_flag.item() == 1:
            if rank == 0:
                # 主进程发送路径
                max_path_len = 512
                path_bytes = list(resume_from_checkpoint.encode('utf-8'))
                path_tensor = torch.zeros(max_path_len, dtype=torch.uint8, device=f"cuda:{local_rank}")
                path_tensor[:len(path_bytes)] = torch.tensor(path_bytes, device=f"cuda:{local_rank}")
                dist.broadcast(path_tensor, src=0)
            else:
                # 其他进程接收路径
                max_path_len = 512
                path_tensor = torch.zeros(max_path_len, dtype=torch.uint8, device=f"cuda:{local_rank}")
                dist.broadcast(path_tensor, src=0)
                path_bytes = path_tensor.cpu().numpy().astype('uint8')
                resume_from_checkpoint = bytes(path_bytes[path_bytes != 0]).decode('utf-8', errors='ignore')
    else:
        # 非分布式环境
        pass

    print(f"启动训练，当前节点GPU数量: {n_gpu}")
    print(f"本地Rank: {local_rank}, 总世界大小: {world_size}")

    if n_gpu < 1:
        raise RuntimeError("至少需要一个GPU进行训练")

    # 打印GPU信息
    for i in range(n_gpu):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    print("开始加载Qwen3-VL模型...")

    # 加载模型和处理器 - 使用推荐的类名
    model_name = config['model_name']

    # 使用合适的模型精度
    dtype = torch.bfloat16 if config['model_dtype'] == 'bfloat16' else torch.float16

    # 加载基础模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,  # 使用配置文件中指定的精度
        device_map=f"cuda:{local_rank}",  # 为DDP设置正确的device_map
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # 将模型移至指定设备
    if torch.cuda.device_count() > 0:
        torch.cuda.set_device(local_rank)
        model = model.to(f"cuda:{local_rank}")

    processor = Qwen3VLProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    print("模型加载完成")
    print(f"模型参数数量: {model.num_parameters():,}")

    # 准备数据集
    print("准备数据集...")
    full_dataset = prepare_dataset(config)

    # 读取原始数据用于后续保存验证集
    with open(config['dataset_path'], 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # 划分训练集和验证集 (根据配置文件)
    split_dataset = full_dataset.train_test_split(test_size=config['validation_split'], seed=42)  # 使用固定种子以确保一致性
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    print(f"训练集准备完成，共 {len(train_dataset)} 个样本")
    print(f"验证集准备完成，共 {len(eval_dataset)} 个样本")

    # 保存验证集到指定路径，只在主进程(rank 0)中执行
    if rank == 0:  # 只在主进程保存
        test_set_path = "./data/vlm_finetune_dataset_fixed/testset.json"
        os.makedirs(os.path.dirname(test_set_path), exist_ok=True)

        # 获取验证集在原始数据中的索引
        # 使用datasets库的info或indices属性来获取原始索引
        if hasattr(split_dataset['test'], 'indices'):
            val_indices = split_dataset['test'].indices
        else:
            # 如果无法获取indices，使用固定种子重新计算
            import random
            indices = list(range(len(original_data)))
            random.seed(42)  # 使用与train_test_split相同的种子
            random.shuffle(indices)
            val_size = int(len(original_data) * config['validation_split'])
            val_indices = indices[-val_size:]

        # 从原始数据中提取对应索引的数据
        validation_data = [original_data[i] for i in val_indices]

        with open(test_set_path, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, ensure_ascii=False, indent=2)
        print(f"验证集已保存到 {test_set_path} (共 {len(validation_data)} 个样本)")

    # 确保所有进程等待验证集保存完成
    if dist.is_initialized():
        dist.barrier()  # 同步所有进程

    # 根据配置选择微调方法
    if config['finetune_method'] == 'lora':
        # 检查是否从检查点恢复
        if resume_from_checkpoint is not None:
            print(f"从检查点 {resume_from_checkpoint} 加载LoRA适配器")
            # 先加载基础模型
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=f"cuda:{local_rank}",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # 关键：在加载适配器前将模型设为训练模式
            model.train()

            # 加载LoRA适配器
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, resume_from_checkpoint, is_trainable=True)

            # 关键：再次确保模型处于训练模式
            model.train()

            # 确保所有LoRA参数可训练
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "adapter" in name.lower():
                    param.requires_grad = True

            # 只在主进程打印可训练参数
            if rank == 0:
                print("LoRA适配器加载后参数状态:")
                model.print_trainable_parameters()
        else:
            # 配置LoRA - 使用配置文件中的参数
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,  # 确保是训练模式
                r=config['lora_r'],
                lora_alpha=config['lora_alpha'],
                lora_dropout=config['lora_dropout'],
                target_modules=config['target_modules']
            )
            # 应用LoRA配置
            model = get_peft_model(model, peft_config)
            model.train()  # 确保模型处于训练模式
        # 只有在LoRA模式下才有print_trainable_parameters方法
        if rank == 0 or not dist.is_initialized():  # 只在主进程或非分布式情况下打印
            model.print_trainable_parameters()
    elif config['finetune_method'] == 'full':
        # 检查是否从检查点恢复
        if resume_from_checkpoint is not None and checkpoint_type == 'full':
            print(f"从全量检查点加载: {resume_from_checkpoint}")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                resume_from_checkpoint,  # 直接加载完整模型
                torch_dtype=dtype,
                device_map=f"cuda:{local_rank}",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model.train()
        else:
            # 从头开始全量训练
            print("使用全量微调，所有参数都将被更新")
            # 为全量微调打印模型参数信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"总参数: {total_params:,}")
            print(f"可训练参数: {trainable_params:,}")
            print(f"可训练参数占比: {100 * trainable_params / total_params:.2f}%")
    else:
        raise ValueError(f"未知的微调方法: {config['finetune_method']}，请使用 'lora' 或 'full'")

    # 训练参数 - 使用配置文件中的参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=config['overwrite_output_dir'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        warmup_ratio=config['warmup_ratio'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_steps=config['eval_steps'],
        eval_strategy="steps",
        save_total_limit=config['save_total_limit'],
        report_to=None,
        remove_unused_columns=False,
        gradient_checkpointing=config['gradient_checkpointing'],
        bf16=config.get('bf16', False),  # 使用配置中指定的精度
        logging_dir=config['logging_dir'],
        dataloader_pin_memory=config['dataloader_pin_memory'],
        dataloader_num_workers=config['dataloader_num_workers'],
        max_grad_norm=config['max_grad_norm'],
        dataloader_drop_last=config['dataloader_drop_last'],
        # 分布式训练相关参数
        local_rank=local_rank,
        ddp_find_unused_parameters=config['ddp_find_unused_parameters'],
    )

    # 在模型创建之后、训练之前处理DDP问题
    # 注意：必须在应用LoRA后再应用DDP
    if dist.is_initialized():  # 使用分布式数据并行训练
        # 检查是否有可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if trainable_params == 0:
            print(f"[Rank {rank}] 警告：模型没有可训练参数，尝试修复...")
            # 尝试修复：重新激活LoRA参数
            if config['finetune_method'] == 'lora':
                for name, param in model.named_parameters():
                    if "lora" in name.lower() or "adapter" in name.lower():
                        param.requires_grad = True
                print(f"[Rank {rank}] 重新激活了LoRA参数")

        # 再次检查
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_params == 0:
            raise RuntimeError(f"[Rank {rank}] 模型仍然没有可训练参数，无法进行分布式训练")

        print(f"[Rank {rank}] 初始化分布式训练，可训练参数数量: {trainable_params:,}")
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=config['ddp_find_unused_parameters'])

        # 为DDP模型添加必要的方法以满足Trainer要求
        def gradient_checkpointing_enable(**kwargs):
            model.module.gradient_checkpointing_enable(**kwargs)
        model.gradient_checkpointing_enable = gradient_checkpointing_enable

        def enable_input_require_grads():
            model.module.enable_input_require_grads()
        model.enable_input_require_grads = enable_input_require_grads

        def disable_input_require_grads():
            model.module.disable_input_require_grads()
        model.disable_input_require_grads = disable_input_require_grads

        # 添加forward方法以支持Trainer
        def forward(*args, **kwargs):
            return model.module(*args, **kwargs)
        model.forward = forward

        # 只有在LoRA模式下才访问peft模型的方法
        if config['finetune_method'] == 'lora':
            if rank == 0:  # 只在主进程打印
                model.module.print_trainable_parameters()
    else:
        # 单GPU情况
        if config['finetune_method'] == 'lora':
            model.print_trainable_parameters()

    print(f"使用每设备批次大小: {config['per_device_train_batch_size']}")
    print("开始训练...")

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForQwen3VL(processor, config),
        processing_class=processor,
        callbacks=[SafeSaveCallback(finetune_method=config['finetune_method'])]  # 添加安全保存回调
    )

    # 手动处理恢复逻辑 - 不使用trainer.train(resume_from_checkpoint)
    if resume_from_checkpoint is not None and rank == 0:
        print(f"手动从检查点 {resume_from_checkpoint} 恢复训练状态...")
        training_state_path = os.path.join(resume_from_checkpoint, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            # 这里可以恢复更多状态，如随机数生成器状态
            print(f"已恢复训练状态，步数: {training_state['global_step']}, 轮次: {training_state['epoch']}")
        else:
            print("警告：找不到训练状态文件，将从检查点继续但不恢复训练状态")

    # 开始训练 - 不使用resume_from_checkpoint参数
    trainer.train()

    # 训练完成后保存最终模型
    print("正在保存最终模型...")
    final_model_dir = os.path.join(output_dir, "final_model")
    if dist.is_initialized():
        if dist.get_rank() == 0:  # 只有rank 0保存
            save_model = model.module if hasattr(model, 'module') else model
            save_model.eval()  # 保存前切换到评估模式

            # 保存LoRA适配器
            if config['finetune_method'] == 'lora':
                save_model.save_pretrained(final_model_dir)

            # 保存processor
            processor.save_pretrained(final_model_dir)
            print(f"最终模型已保存到 {final_model_dir}")

            # 保存后切换回训练模式
            save_model.train()
    else:
        # 非分布式情况
        trainer.model.eval()
        trainer.save_model(final_model_dir)
        processor.save_pretrained(final_model_dir)
        trainer.model.train()
        print(f"最终模型已保存到 {final_model_dir}")

    # 合并LoRA权重（如果需要）
    # 如果使用DataParallel或DDP，需要通过.module访问底层模型
    if hasattr(model, 'module'):
        base_model = model.module
    else:
        base_model = model

    # 如果需要合并权重
    # merged_model = base_model.merge_and_unload()
    # merged_model.save_pretrained("./qwen3-vl-2b-instruct-finetuned")


if __name__ == "__main__":
    # 解析命令行参数（如果有的话）
    parser = argparse.ArgumentParser(description='Qwen3-VL Multi-GPU Finetuning')
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--config", type=str, default="./config/multi_gpu_config.yaml", help="Configuration file path")

    args = parser.parse_args()
    if args.local_rank != -1:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    fine_tune(config_path=args.config)
