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
    """准备训练数据集 - 使用Qwen3-VL官方结构化格式"""
    with open(config['dataset_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', {})
        images = item.get('images', [])

        # 构建用户输入文本
        user_text = instruction
        if input_text:
            user_text += "\n\n" + input_text

        # 构建助手回复（确保为纯净JSON）
        if isinstance(output, dict):
            try:
                assistant_content = json.dumps(output, ensure_ascii=False)
            except:
                parts = [f"{k}: {v}" for k, v in output.items()]
                assistant_content = "\n".join(parts)
        else:
            assistant_content = str(output)

        # 清洗输出数据，移除markdown代码块
        if assistant_content.startswith("```"):
            import re
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', assistant_content, re.DOTALL)
            if match:
                assistant_content = match.group(1).strip()

        # 确保images始终是一个列表，即使为空
        if not isinstance(images, list):
            images = [images] if images else []

        # 构建消息内容，支持多个图像
        content_list = []

        # 添加所有图像
        for _ in images:
            content_list.append({"type": "image"})

        # 添加文本内容
        content_list.append({"type": "text", "text": user_text})

        # 将复杂的消息结构存储为JSON字符串，避免PyArrow嵌套结构问题
        messages_json = json.dumps([
            {
                "role": "user",
                "content": content_list
            },
            {
                "role": "assistant",
                "content": assistant_content  # 必须是纯字符串
            }
        ], ensure_ascii=False)

        formatted_data.append({
            "messages_json": messages_json,  # 将messages作为JSON字符串存储
            "images": images
        })

    return Dataset.from_list(formatted_data)


class DataCollatorForQwen3VL:
    def __init__(self, processor, config):
        self.processor = processor
        self.config = config

    def load_image(self, image_path):
        """安全加载图像"""
        if not image_path:
            return None
        try:
            if image_path.startswith('data:image'):
                import base64
                import io
                from PIL import Image
                base64_str = image_path.split(',')[1]
                image_bytes = base64.b64decode(base64_str)
                return Image.open(io.BytesIO(image_bytes)).convert('RGB')

            import os
            possible_paths = [
                image_path,
                os.path.join(os.path.dirname(self.config['dataset_path']), image_path),
                os.path.join('.', image_path)
            ]
            for img_path in possible_paths:
                if os.path.exists(img_path):
                    from PIL import Image
                    return Image.open(img_path).convert('RGB')
            return None
        except:
            return None

    def __call__(self, features):
        # 处理每个特征
        batch = {'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'image_grid_thw': [], 'labels': []}

        for feature in features:
            # 从JSON字符串解析messages
            import json as json_module
            messages = json_module.loads(feature['messages_json'])

            # 加载多个图像
            pil_images = []
            if feature.get('images'):
                for img_path in feature['images']:
                    img = self.load_image(img_path)
                    if img is not None:
                        pil_images.append(img)

            if not pil_images:
                continue  # 跳过没有图像的样本

            # 应用聊天模板
            formatted_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # 处理文本和多个图像
            inputs = self.processor(
                text=[formatted_text],
                images=pil_images,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            )

            # 提取各个组件
            batch['input_ids'].append(inputs['input_ids'][0])
            batch['attention_mask'].append(inputs['attention_mask'][0])

            # 正确处理多个图像的pixel_values和image_grid_thw
            # 将所有图像的pixel_values连接起来
            if inputs['pixel_values'].dim() == 4:  # 如果已经是批量维度
                batch['pixel_values'].append(inputs['pixel_values'])
            else:  # 如果不是批量维度，则扩展维度
                batch['pixel_values'].append(inputs['pixel_values'].unsqueeze(0))

            # image_grid_thw也需要正确处理
            batch['image_grid_thw'].append(inputs['image_grid_thw'])

            # 创建labels - 只对assistant部分计算损失
            input_ids = inputs['input_ids'][0]
            labels = input_ids.clone()

            # 找到assistant消息的起始位置，将user部分设为-100
            # 这里简化处理，假设格式是固定的
            assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|assistant|>")
            if assistant_token_id is not None and assistant_token_id in input_ids:
                assistant_start_idx = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0][0]
                # 将assistant之前的部分设为-100（不计算损失）
                labels[:assistant_start_idx] = -100
            else:
                # 如果找不到assistant token，将所有内容都计算损失
                pass

            batch['labels'].append(labels)

        # 堆叠张量
        import torch
        if batch['input_ids']:  # 确保批次不为空
            # 找到最大长度用于padding
            max_length = max(ids.size(0) for ids in batch['input_ids'])

            # Padding
            padded_input_ids = []
            padded_attention_masks = []
            padded_labels = []

            for i in range(len(batch['input_ids'])):
                ids = batch['input_ids'][i]
                attn = batch['attention_mask'][i]
                lbl = batch['labels'][i]

                pad_length = max_length - ids.size(0)
                pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id

                padded_ids = torch.cat([ids, torch.full((pad_length,), pad_id, dtype=ids.dtype)])
                padded_attn = torch.cat([attn, torch.zeros(pad_length, dtype=attn.dtype)])
                padded_lbl = torch.cat([lbl, torch.full((pad_length,), -100, dtype=lbl.dtype)])

                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_attn)
                padded_labels.append(padded_lbl)

            batch['input_ids'] = torch.stack(padded_input_ids)
            batch['attention_mask'] = torch.stack(padded_attention_masks)
            batch['labels'] = torch.stack(padded_labels)

            # 正确堆叠pixel_values和image_grid_thw
            # 对于Qwen3-VL，我们需要特别小心处理这些张量
            if len(batch['pixel_values']) > 0:
                # pixel_values可能有不同的形状，需要特殊处理
                batch['pixel_values'] = torch.cat(batch['pixel_values'], dim=0)

            if len(batch['image_grid_thw']) > 0:
                # image_grid_thw的形状可能不同，需要特殊处理
                # 将所有image_grid_thw连接起来
                batch['image_grid_thw'] = torch.cat(batch['image_grid_thw'], dim=0)

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

    # 确保模型处于训练模式并激活LoRA参数
    model.train()
    if config['finetune_method'] == 'lora':
        # 确保LoRA参数可训练
        for name, param in model.named_parameters():
            if "lora_" in name.lower():
                param.requires_grad = True
                print(f"激活LoRA参数: {name}")

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
            def print_trainable_parameters():
                model.module.print_trainable_parameters()
            model.print_trainable_parameters = print_trainable_parameters
            if rank == 0:  # 只在主进程打印
                model.module.print_trainable_parameters()
    else:
        # 单GPU情况
        if config['finetune_method'] == 'lora':
            model.print_trainable_parameters()
        model.train()  # 确保单GPU模式下也处于训练模式

    print(f"使用每设备批次大小: {config['per_device_train_batch_size']}")
    print("开始训练...")
    
    # 验证数据加载和loss计算
    if rank == 0:
        print("\n=== 训练前验证 ===")
        # 1. 检查单样本processor输出
        sample = train_dataset[0]
        import json as json_module
        test_messages = json_module.loads(sample["messages_json"])  # 从JSON字符串解析messages

        # 加载所有图像
        test_images = []
        for img_path in sample["images"]:
            img = DataCollatorForQwen3VL(processor, config).load_image(img_path)
            if img is not None:
                test_images.append(img)

        # 使用apply_chat_template处理消息
        formatted_text = processor.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        test_batch = processor(
            text=formatted_text,
            images=test_images if test_images else None,
            padding=True,
            return_tensors="pt"
        )

        # 检查图像token
        image_token_id = processor.image_token_id
        img_token_count = (test_batch["input_ids"][0] == image_token_id).sum().item()
        print(f"图像token数量: {img_token_count} (可能因图像尺寸而异)")

        # 检查labels mask
        labels = test_batch.get("labels", test_batch["input_ids"].clone())
        if labels is not None:
            user_mask_ratio = (labels[0] == -100).sum().item() / labels[0].numel()
            print(f"用户消息mask比例: {user_mask_ratio:.1%} (可能因数据而异)")
        else:
            print("警告: 未找到labels")

        # 2. 前向传播测试loss
        model.eval()
        with torch.no_grad():
            # 确保只传递模型需要的参数，包括Qwen3-VL特有的键
            model_inputs = {k: v.to(f"cuda:{local_rank}") for k, v in test_batch.items() if k in ['input_ids', 'attention_mask', 'pixel_values', 'position_ids', 'image_grid_thw']}
            outputs = model(**model_inputs)
            if outputs is not None and hasattr(outputs, 'loss') and outputs.loss is not None:
                print(f"初始loss: {outputs.loss.item():.4f} (合理范围: 2.0~8.0)")
            else:
                print("警告: loss为None或不可访问，继续训练...")
        model.train()
        print("=== 验证完成 ===\n")
        
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForQwen3VL(processor, config),
        tokenizer=processor,  # 使用tokenizer参数而不是processing_class
        callbacks=[SafeSaveCallback(finetune_method=config['finetune_method'])],  # 添加安全保存回调
    )

    # 确保启用input_require_grads
    model.enable_input_require_grads()

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

    merged_model = base_model.merge_and_unload()
    merged_model.save_pretrained("./qwen3-vl-2b-instruct-finetuned")


if __name__ == "__main__":
    # 解析命令行参数（如果有的话）
    parser = argparse.ArgumentParser(description='Qwen3-VL Multi-GPU Finetuning')
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--config", type=str, default="./config/multi_gpu_config.yaml", help="Configuration file path")

    args = parser.parse_args()
    if args.local_rank != -1:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    fine_tune(config_path=args.config)
