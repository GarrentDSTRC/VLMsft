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
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments, Trainer
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
    """准备训练数据集"""
    # 读取转换后的数据
    with open(config['dataset_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 转换为HuggingFace Dataset格式
    formatted_data = []
    for item in data:
        # 处理对话数据
        conversations = item.get('conversations', [])
        if not conversations:
            continue

        # 构建提示和响应
        conversation_str = ""
        for conv in conversations:
            role = conv.get('role', '')
            content = conv.get('content', '')

            # 根据角色构建消息
            if role == 'user':
                conversation_str += f"<|user|>\n{content}"
            elif role == 'assistant':
                conversation_str += f"\n<|assistant|>\n{content}"

        formatted_data.append({
            "text": conversation_str,
        })

    # 创建Dataset对象
    dataset = Dataset.from_list(formatted_data)
    return dataset


class DataCollatorForQwen3VL:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # 提取文本
        texts = [feature['text'] for feature in features]

        # 处理文本 - 确保返回字典而不是嵌套张量
        inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )

        # 不要在collate_fn中处理CUDA设备分配，让Trainer自动处理
        # 避免在DataLoader worker进程中引发CUDA reinitialization错误

        # 设置标签
        inputs['labels'] = inputs['input_ids'].clone()
        return inputs


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

    # 检查是否存在检查点，如果有则从中恢复训练
    output_dir = config['output_dir']
    resume_from_checkpoint = None

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

    # 只在主进程(rank 0)中查找检查点
    if rank == 0:
        # 查找最新的检查点
        if os.path.exists(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))
                resume_from_checkpoint = os.path.join(output_dir, checkpoints[-1])
                print(f"发现检查点，将从 {resume_from_checkpoint} 继续训练")
            else:
                print(f"输出目录 {output_dir} 已存在但没有检查点，将从头开始训练")
        else:
            print(f"输出目录 {output_dir} 不存在，将创建新目录并从头开始训练")

    # 同步所有进程，确保它们都获取到相同的resume_from_checkpoint状态
    if dist.is_initialized():
        # 创建一个包含检查点路径存在状态的张量
        resume_flag = torch.tensor(1 if resume_from_checkpoint is not None else 0, device=f"cuda:{gpu}" if torch.cuda.is_available() and gpu < torch.cuda.device_count() else "cpu")
        dist.broadcast(resume_flag, src=0)

        # 如果检查点存在，需要同步路径信息
        if resume_flag.item() == 1 and rank != 0:
            # 使用一个固定的路径长度来同步路径
            max_path_len = 512
            path_tensor = torch.zeros(max_path_len, dtype=torch.uint8, device=f"cuda:{gpu}" if torch.cuda.is_available() and gpu < torch.cuda.device_count() else "cpu")
            dist.broadcast(path_tensor, src=0)

            # 将张量转换回字符串
            path_bytes = path_tensor.cpu().numpy().astype('uint8')
            resume_from_checkpoint = bytes(path_bytes[path_bytes != 0]).decode('utf-8', errors='ignore')
        elif resume_flag.item() == 1 and rank == 0:
            # 主进程发送路径
            max_path_len = 512
            path_bytes = list(resume_from_checkpoint.encode('utf-8'))
            path_tensor = torch.zeros(max_path_len, dtype=torch.uint8, device=f"cuda:{gpu}" if torch.cuda.is_available() and gpu < torch.cuda.device_count() else "cpu")
            path_tensor[:len(path_bytes)] = torch.tensor(path_bytes)
            dist.broadcast(path_tensor, src=0)
    else:
        # 非分布式训练，直接使用local_rank判断
        if local_rank == 0:
            # 已经在上面设置了resume_from_checkpoint
            pass
        else:
            # 非分布式训练下，其他local_rank不会存在，所以不需要处理
            resume_from_checkpoint = None

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

    # 如果是从检查点恢复，则加载检查点中的模型
    if resume_from_checkpoint is not None and config['finetune_method'] == 'lora':
        print(f"从检查点 {resume_from_checkpoint} 加载模型")
        model = AutoModelForImageTextToText.from_pretrained(
            resume_from_checkpoint,
            torch_dtype=dtype,
            device_map=f"cuda:{local_rank}",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
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

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    print("模型加载完成")
    print(f"模型参数数量: {model.num_parameters():,}")

    # 准备数据集
    print("准备数据集...")
    full_dataset = prepare_dataset(config)

    # 划分训练集和验证集 (根据配置文件)
    split_dataset = full_dataset.train_test_split(test_size=config['validation_split'])
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    print(f"训练集准备完成，共 {len(train_dataset)} 个样本")
    print(f"验证集准备完成，共 {len(eval_dataset)} 个样本")

    # 保存验证集到指定路径，只在主进程(rank 0)中执行
    if local_rank == 0:
        test_set_path = "./data/vlm_finetune_dataset_fixed/testset.json"
        os.makedirs(os.path.dirname(test_set_path), exist_ok=True)

        # 将验证集转换为列表格式并保存
        eval_data_list = [eval_dataset[i] for i in range(len(eval_dataset))]
        with open(test_set_path, 'w', encoding='utf-8') as f:
            json.dump(eval_data_list, f, ensure_ascii=False, indent=2)
        print(f"验证集已保存到 {test_set_path}")

    # 根据配置选择微调方法
    if config['finetune_method'] == 'lora':
        # 如果不是从检查点恢复，则配置LoRA
        if resume_from_checkpoint is None:
            # 配置LoRA - 使用配置文件中的参数
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config['lora_r'],
                lora_alpha=config['lora_alpha'],
                lora_dropout=config['lora_dropout'],
                target_modules=config['target_modules']
            )
            # 应用LoRA配置
            model = get_peft_model(model, peft_config)
        # 只有在LoRA模式下才有print_trainable_parameters方法
        model.print_trainable_parameters()
    elif config['finetune_method'] == 'full':
        # 全量微调 - 不应用LoRA，直接使用原始模型
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

    # 在模型创建之后、训练之前处理DataParallel/DDP问题
    if n_gpu > 1 and not dist.is_initialized():
        print(f"检测到 {n_gpu} 个GPU，使用 DataParallel")
        # 为DataParallel模型添加缺失的方法
        model = torch.nn.DataParallel(model)

        # 为DataParallel模型添加必要的方法以满足Trainer要求
        def gradient_checkpointing_enable(**kwargs):
            model.module.gradient_checkpointing_enable(**kwargs)
        model.gradient_checkpointing_enable = gradient_checkpointing_enable

        def enable_input_require_grads():
            model.module.enable_input_require_grads()
        model.enable_input_require_grads = enable_input_require_grads

        def disable_input_require_grads():
            model.module.disable_input_require_grads()
        model.disable_input_require_grads = disable_input_require_grads

        # 只有在LoRA模式下才访问peft模型的方法
        if config['finetune_method'] == 'lora':
            model.module.print_trainable_parameters()
    elif dist.is_initialized():  # 使用分布式数据并行训练
        print(f"初始化分布式训练，world_size: {world_size}, rank: {local_rank}")
        model = DDP(model, device_ids=[local_rank])

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

        # 只有在LoRA模式下才访问peft模型的方法
        if config['finetune_method'] == 'lora':
            model.module.print_trainable_parameters()
    elif dist.is_initialized():  # 使用分布式数据并行训练
        print(f"初始化分布式训练，world_size: {world_size}, rank: {local_rank}")
        model = DDP(model, device_ids=[local_rank])

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

        # 只有在LoRA模式下才访问peft模型的方法
        if config['finetune_method'] == 'lora':
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
        data_collator=DataCollatorForQwen3VL(processor),
        processing_class=processor,
    )

    # 开始训练，如果存在检查点则从中恢复
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 在适当的时候保存模型 - 对于分布式训练，只有rank 0保存
    print("正在保存模型...")

    # 对于分布式训练，通常只需要主进程(rank 0)保存
    if dist.is_initialized():
        if dist.get_rank() == 0:
            trainer.save_model()
            processor.save_pretrained(config['output_dir'])
            print(f"模型微调完成，已保存到 {config['output_dir']}")
    else:
        # 非分布式情况下
        trainer.save_model()
        processor.save_pretrained(config['output_dir'])
        print(f"模型微调完成，已保存到 {config['output_dir']}")

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