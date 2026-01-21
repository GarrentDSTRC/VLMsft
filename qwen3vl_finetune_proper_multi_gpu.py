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

    # 获取GPU数量
    n_gpu = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

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

    # 根据配置选择微调方法
    if config['finetune_method'] == 'lora':
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
        output_dir=config['output_dir'],
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

    # 开始训练
    trainer.train()

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