#!/usr/bin/env python
"""
使用Qwen3-VL模型进行微调 - 完全正确的数据处理版本
"""
import json
import torch
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
import os
import yaml

def load_config(config_path="./config/single_gpu_config.yaml"):
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

def fine_tune(config_path="./config/single_gpu_config.yaml"):
    """开始微调过程"""
    print("开始加载Qwen3-VL模型...")

    # 加载配置
    config = load_config(config_path)
    print(f"使用配置文件: {config_path}")

    # 检查是否存在检查点，如果有则从中恢复训练
    output_dir = config['output_dir']
    resume_from_checkpoint = None

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

    # 加载模型和处理器 - 使用推荐的类名
    model_name = config['model_name']

    # 检查CUDA是否可用并设置GPU设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        print(f"检测到 {n_gpu} 个GPU设备")

        # 如果有多个GPU，选择第一个
        if n_gpu > 1:
            print(f"注意：检测到多个GPU ({n_gpu})，将使用单个GPU进行训练以避免设备不匹配")
            # 强制使用单个GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        device = torch.device("cpu")
        print("未检测到CUDA设备，将使用CPU进行训练")

    # 根据配置选择模型精度
    dtype = torch.float16 if config['model_dtype'] == 'float16' else torch.bfloat16

    # 使用正确的模型类
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=dtype,  # 使用配置中指定的精度
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    processor = AutoProcessor.from_pretrained(
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

    # 保存验证集到指定路径
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

    # 根据配置选择微调方法
    if config['finetune_method'] == 'lora':
        # 配置LoRA - 使用配置文件中的参数
        from peft import LoraConfig, get_peft_model, TaskType

        # 如果是从检查点恢复，尝试加载PEFT模型
        if resume_from_checkpoint is not None:
            print(f"从检查点 {resume_from_checkpoint} 加载LoRA模型")
            model = AutoModelForImageTextToText.from_pretrained(
                resume_from_checkpoint,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,  # 修正task_type
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
        eval_strategy="steps",  # 现在有了验证集，可以使用steps策略
        save_total_limit=config['save_total_limit'],
        report_to=None,
        remove_unused_columns=False,  # 不删除列，因为我们稍后手动处理
        gradient_checkpointing=config['gradient_checkpointing'],
        fp16=config.get('fp16', False),  # 使用配置中指定的精度
        bf16=config.get('bf16', False),
        logging_dir=config['logging_dir'],
        dataloader_pin_memory=config['dataloader_pin_memory'],
        dataloader_num_workers=config['dataloader_num_workers'],
        max_grad_norm=config['max_grad_norm'],
        ddp_find_unused_parameters=config.get('ddp_find_unused_parameters', False),
        dataloader_drop_last=config['dataloader_drop_last'],
    )

    from transformers import Trainer

    # 自定义数据整理器（Data Collator）来处理Qwen3-VL模型的输入
    class DataCollatorForQwen3VL:
        def __init__(self, processor):
            self.processor = processor

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
                            os.path.join(os.path.dirname(config['dataset_path']), image_path),  # 相对于数据集文件
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
            print(images_list)

            # 处理文本和图像 - 根据是否有图像决定处理方式
            if any(img is not None for img in images_list):
                # 有图像的情况，使用messages格式
                processed_inputs_list = []
                for i, (text, img) in enumerate(zip(texts, images_list)):
                    if img is not None:
                        # 包含图像的消息格式
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": text.split('\n<|assistant|>\n')[0].replace('<|user|>\n', '')}
                                ]
                            }
                        ]

                        # 应用对话模板
                        formatted_text = self.processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )

                        # 处理文本和图像
                        inputs = self.processor(
                            text=formatted_text,
                            images=img,
                            padding=True,
                            truncation=True,
                            max_length=2048,
                            return_tensors="pt"
                        )
                    else:
                        # 纯文本处理
                        inputs = self.processor(
                            text=text,
                            padding=True,
                            truncation=True,
                            max_length=2048,
                            return_tensors="pt"
                        )
                    processed_inputs_list.append(inputs)

                # 合并批次
                batch = {}
                for key in processed_inputs_list[0].keys():
                    batch[key] = torch.cat([inputs[key] for inputs in processed_inputs_list], dim=0)
            else:
                # 纯文本处理
                batch = self.processor(
                    text=texts,
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_tensors="pt"
                )

            # 确保所有张量都在同一设备上
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                # 将所有张量移动到指定的主设备上
                main_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

                # 移动所有张量到主设备
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(main_device)

            # 设置标签
            batch['labels'] = batch['input_ids'].clone()
            return batch

    print("开始训练...")
    print("数据集预处理完成")

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 提供验证集
        data_collator=DataCollatorForQwen3VL(processor),  # 使用自定义数据整理器
        processing_class=processor,  # 使用processing_class替换弃用的tokenizer
    )

    # 开始训练，如果存在检查点则从中恢复
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 保存模型
    trainer.save_model()
    processor.save_pretrained(config['output_dir'])
    print(f"模型微调完成，已保存到 {config['output_dir']}")

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./config/single_gpu_config.yaml"
    fine_tune(config_path=config_path)
