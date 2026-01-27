#!/usr/bin/env python
"""
Qwen3-VL-2B-Instruct 模型微调效果评估脚本

此脚本用于测试微调前后模型在测试集上的表现对比
"""
import json
import torch
import datetime
import os
from datasets import Dataset
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from peft import PeftModel
from PIL import Image
import base64
from io import BytesIO
import re
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np
import yaml

# 确保日志目录存在
os.makedirs('./logs', exist_ok=True)

def load_config(config_path="./config/evaluation_config.yaml"):
    """加载评估配置文件"""
    # 如果配置文件不存在，使用默认设置
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认设置")
        return {
            'test_dataset_path': './vlm_finetune_dataset.json',
            'base_model_path': '/root/.cache/modelscope/hub/models/qwen/Qwen3-VL-2B-Instruct',
            'results_output_dir': './logs/',
            'max_test_samples': 60,
            'max_preview_samples': 3,
            'max_tokens': 256,
            'temperature': 0.1
        }

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_test_data(config):
    """加载测试数据集 - 适配Llama Factory格式"""
    print(f"从 {config['test_dataset_path']} 加载测试数据...")

    try:
        # 尝试加载新的Llama Factory格式数据
        with open(config['test_dataset_path'], 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("警告: 找不到指定的测试数据集，尝试加载action_data.json...")
        try:
            with open("./action_data.json", 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print("错误: 找不到action_data.json")
            return []

    # 处理Llama Factory格式的数据
    test_data = []
    for item in raw_data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', {})

        # 构建问题 - 将instruction和input组合
        question = instruction + "\n\n" + input_text if input_text else instruction

        # 处理输出 - output可能是字典格式
        if isinstance(output, dict):
            if 'reasoning' in output and 'action' in output:
                # 按照JSON格式构建答案
                answer = json.dumps(output, ensure_ascii=False, indent=2)
            else:
                # 如果只有部分字段，尝试构建答案
                answer_parts = []
                for key, value in output.items():
                    if isinstance(value, list):
                        answer_parts.append(f"{key}: {str(value)}")
                    else:
                        answer_parts.append(f"{key}: {value}")
                answer = "\n".join(answer_parts)
        else:
            answer = str(output)

        # 获取图像路径
        image_paths = item.get('images', [])

        if question and answer:
            test_data.append({
                'id': item.get('id', f'test_{len(test_data)}'),
                'question': question.strip(),
                'answer': answer.strip(),
                'images': image_paths  # 保存图像路径信息
            })

    print(f"加载了 {len(test_data)} 个测试样本")
    return test_data

def load_model_for_evaluation(model_path, adapter_path=None, device="cuda"):
    """
    智能加载模型：自动识别LoRA/全量模型
    """
    # 检测模型类型
    is_lora = (os.path.exists(os.path.join(model_path, "adapter_model.bin")) or
               os.path.exists(os.path.join(model_path, "adapter_model.safetensors"))) and \
              os.path.exists(os.path.join(model_path, "adapter_config.json"))

    # 检测全量微调模型
    is_full = (os.path.exists(os.path.join(model_path, "pytorch_model.bin")) or
               os.path.exists(os.path.join(model_path, "model.safetensors"))) and \
              os.path.exists(os.path.join(model_path, "config.json"))

    if is_lora or adapter_path:
        # LoRA模式：需指定基础模型
        if adapter_path:
            # 如果提供了adapter_path，使用它作为LoRA路径
            adapter_dir = adapter_path
        else:
            # 否则使用model_path作为LoRA路径
            adapter_dir = model_path

        # 从adapter_config.json中获取基础模型名称，如果不存在则使用默认值
        adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r', encoding='utf-8') as f:
                adapter_config = json.load(f)
            base_model = adapter_config.get("base_model_name_or_path", "/root/.cache/modelscope/hub/models/qwen/Qwen3-VL-2B-Instruct")
        else:
            base_model = "/root/.cache/modelscope/hub/models/qwen/Qwen3-VL-2B-Instruct"

        print(f"→ 检测到LoRA模型，加载基础模型: {base_model}")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )

        # 加载适配器
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        print(f"✓ LoRA适配器加载成功: {adapter_dir}")

    elif is_full:
        # 全量模型：直接加载
        print(f"→ 加载全量微调模型: {model_path}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
    else:
        # 如果都不是，尝试作为基础模型加载
        print(f"→ 尝试作为基础模型加载: {model_path}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )

    model.eval()
    # 确定processor的路径 - 如果是LoRA，可能需要从基础模型路径加载
    if is_lora and not os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        # 如果是LoRA且目标路径缺少tokenizer配置，从基础模型加载
        processor = Qwen3VLProcessor.from_pretrained(
            base_model if 'base_model' in locals() else model_path,
            trust_remote_code=True
        )
    else:
        # 否则从目标路径加载
        processor = Qwen3VLProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

    return model, processor


def prepare_model(config, is_finetuned=False):
    """准备模型 - 基础模型或微调模型"""
    model_name = config['base_model_path']

    # 确定使用的设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        # 使用特定的GPU设备而不是auto，以避免多GPU设备不匹配问题
        device = torch.device('cuda:0')  # 强制使用第一块GPU

    if is_finetuned:
        print("准备微调后的模型...")

        # 从配置文件获取模型搜索路径
        model_paths = config.get('finetuned_model_paths', [
            "./qwen3-vl-2b-instruct-full",          # 全量微调模型路径
            "./qwen3-vl-2b-instruct-lora",          # LoRA单卡模型路径
            "./qwen3-vl-2b-instruct-lora-multigpu", # LoRA多卡模型路径
            "./logs/qwen3-vl-2b-instruct-lora",      # LoRA单卡模型路径(日志)
            "./logs/qwen3-vl-2b-instruct-lora-multigpu"  # LoRA多卡模型路径(日志)
        ])

        loaded_model = False
        model = None
        processor = None

        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"发现模型路径: {model_path}")

                try:
                    # 首先尝试直接加载
                    model, processor = load_model_for_evaluation(model_path, device=str(device))
                    loaded_model = True
                    print(f"成功加载模型: {model_path}")
                    break
                except Exception as e:
                    print(f"直接加载模型失败 {model_path}: {e}")

                    # 如果直接加载失败，尝试搜索子目录中的检查点
                    if os.path.isdir(model_path):
                        subdirs = [d for d in os.listdir(model_path)
                                  if os.path.isdir(os.path.join(model_path, d))]

                        # 按名称排序，优先尝试checkpoint-*目录
                        subdirs_sorted = sorted(subdirs,
                                              key=lambda x: (not x.startswith('checkpoint-'), x))

                        for subdir in subdirs_sorted:
                            subdir_path = os.path.join(model_path, subdir)
                            print(f"尝试子目录: {subdir_path}")

                            try:
                                model, processor = load_model_for_evaluation(subdir_path, device=str(device))
                                loaded_model = True
                                print(f"成功从子目录加载模型: {subdir_path}")
                                break
                            except Exception as sub_e:
                                print(f"子目录加载失败 {subdir_path}: {sub_e}")
                                continue

                    if loaded_model:
                        break

        if not loaded_model:
            print(f"警告: 未找到有效的微调模型路径")
            # 如果没找到微调模型，返回基础模型
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            processor = Qwen3VLProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
    else:
        print("准备基础模型...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,  # 使用指定的单一设备
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        processor = Qwen3VLProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    return model, processor

def evaluate_model(model, processor, test_data, config, model_name="模型", log_details=None):
    """评估模型性能"""
    print(f"\n开始评估 {model_name}...")

    correct_predictions = 0
    total_samples = 0
    predictions = []
    references = []
    detailed_results = []  # 存储详细结果

    # 限制测试样本数量以节省时间
    test_samples = test_data[:min(config.get('max_test_samples', 60), len(test_data))]

    for i, item in enumerate(test_samples):
        print(f"处理测试样本 {i+1}/{len(test_samples)}", end='', flush=True)

        try:
            question = item.get('question', '')
            expected_answer = item.get('answer', '')
            sample_id = item.get('id', f'sample_{i}')

            if not question or not expected_answer:
                print(" (跳过 - 缺少问题或答案)")
                continue

            # 构建消息格式
            # 检查是否有图像信息
            import base64
            from PIL import Image
            import io

            if 'images' in item and item['images']:
                # 包含图像的消息格式
                image_path = item['images'][0]  # 假设只处理第一个图像

                # 检查图像路径是本地文件还是base64
                if image_path.startswith('data:image'):
                    # 处理base64图像
                    base64_str = image_path.split(',')[1]
                    image_bytes = base64.b64decode(base64_str)
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                else:
                    # 处理本地文件路径
                    # 如果是相对路径，尝试在几个可能的位置查找
                    import os
                    possible_paths = [
                        image_path,  # 原始路径
                        os.path.join(os.path.dirname(config['test_dataset_path']), image_path),  # 相对于数据集文件
                        os.path.join('.', image_path),  # 相对当前目录
                        os.path.join('..', image_path),  # 上级目录
                    ]

                    pil_image = None
                    for img_path in possible_paths:
                        if os.path.exists(img_path):
                            pil_image = Image.open(img_path).convert('RGB')
                            break

                    if pil_image is None:
                        print(f"警告: 无法找到图像文件 {image_path}")
                        # 如果找不到图像，按纯文本处理
                        messages = [
                            {
                                "role": "user",
                                "content": question
                            }
                        ]

                        # 应用对话模板
                        text = processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )

                        # 准备输入
                        inputs = processor(
                            text=text,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=2048
                        )
                    else:
                        # 使用messages格式，包含图像标记，然后应用模板
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": question}
                                ]
                            }
                        ]

                        # 应用对话模板
                        text = processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )

                        # 使用processor处理文本和图像
                        inputs = processor(
                            text=text,
                            images=[pil_image],
                            return_tensors="pt"
                        )
            else:
                # 纯文本消息格式
                messages = [
                    {
                        "role": "user",
                        "content": question
                    }
                ]

                # 应用对话模板
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # 准备输入
                inputs = processor(
                    text=text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )

            # 确保所有输入张量都在同一设备上
            device = next(model.parameters()).device  # 获取模型参数所在的设备
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 设置生成配置
            model.generation_config.max_new_tokens = config.get('max_tokens', 256)
            model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

            # 生成预测
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs
                )

            # 解码生成结果
            generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]

            # 确保解码在CPU上进行以避免设备不匹配
            generated_ids_trimmed = generated_ids_trimmed.cpu()
            predicted_answer = processor.tokenizer.decode(
                generated_ids_trimmed[0],
                skip_special_tokens=True
            ).strip()

            # 检查预测是否与期望答案匹配
            expected_clean = re.sub(r'[^\w\s]', '', expected_answer.lower().strip())
            predicted_clean = re.sub(r'[^\w\s]', '', predicted_answer.lower().strip())

            # 多种匹配策略
            is_correct = False
            if expected_clean == predicted_clean:
                is_correct = True
            elif expected_clean in predicted_clean or predicted_clean in expected_clean:
                is_correct = True
            else:
                # 关键词匹配
                expected_words = set(expected_clean.split())
                predicted_words = set(predicted_clean.split())

                if len(expected_words) > 0:
                    common_words = expected_words.intersection(predicted_words)
                    if len(common_words) / len(expected_words) >= 0.8:  # 60%的关键词匹配
                        is_correct = True

            if is_correct:
                correct_predictions += 1
                status = "✓"
            else:
                status = "✗"

            total_samples += 1

            predictions.append(predicted_answer)
            references.append(expected_answer)

            # 记录详细结果
            detailed_result = {
                'sample_id': sample_id,
                'question': question,
                'expected_answer': expected_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'status': status
            }
            detailed_results.append(detailed_result)

            print(f" {status}")

        except Exception as e:
            print(f" 错误: {e}")
            # 即使出错也记录该样本的结果
            detailed_result = {
                'sample_id': f'sample_{i}',
                'question': item.get('question', ''),
                'expected_answer': item.get('answer', ''),
                'predicted_answer': f'ERROR: {str(e)}',
                'is_correct': False,
                'status': '✗'
            }
            detailed_results.append(detailed_result)
            continue

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    print(f"\n{model_name} 评估完成")
    print(f"准确率: {correct_predictions}/{total_samples} ({accuracy:.4f})")

    return accuracy, predictions, references, detailed_results

def main():
    """主函数"""
    print("="*70)
    print("Qwen3-VL-2B-Instruct 模型微调效果评估")
    print("="*70)

    # 加载配置
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./config/evaluation_config.yaml"
    config = load_config(config_path)
    print(f"使用评估配置: {config_path}")

    # 加载测试数据
    test_data = load_test_data(config)

    if len(test_data) == 0:
        print("错误: 测试数据为空")
        return

    print(f"使用 {min(config.get('max_test_samples', 60), len(test_data))} 个样本进行评估")

    # 评估微调模型
    print("\n" + "-"*60)
    print("评估微调模型 (Qwen3-VL-2B-Instruct-LoRA)...")
    ft_model, ft_processor = prepare_model(config, is_finetuned=True)
    ft_acc, ft_preds, ft_refs, ft_detailed_results = evaluate_model(ft_model, ft_processor, test_data, config, "微调模型")

    # 评估基础模型
    print("\n" + "-"*60)
    print("评估基础模型 (Qwen3-VL-2B-Instruct)...")
    base_model, base_processor = prepare_model(config, is_finetuned=False)
    base_acc, base_preds, base_refs, base_detailed_results = evaluate_model(base_model, base_processor, test_data, config, "基础模型")

    # 计算改进
    acc_improvement = ft_acc - base_acc
    if base_acc != 0:
        improvement_percentage = (acc_improvement / base_acc) * 100
    else:
        improvement_percentage = float('inf') if acc_improvement > 0 else float('-inf')

    # 显示比较结果
    print("\n" + "="*70)
    print("微调效果比较结果")
    print("="*70)
    print(f"测试样本数: {min(60, len(test_data))}")
    print()
    print("微调模型:")
    print(f"  准确率: {ft_acc:.4f}")
    print()
    print("基础模型:")
    print(f"  准确率: {base_acc:.4f}")
    print()
    print("性能改进:")
    print(f"  绝对改进: {acc_improvement:+.4f}")
    print(f"  相对改进: {improvement_percentage:+.2f}%")

    # 分析改进情况
    print()
    if acc_improvement > 0.01:  # 超过1%的改进
        print("🎉 微调显著提升了模型性能!")
    elif acc_improvement > 0:
        print("✅ 微调对模型有轻微改进")
    elif acc_improvement == 0:
        print("→ 微调对模型性能无影响")
    else:
        print("⚠️  微调后模型性能略有下降")

    # 显示详细结果示例
    print("\n" + "-"*70)
    print("结果对比示例 (前3个样本):")
    print("-"*70)
    for i in range(min(3, len(test_data))):
        if i < len(base_preds) and i < len(ft_preds):
            print(f"样本 {i+1}:")
            print(f"  问题: {test_data[i]['question'][:100]}...")
            print(f"  答案: {test_data[i]['answer'][:100]}")
            print(f"  微调: {ft_preds[i][:100]}")
            print(f"  基础: {base_preds[i][:100]}")
            print()

    # 创建完整的对比结果
    comparison_results = []
    for i in range(len(base_detailed_results)):
        if i < len(ft_detailed_results):
            comparison_entry = {
                'sample_id': base_detailed_results[i]['sample_id'],
                'question': base_detailed_results[i]['question'],
                'expected_answer': base_detailed_results[i]['expected_answer'],
                'ft_prediction': ft_detailed_results[i]['predicted_answer'],
                'ft_is_correct': ft_detailed_results[i]['is_correct'],
                'ft_status': ft_detailed_results[i]['status'],
                'base_prediction': base_detailed_results[i]['predicted_answer'],
                'base_is_correct': base_detailed_results[i]['is_correct'],
                'base_status': base_detailed_results[i]['status'],
                'improved': ft_detailed_results[i]['is_correct'] and not base_detailed_results[i]['is_correct'],
                'regressed': not ft_detailed_results[i]['is_correct'] and base_detailed_results[i]['is_correct']
            }
            comparison_results.append(comparison_entry)

    # 保存评估结果
    evaluation_results = {
        'test_samples_count': min(config.get('max_test_samples', 60), len(test_data)),
        'fine_tuned_model_accuracy': ft_acc,
        'base_model_accuracy': base_acc,
        'absolute_improvement': acc_improvement,
        'relative_improvement_percent': improvement_percentage,
        'timestamp': str(datetime.datetime.now()),
        'test_samples_preview': [
            {
                'question': test_data[i]['question'][:200],
                'expected': test_data[i]['answer'][:200] if i < len(test_data) else '',
                'fine_tuned_prediction': ft_preds[i][:200] if i < len(ft_preds) else '',
                'base_prediction': base_preds[i][:200] if i < len(base_preds) else ''
            }
            for i in range(min(config.get('max_preview_samples', 3), len(test_data)))
        ],
        'detailed_comparison_results': comparison_results,
        'fine_tuned_model_detailed_results': ft_detailed_results,
        'base_model_detailed_results': base_detailed_results
    }

    # 保存结果到JSON文件
    result_file = config['results_output_dir'] + "qwen3_vl_finetuning_evaluation.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    # 保存详细的文本日志
    log_file = config['results_output_dir'] + "fine_tuning_evaluation_final.log"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Qwen3-VL-2B-Instruct 模型微调效果评估详细日志\n")
        f.write("="*70 + "\n")
        f.write(f"评估时间: {datetime.datetime.now()}\n")
        f.write(f"测试样本数: {min(config.get('max_test_samples', 60), len(test_data))}\n")
        f.write(f"微调模型准确率: {ft_acc:.4f}\n")
        f.write(f"基础模型准确率: {base_acc:.4f}\n")
        f.write(f"绝对改进: {acc_improvement:+.4f}\n")
        f.write(f"相对改进: {improvement_percentage:+.2f}%\n")

        f.write("\n" + "="*70 + "\n")
        f.write("详细对比结果:\n")
        f.write("="*70 + "\n")

        improved_count = 0
        regressed_count = 0
        same_count = 0

        for i, result in enumerate(comparison_results):
            f.write(f"样本 {i+1} (ID: {result['sample_id']}):\n")
            f.write(f"  问题: {result['question']}\n")
            f.write(f"  标准答案: {result['expected_answer']}\n")
            f.write(f"  微调模型预测: {result['ft_prediction']} [{result['ft_status']}]\n")
            f.write(f"  基础模型预测: {result['base_prediction']} [{result['base_status']}]\n")

            if result['improved']:
                f.write(f"  结果: ✅ 微调模型改进\n")
                improved_count += 1
            elif result['regressed']:
                f.write(f"  结果: ❌ 微调模型退步\n")
                regressed_count += 1
            else:
                f.write(f"  结果: ↔️  无变化\n")
                same_count += 1

            f.write("\n")

        f.write("="*70 + "\n")
        f.write("统计摘要:\n")
        f.write("="*70 + "\n")
        f.write(f"总样本数: {len(comparison_results)}\n")
        f.write(f"微调改进样本数: {improved_count}\n")
        f.write(f"微调退步样本数: {regressed_count}\n")
        f.write(f"无变化样本数: {same_count}\n")
        f.write(f"微调模型准确率: {ft_acc:.4f}\n")
        f.write(f"基础模型准确率: {base_acc:.4f}\n")
        f.write(f"准确率提升: {acc_improvement:+.4f}\n")

    print(f"详细评估结果已保存到: {result_file}")
    print(f"详细对比日志已保存到: {log_file}")

    print("\n" + "="*70)
    print("评估完成!")
    print("="*70)

if __name__ == "__main__":
    main()