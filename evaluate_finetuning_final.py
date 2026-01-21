#!/usr/bin/env python
"""
Qwen3-VL-2B-Instruct 模型微调效果评估脚本

此脚本用于测试微调前后模型在测试集上的表现对比
"""
import json
import torch
import datetime
from datasets import Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import os
from PIL import Image
import base64
from io import BytesIO
import re
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np
import os
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
    """加载测试数据集"""
    print(f"从 {config['test_dataset_path']} 加载测试数据...")

    try:
        #with open('./vlm_test_dataset.json', 'r', encoding='utf-8') as f:
        with open(config['test_dataset_path'], 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("警告: 找不到vlm_test_dataset.json，尝试使用训练数据的最后部分作为测试集...")
        backup_path = config['test_dataset_path'].replace('.json', '_fixed.json')
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"错误: 找不到备用数据集文件 {backup_path}")
            return []
        # 使用最后10%作为测试集
        split_idx = int(0.9 * len(raw_data))
        raw_data = raw_data[split_idx:]

    # 简单的数据提取 - 从conversations字段中提取问题和答案
    test_data = []
    for item in raw_data:
        conversations = item.get('conversations', [])
        if not conversations or len(conversations) < 0:
            continue

        # 提取第一个用户问题和对应助手回答
        user_msg = ""
        assistant_msg = ""

        for conv in conversations:
            role = conv.get('role', '')
            content = conv.get('content', [])  # content可能是字符串或列表

            if role == 'user':
                if isinstance(content, str):
                    user_msg = content
                elif isinstance(content, list):
                    for content_item in content:
                        if isinstance(content_item, dict):
                            content_type = content_item.get('type', '')
                            if content_type == 'text':
                                user_msg += content_item.get('text', '')
                            elif content_type == 'image':
                                # 添加图像标记，实际图像将在模型推理时处理
                                user_msg += " [IMAGE]"
            elif role == 'assistant':
                if isinstance(content, str):
                    assistant_msg = content
                elif isinstance(content, list):
                    for content_item in content:
                        if isinstance(content_item, dict):
                            content_type = content_item.get('type', '')
                            if content_type == 'text':
                                assistant_msg += content_item.get('text', '')

        if user_msg and assistant_msg:
            test_data.append({
                'id': item.get('id', f'test_{len(test_data)}'),
                'question': user_msg.strip(),
                'answer': assistant_msg.strip(),
            })

    print(f"加载了 {len(test_data)} 个测试样本")
    return test_data

def prepare_model(config, is_finetuned=False):
    """准备模型 - 基础模型或微调模型"""
    model_name = config['base_model_path']

    if is_finetuned:
        print("准备微调后的模型...")

        # 加载基础模型
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # 尝试加载微调模型（可能是LoRA或全量微调）
        try:
            from peft import PeftModel
            import json

            # 从配置文件获取模型搜索路径
            model_paths = config.get('finetuned_model_paths', [
                "./qwen3-vl-2b-instruct-full",          # 全量微调模型路径
                "./qwen3-vl-2b-instruct-lora",          # LoRA单卡模型路径
                "./qwen3-vl-2b-instruct-lora-multigpu", # LoRA多卡模型路径
                "./logs/qwen3-vl-2b-instruct-lora",      # LoRA单卡模型路径(日志)
                "./logs/qwen3-vl-2b-instruct-lora-multigpu"  # LoRA多卡模型路径(日志)
            ])

            loaded_model = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    print(f"发现模型路径: {model_path}")

                    # 检查是否为全量微调模型
                    config_path = os.path.join(model_path, "config.json")
                    if os.path.exists(config_path):
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)

                        # 检查配置中是否有adapter相关的信息来判断模型类型
                        has_adapter_info = any(key in config for key in ["peft_type", "adapter_config", "adapters"])

                        if has_adapter_info:
                            # 这是一个LoRA模型
                            print(f"加载LoRA适配器: {model_path}")
                            model = PeftModel.from_pretrained(model, model_path)
                            model = model.merge_and_unload()  # 合并LoRA权重进行评估
                            print("LoRA适配器已合并到基础模型中")
                            loaded_model = True
                            break
                        else:
                            # 这可能是一个全量微调模型
                            print(f"加载全量微调模型: {model_path}")
                            model = AutoModelForVision2Seq.from_pretrained(
                                model_path,
                                torch_dtype=torch.bfloat16,
                                device_map="auto",
                                trust_remote_code=True,
                                low_cpu_mem_usage=True
                            )
                            loaded_model = True
                            break
                    else:
                        # 如果没有配置文件，尝试作为LoRA模型加载
                        try:
                            print(f"尝试作为LoRA模型加载: {model_path}")
                            model = PeftModel.from_pretrained(model, model_path)
                            model = model.merge_and_unload()  # 合并LoRA权重进行评估
                            print("LoRA适配器已合并到基础模型中")
                            loaded_model = True
                            break
                        except:
                            print(f"无法作为LoRA模型加载: {model_path}")
                            continue

            if not loaded_model:
                print(f"警告: 未找到有效的微调模型路径")

        except ImportError:
            print(f"警告: PEFT库未安装或不可用，尝试直接加载模型")
            # 如果无法导入peft，尝试作为完整模型加载
            model_paths = config.get('finetuned_model_paths', [
                "./qwen3-vl-2b-instruct-full",
                "./qwen3-vl-2b-instruct-lora",
                "./qwen3-vl-2b-instruct-lora-multigpu",
                "./logs/qwen3-vl-2b-instruct-lora",
                "./logs/qwen3-vl-2b-instruct-lora-multigpu"
            ])

            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        print(f"直接加载模型: {model_path}")
                        model = AutoModelForVision2Seq.from_pretrained(
                            model_path,
                            torch_dtype=torch.bfloat16,
                            device_map="auto",
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
                        print(f"成功加载模型: {model_path}")
                        break
                    except Exception as e:
                        print(f"加载模型失败 {model_path}: {e}")
                        continue
        except Exception as e:
            print(f"警告: 加载微调模型失败: {e}")
    else:
        print("准备基础模型...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

    processor = AutoProcessor.from_pretrained(
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

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 生成预测
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=config.get('max_tokens', 256),
                    temperature=config.get('temperature', 0.1),
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )

            # 解码生成结果
            generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]
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
                    if len(common_words) / len(expected_words) >= 0.6:  # 60%的关键词匹配
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

    # 评估基础模型
    print("\n" + "-"*60)
    print("评估基础模型 (Qwen3-VL-2B-Instruct)...")
    base_model, base_processor = prepare_model(config, is_finetuned=False)
    base_acc, base_preds, base_refs, base_detailed_results = evaluate_model(base_model, base_processor, test_data, config, "基础模型")

    # 评估微调模型
    print("\n" + "-"*60)
    print("评估微调模型 (Qwen3-VL-2B-Instruct-LoRA)...")
    ft_model, ft_processor = prepare_model(config, is_finetuned=True)
    ft_acc, ft_preds, ft_refs, ft_detailed_results = evaluate_model(ft_model, ft_processor, test_data, config, "微调模型")

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
    print("基础模型:")
    print(f"  准确率: {base_acc:.4f}")
    print()
    print("微调模型:")
    print(f"  准确率: {ft_acc:.4f}")
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
            print(f"  基础: {base_preds[i][:100]}")
            print(f"  微调: {ft_preds[i][:100]}")
            print()

    # 创建完整的对比结果
    comparison_results = []
    for i in range(len(base_detailed_results)):
        if i < len(ft_detailed_results):
            comparison_entry = {
                'sample_id': base_detailed_results[i]['sample_id'],
                'question': base_detailed_results[i]['question'],
                'expected_answer': base_detailed_results[i]['expected_answer'],
                'base_prediction': base_detailed_results[i]['predicted_answer'],
                'base_is_correct': base_detailed_results[i]['is_correct'],
                'base_status': base_detailed_results[i]['status'],
                'ft_prediction': ft_detailed_results[i]['predicted_answer'],
                'ft_is_correct': ft_detailed_results[i]['is_correct'],
                'ft_status': ft_detailed_results[i]['status'],
                'improved': ft_detailed_results[i]['is_correct'] and not base_detailed_results[i]['is_correct'],
                'regressed': not ft_detailed_results[i]['is_correct'] and base_detailed_results[i]['is_correct']
            }
            comparison_results.append(comparison_entry)

    # 保存评估结果
    evaluation_results = {
        'test_samples_count': min(config.get('max_test_samples', 60), len(test_data)),
        'base_model_accuracy': base_acc,
        'fine_tuned_model_accuracy': ft_acc,
        'absolute_improvement': acc_improvement,
        'relative_improvement_percent': improvement_percentage,
        'timestamp': str(datetime.datetime.now()),
        'test_samples_preview': [
            {
                'question': test_data[i]['question'][:200],
                'expected': test_data[i]['answer'][:200] if i < len(test_data) else '',
                'base_prediction': base_preds[i][:200] if i < len(base_preds) else '',
                'fine_tuned_prediction': ft_preds[i][:200] if i < len(ft_preds) else ''
            }
            for i in range(min(config.get('max_preview_samples', 3), len(test_data)))
        ],
        'detailed_comparison_results': comparison_results,
        'base_model_detailed_results': base_detailed_results,
        'fine_tuned_model_detailed_results': ft_detailed_results
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
        f.write(f"基础模型准确率: {base_acc:.4f}\n")
        f.write(f"微调模型准确率: {ft_acc:.4f}\n")
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
            f.write(f"  基础模型预测: {result['base_prediction']} [{result['base_status']}]\n")
            f.write(f"  微调模型预测: {result['ft_prediction']} [{result['ft_status']}]\n")

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
        f.write(f"基础模型准确率: {base_acc:.4f}\n")
        f.write(f"微调模型准确率: {ft_acc:.4f}\n")
        f.write(f"准确率提升: {acc_improvement:+.4f}\n")

    print(f"详细评估结果已保存到: {result_file}")
    print(f"详细对比日志已保存到: {log_file}")

    print("\n" + "="*70)
    print("评估完成!")
    print("="*70)

if __name__ == "__main__":
    main()