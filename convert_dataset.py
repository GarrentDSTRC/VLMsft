#!/usr/bin/env python
"""
将数据集转换为LlamaFactory可接受的格式
"""
import json
import os
import re

def clean_text(text):
    """清理文本中的控制字符等非法字符"""
    if not isinstance(text, str):
        return text

    # 移除控制字符 (ASCII 0-31 除了制表符、换行符和回车符)
    cleaned = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')

    # 替换多余的空白字符
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned.strip()

def clean_json_data(data):
    """递归清理JSON数据"""
    if isinstance(data, dict):
        return {key: clean_json_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data]
    elif isinstance(data, str):
        return clean_text(data)
    else:
        return data

# 尝试读取原始数据并处理可能的格式问题
try:
    with open('/mnt/workspace/Alldata.json', 'r', encoding='utf-8') as f:
        original_data = json.load(f)
except json.JSONDecodeError as e:
    print(f"原始JSON格式错误: {e}")
    # 尝试从原始内容修复
    with open('/mnt/workspace/Alldata.json', 'r', encoding='utf-8', errors='ignore') as f:
        raw_data = f.read()

    # 移除控制字符
    cleaned_raw = ''.join(char for char in raw_data if ord(char) >= 32 or char in '\t\n\r ')
    original_data = json.loads(cleaned_raw)

# 递归清理整个数据结构
original_data = clean_json_data(original_data)

# 转换数据格式
converted_data = []
for i, item in enumerate(original_data):
    # 提取conversations字段
    conversations = item.get("conversations", [])

    # 构建新的条目
    converted_entry = {
        "id": item.get("id", f"sample_{i}"),
        "conversations": []
    }

    for conv in conversations:
        role = conv.get("role", "")
        content_raw = conv.get("content", "")

        # 处理content，可能是字符串也可能是列表
        content_str = ""
        images = []  # 存储图像

        if isinstance(content_raw, list):
            # 处理列表格式的content
            for content_item in content_raw:
                content_type = content_item.get("type", "")
                if content_type == "text":
                    text_content = content_item.get("text", "")
                    content_str += text_content
                elif content_type == "image":
                    # 添加图像标签并收集图像
                    content_str += "<image>"  # 这是Qwen模型的图像标记
                    images.append(content_item.get("image", ""))
        else:
            # 处理字符串格式的content
            content_str = str(content_raw)

        converted_conv = {
            "role": role,
            "content": content_str
        }
        if images:
            converted_conv["images"] = images

        converted_entry["conversations"].append(converted_conv)

    converted_data.append(converted_entry)

# 确保目录存在
os.makedirs('/mnt/workspace/data/vlm_finetune_dataset_fixed', exist_ok=True)

# 保存转换后的数据
output_file = '/mnt/workspace/data/vlm_finetune_dataset_fixed/train_0.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print(f"已将数据转换并保存到: {output_file}")
print(f"总共转换了 {len(converted_data)} 个样本")
