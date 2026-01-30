#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 Alpaca 格式的数据转换为聊天格式
Alpaca 格式包含: instruction, input, output, images
聊天格式包含: messages, images
"""

import json
import argparse
import re
from typing import Dict, List


def alpaca_to_chat_format(alpaca_data: List[Dict]) -> List[Dict]:
    """
    将 Alpaca 格式的数据转换为聊天格式

    Args:
        alpaca_data: Alpaca 格式的列表，每个元素包含 instruction, input, output, images

    Returns:
        聊天格式的列表，每个元素包含 messages, images
    """
    chat_data = []

    for item in alpaca_data:
        # 提取各个字段
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        images = item.get("images", [])

        # 构建消息列表
        messages = []

        # 如果有输入，则组合 instruction 和 input 作为用户消息
        if input_text.strip():
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        # 添加用户消息
        if user_content.strip():
            messages.append({
                "role": "user",
                "content": user_content
            })

        # 处理输出，如果输出是字典则转换为 JSON 字符串
        if isinstance(output_text, dict):
            output_text = json.dumps(output_text, ensure_ascii=False)
        elif isinstance(output_text, str) and output_text.strip().startswith('```json'):
            # 如果输出是包含 JSON 的代码块，提取其中的 JSON 部分
            json_match = re.search(r'```json\s*\n(.*?)\n```', output_text, re.DOTALL)
            if json_match:
                try:
                    json_obj = json.loads(json_match.group(1))
                    output_text = json.dumps(json_obj, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass  # 如果解析失败，保留原始内容

        if output_text and str(output_text).strip():
            messages.append({
                "role": "assistant",
                "content": str(output_text)
            })

        # 创建聊天格式的数据项
        chat_item = {
            "messages": messages,
            "images": images
        }

        chat_data.append(chat_item)

    return chat_data


def main():
    parser = argparse.ArgumentParser(description="Convert Alpaca format to ChatML format")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input file in Alpaca format (JSON)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output file in ChatML format (JSONL)")

    args = parser.parse_args()

    # 读取 Alpaca 格式的数据
    with open(args.input, 'r', encoding='utf-8') as f:
        alpaca_data = json.load(f)

    # 转换格式
    chat_data = alpaca_to_chat_format(alpaca_data)

    # 写入聊天格式的数据
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in chat_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Successfully converted {len(chat_data)} items from {args.input} to {args.output}")


if __name__ == "__main__":
    main()