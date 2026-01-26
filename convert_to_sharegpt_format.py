#!/usr/bin/env python
"""
将Alldata.json数据集转换为llama-factory支持的格式
并将base64编码的图片转换为PNG文件保存到data/vlm_finetune_dataset_fixed目录
"""

import json
import os
import base64
from PIL import Image
from io import BytesIO


def base64_to_image(base64_str):
    """将base64字符串转换为PIL Image对象"""
    try:
        # 如果base64字符串包含data:image前缀，则去除它
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # 解码base64字符串
        image_bytes = base64.b64decode(base64_str)
        
        # 创建PIL Image对象
        image = Image.open(BytesIO(image_bytes))
        
        return image
    except Exception as e:
        print(f"转换base64到图片时出错: {e}")
        return None


def save_base64_image(base64_str, image_path):
    """将base64字符串保存为PNG图片"""
    image = base64_to_image(base64_str)
    if image:
        # 确保目录存在
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # 保存为PNG格式
        image.save(image_path, 'PNG')
        return True
    else:
        return False


def convert_dataset(input_file, output_dir):
    """转换数据集格式"""
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图片存储目录
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # 转换数据
    converted_data = []
    
    for idx, item in enumerate(original_data):
        # 获取ID
        sample_id = item.get('id', f'sample_{idx}')
        
        # 初始化转换后的对话
        conversations = []
        
        # 遍历原始对话
        for conv in item.get('conversations', []):
            role = conv.get('role', '')
            content_list = conv.get('content', [])
            
            # 如果content是字符串，直接使用
            if isinstance(content_list, str):
                new_conv = {
                    'role': role,
                    'content': content_list
                }
                conversations.append(new_conv)
                continue
            
            # 如果content是列表，需要处理其中的图片和文本
            content_parts = []
            images = []
            
            for content_item in content_list:
                content_type = content_item.get('type', '')
                
                if content_type == 'text':
                    text = content_item.get('text', '')
                    content_parts.append(text)
                
                elif content_type == 'image':
                    # 处理图片
                    base64_img = content_item.get('image', '')
                    
                    # 生成图片文件名
                    img_filename = f"{sample_id}_{len(images)}.png"
                    img_path = os.path.join(images_dir, img_filename)
                    
                    # 保存图片
                    if save_base64_image(base64_img, img_path):
                        images.append(f"./data/images/{img_filename}")
                        content_parts.append("<image>")  # llama-factory使用的图片占位符
                    else:
                        print(f"无法保存图片: {img_filename}")
            
            # 合并所有文本内容
            full_content = "".join(content_parts)
            
            # 创建新的对话项
            new_conv = {
                'role': role,
                'content': full_content
            }
            
            # 如果有图片，添加到对话项中
            if images:
                new_conv['images'] = images
            
            conversations.append(new_conv)
        
        # 创建转换后的数据项
        converted_item = {
            'id': sample_id,
            'conversations': conversations
        }
        
        converted_data.append(converted_item)
    
    # 保存转换后的数据
    output_file = os.path.join(output_dir, 'train_0.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据转换完成！")
    print(f"转换了 {len(converted_data)} 个样本")
    print(f"输出文件: {output_file}")
    
    # 统计图片数量
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    print(f"保存了 {len(image_files)} 张图片到 {images_dir}")


if __name__ == "__main__":
    # 输入文件路径
    input_file = "Alldata.json"
    
    # 输出目录
    output_dir = "data/vlm_finetune_dataset_fixed"
    
    # 执行转换
    convert_dataset(input_file, output_dir)