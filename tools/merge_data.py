import json
import os
import shutil
from pathlib import Path

def load_json_file(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, file_path):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def merge_datasets():
    # 主数据文件路径
    main_data_path = "/home/wangzhangyuan/Downloads/data(1)/data/action_data.json"
    
    # 加载主数据集
    main_data = load_json_file(main_data_path)
    print(f"原始数据集中包含 {len(main_data)} 条记录")
    
    # 定义要合并的目录
    dirs_to_merge = [
        ("/home/wangzhangyuan/Downloads/data(1)/T=0.4, 90/T=0.4, 90", "temp_90"),
        ("/home/wangzhangyuan/Downloads/data(1)/T=0.4,20/T=0.4,20", "temp_20"),
        ("/home/wangzhangyuan/Downloads/data(1)/T=0.4,307/qwen,T=0.4,307", "temp_307")
    ]
    
    # 记录已有的图片名称，用于避免重复
    existing_images = set()
    for item in main_data:
        if "images" in item:
            for img_path in item["images"]:
                # 提取图片文件名
                img_filename = os.path.basename(img_path)
                existing_images.add(img_filename)
    
    print(f"已存在的图片数量: {len(existing_images)}")
    
    # 处理每个目录
    for dir_path, prefix in dirs_to_merge:
        results_dir = os.path.join(dir_path, "results")
        images_dir = os.path.join(dir_path, "images")
        
        # 获取所有结果文件
        result_files = []
        for file in os.listdir(results_dir):
            if file.endswith(".json"):
                result_files.append(file)
        
        print(f"\n处理目录 {dir_path}: 找到 {len(result_files)} 个结果文件")
        
        # 处理每个结果文件
        for result_file in result_files:
            result_path = os.path.join(results_dir, result_file)
            
            try:
                result_data = load_json_file(result_path)
                
                # 检查是否是有效的数据条目
                if "input" in result_data and "output" in result_data:
                    # 复制数据条目并修改图片路径
                    new_entry = result_data.copy()
                    
                    if "images" in result_data:
                        new_images = []
                        for img_path in result_data["images"]:
                            # 获取原图片文件名
                            original_img_name = os.path.basename(img_path)
                            
                            # 创建新的图片名称以避免冲突
                            base_name, ext = os.path.splitext(original_img_name)
                            counter = 1
                            new_img_name = original_img_name
                            
                            # 查找未使用的图片名称
                            while new_img_name in existing_images:
                                new_img_name = f"{prefix}_{base_name}_{counter}{ext}"
                                counter += 1
                            
                            # 添加到已使用集合
                            existing_images.add(new_img_name)
                            
                            # 更新图片路径
                            new_images.append(f"./data/images/{new_img_name}")
                            
                            # 复制图片文件
                            original_img_path = os.path.join(images_dir, original_img_name)
                            if os.path.exists(original_img_path):
                                dest_img_path = os.path.join("/home/wangzhangyuan/Downloads/data(1)/data/images", new_img_name)
                                shutil.copy2(original_img_path, dest_img_path)
                                print(f"复制图片: {original_img_name} -> {new_img_name}")
                        
                        new_entry["images"] = new_images
                    
                    # 添加到主数据集
                    main_data.append(new_entry)
                    
            except Exception as e:
                print(f"处理文件 {result_path} 时出错: {str(e)}")
    
    print(f"\n合并完成后数据集包含 {len(main_data)} 条记录")
    
    # 保存合并后的数据
    output_path = "/home/wangzhangyuan/Downloads/data(1)/data/merged_action_data.json"
    save_json_file(main_data, output_path)
    print(f"合并后的数据已保存到: {output_path}")

if __name__ == "__main__":
    merge_datasets()