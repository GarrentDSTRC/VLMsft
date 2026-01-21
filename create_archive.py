#!/usr/bin/env python
"""
脚本：创建项目备份压缩包（排除模型文件夹）
"""
import os
import shutil
import tarfile
from pathlib import Path

# 定义源目录和目标目录
source_dir = Path("/mnt/workspace")
dest_dir = Path("/tmp/workspace_temp")
exclude_items = {
    "qwen3-vl-2b-instruct-lora",
    "qwen3-vl-2b-instruct-lora-multigpu",
    "wangzhangyuan-oss",
    "wangzhangyuan-oss-clean",
    "qwen3_vl_finetuning_project.tar.gz"
}

# 创建目标目录
dest_dir.mkdir(exist_ok=True)

# 复制除指定项目外的所有内容
for item in source_dir.iterdir():
    if item.name not in exclude_items:
        dest_path = dest_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest_path)
        else:
            shutil.copy2(item, dest_path)

print("文件复制完成")

# 创建压缩包在源目录
archive_path = source_dir / "qwen3_vl_finetuning_project.tar.gz"
with tarfile.open(archive_path, "w:gz") as tar:
    for item in dest_dir.iterdir():
        tar.add(item, arcname=item.name)

print("压缩包创建完成:", archive_path)

# 清理临时目录
shutil.rmtree(dest_dir)
print("临时目录清理完成")