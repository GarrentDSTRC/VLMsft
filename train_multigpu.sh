#!/bin/bash
# 专为4x4090设置优化的多GPU训练脚本

echo "=== 4x4090 多GPU微调启动脚本 ==="
echo "检测到的GPU数量: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "开始多GPU微调..."

# 设置环境变量以优化多GPU训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# 使用分布式训练方式（推荐）
echo "使用 torchrun 启动分布式训练..."

# 启动4卡分布式训练
python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=12355 \
    qwen3vl_finetune_proper_multi_gpu.py

echo "训练完成！查看结果在 ./qwen3-vl-2b-instruct-lora-multigpu 目录"