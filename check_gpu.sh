#!/usr/bin/env bash
# GPU 健康检查: 找到能跑 Qwen3-32B BF16 的 GPU (~68GB 空闲)
# 用法: source check_gpu.sh && select_gpu
# 返回: 设置 SELECTED_GPU 变量; 如果没有可用GPU, 返回 1

set -euo pipefail

REQUIRED_MEM_MB=24000   # Qwen3-32B 4-bit量化 约需 20-24GB
VERBOSE="${VERBOSE:-1}"

check_gpu() {
    echo "=== GPU 状态检查 ($(date '+%H:%M:%S')) ==="

    local available_gpus=()
    local gpu_info=""

    while IFS=',' read -r idx used free; do
        idx=$(echo "$idx" | xargs)
        free=$(echo "$free" | xargs | sed 's/ MiB//')

        local free_gb=$(awk "BEGIN {printf \"%.0f\", $free/1024}")
        local status="✅ 可用"

        if [ "$free" -lt "$REQUIRED_MEM_MB" ]; then
            status="❌ 不足 (需24GB, 仅${free_gb}GB空闲)"
        else
            # 检查是否有其他进程
            local used_gb=$(echo "$used" | xargs | sed 's/ MiB//')
            local used_gb_fmt=$(awk "BEGIN {printf \"%.0f\", $used_gb/1024}")
            status="✅ 可用 (${free_gb}GB空闲, ${used_gb_fmt}GB已用)"
            available_gpus+=("$idx:$free")
        fi

        gpu_info+=$(printf "  GPU %s: %s\n" "$idx" "$status")
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null)

    if [ -z "$gpu_info" ]; then
        echo "  ❌ nvidia-smi 不可用"
        return 1
    fi

    echo "$gpu_info"

    if [ ${#available_gpus[@]} -eq 0 ]; then
        echo ""
        echo "❌ 没有GPU能满足 Qwen3-32B 4-bit (24GB) 的要求!"
        echo "   请手动释放GPU或等待其他任务完成。"
        return 1
    fi

    return 0
}

select_gpu() {
    # 选择空闲内存最多的GPU
    local best_gpu=""
    local best_free=0

    while IFS=',' read -r idx free; do
        idx=$(echo "$idx" | xargs)
        free=$(echo "$free" | xargs | sed 's/ MiB//')

        if [ "$free" -ge "$REQUIRED_MEM_MB" ] && [ "$free" -gt "$best_free" ]; then
            best_gpu="$idx"
            best_free="$free"
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null)

    if [ -z "$best_gpu" ]; then
        echo ""
        echo "============================================"
        echo "  ⛔ 终止: 所有GPU显存不足24GB"
        echo "  当前GPU占用情况:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv 2>/dev/null | head -5
        echo "============================================"
        return 1
    fi

    local free_gb=$(awk "BEGIN {printf \"%.0f\", $best_free/1024}")
    echo ""
    echo "✅ 选择 GPU $best_gpu ($free_gb GB 空闲)"
    echo "   CUDA_VISIBLE_DEVICES=$best_gpu"
    export SELECTED_GPU="$best_gpu"
    return 0
}

# 直接运行时打印状态
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    check_gpu
    select_gpu
fi
