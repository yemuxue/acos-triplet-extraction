#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${ACOS_BASE_DIR:-$SCRIPT_DIR}"
LOG_DIR="$BASE_DIR/log"
PYTHON_BIN="${ACOS_PYTHON:-python3}"
SCRIPT_PATH="$BASE_DIR/predict_triples_fuzzy.py"

mkdir -p "$LOG_DIR"
cd "$BASE_DIR"

# ═══ GPU 健康检查 ═══
source "$BASE_DIR/check_gpu.sh"
if ! check_gpu; then
  echo "⛔ GPU检查未通过，任务终止。"
  exit 1
fi
if ! select_gpu; then
  exit 1
fi
GPU_DEVICE="$SELECTED_GPU"

existing_run="$(pgrep -f "$SCRIPT_PATH --sample-size 500" || true)"
if [[ -n "${existing_run}" ]]; then
  echo "检测到已有相同样本规模的运行进程: $existing_run"
  echo "请先执行 pkill -f $SCRIPT_PATH 后再重新启动。"
  exit 1
fi

last_log="$(find "$LOG_DIR" -maxdepth 1 -type f -name '[0-9][0-9][0-9]_predict_triples_fuzzy*.log' | sort | tail -n 1 || true)"
if [[ -n "${last_log}" ]]; then
  last_name="$(basename "$last_log")"
  last_index="${last_name%%_*}"
  next_index=$((10#$last_index + 1))
else
  next_index=1
fi

seq_id="$(printf "%03d" "$next_index")"
timestamp="$(date '+%Y%m%d_%H%M%S')"
log_path="$LOG_DIR/${seq_id}_predict_triples_fuzzy_${timestamp}.log"

echo "即将启动实验，日志文件: $log_path"
{
  echo "任务开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "实验启动脚本: $0"
  echo "模型配置: Qwen3-32B"
  echo "本次改动记录 (v010 简化匹配):"
  echo "="
  echo "【当前架构】"
  echo "A1. 模型: Qwen3-32B LoRA微调 (4-bit量化推理)"
  echo "A2. 解码: Beam Search (num_beams=2) + RAG动态Few-Shot"
  echo "A3. 规则: Phase1已清理 (移除欠预测召回/过预测重路由/短句兜底)"
  echo "A4. 匹配: L1精确+L2包含+否定冲突拦截 (L3-L5已屏蔽)"
  echo "A5. 评估集: dev.csv 500条 (独立标注开发集, 未参与训练)"
  echo "A6. GPU: 自动选择 (当前 GPU $GPU_DEVICE)"
  echo "CUDA_VISIBLE_DEVICES=$GPU_DEVICE"
  echo "样本规模: 500 (开发评估集 dev.csv, 未参与训练)"
  echo
  env HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$PYTHON_BIN" -u "$SCRIPT_PATH" --sample-size 500
  echo
  echo "脚本结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
} > "$log_path" 2>&1

echo "日志路径: $log_path"
