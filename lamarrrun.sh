#!/bin/bash
#SBATCH --job-name=vit_48_bs_100
#SBATCH --time=72:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=out/vit_48_bs_100-%j.out
#SBATCH --error=out/vit_48_bs_100-%j.err


# 获取当前日期和时间
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")

# 获取当前时间的秒数
START_SECONDS=$(date +%s)

# 设置日志文件路径
LOG_FILE="jilulog/experiment_$CURRENT_TIME.txt"

# 检查并创建日志文件目录
LOG_DIR=$(dirname "$LOG_FILE")
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# 记录开始时间和命令
echo "Experiment started at $CURRENT_TIME" >> "$LOG_FILE"

# 执行命令并将stdout追加到日志文件，同时输出到终端（保留控制字符）
function execute_and_log {
    local CMD="$1"
    echo "Executing: $CMD" >> "$LOG_FILE"
    eval "$CMD" | while IFS= read -r line; do
        # 计算从开始到现在所用的时间
        CURRENT_SECONDS=$(date +%s)
        ELAPSED_SECONDS=$((CURRENT_SECONDS - START_SECONDS))
        ELAPSED_MINUTES=$((ELAPSED_SECONDS / 60))
        ELAPSED_SECONDS=$((ELAPSED_SECONDS % 60))
        # 添加时间信息并输出
        printf "[%d:%02d] %s\n" "$ELAPSED_MINUTES" "$ELAPSED_SECONDS" "$line"
    done | tee -ai "$LOG_FILE"
}


# 在后台运行 nvidia-smi 命令并将输出重定向到日志文件
(sleep 300 && nvidia-smi) >> "$LOG_FILE" 2>&1 &

# 执行命令
echo "setting : xxxxx" >> "$LOG_FILE"
execute_and_log "nvidia-smi"
execute_and_log "/home/ma1/anaconda3/envs/vitnew/bin/python -u MainCondition.py --batch_size 800"

# 记录结束时间
echo "Experiment ended at $(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_FILE"
