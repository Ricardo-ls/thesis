#!/bin/bash
set -e

DIRS_TO_CLEAN=(
  "data/stage3_global"
  "outputs/stage3_global"
  "tools/stage3_global"
  "data/stage3_indoor"
  "outputs/stage3_indoor"
  "tools/stage3_indoor"
  "data/simulated"
  "outputs/prior/train/ddpm_finetuned_h128"
)

echo "=== 即将清理的目录 ==="
for dir in "${DIRS_TO_CLEAN[@]}"; do
  if [ -d "$dir" ]; then
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    files=$(find "$dir" -type f | wc -l | xargs)
    echo "  [删除] $dir  ($size, $files files)"
  else
    echo "  [跳过] $dir  (不存在)"
  fi
done

echo ""
echo "=== 项目根目录今天修改的文件（需人工判断是否删除） ==="
find . -maxdepth 1 -type f -mtime -1 \( -name "*.py" -o -name "*.sh" \) 2>/dev/null

echo ""
echo "=== 保留文件确认（必须存在） ==="
for f in \
  "outputs/prior/train/ddpm_eth_ucy_none_h128/seed42-100epoch/best_model.pt" \
  "datasets/processed/data_eth_ucy_20.npy"; do
  if [ -f "$f" ]; then
    echo "  [保留] $f  ✓"
  else
    echo "  [警告] $f  缺失！"
    exit 1
  fi
done

echo ""
read -p "确认删除上面列出的目录？输入 y 继续，其他取消: " confirm
if [ "$confirm" = "y" ]; then
  for dir in "${DIRS_TO_CLEAN[@]}"; do
    [ -d "$dir" ] && rm -rf "$dir" && echo "已删除 $dir"
  done
  echo "=== 清理完成 ==="
else
  echo "=== 已取消 ==="
  exit 0
fi
