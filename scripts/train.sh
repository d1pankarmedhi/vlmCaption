#!/usr/bin/env bash
set -e

# Default parameters
DATASET_DIR="flickr8k"
EPOCHS=10
BATCH_SIZE=16
LR="2e-5"
NUM_WORKERS=4
MIXED_PRECISION="fp16"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset_dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --mixed_precision)
      MIXED_PRECISION="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "=========================================="
echo " Starting VLM Training"
echo "=========================================="
echo " Dataset Directory : $DATASET_DIR"
echo " Epochs            : $EPOCHS"
echo " Batch Size        : $BATCH_SIZE"
echo " Learning Rate     : $LR"
echo " Num Workers       : $NUM_WORKERS"
echo " Mixed Precision   : $MIXED_PRECISION"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  echo " Extra Args        : ${EXTRA_ARGS[*]}"
fi
echo "=========================================="

uv run python main.py train \
  --dataset_dir "$DATASET_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --num_workers "$NUM_WORKERS" \
  --mixed_precision "$MIXED_PRECISION" \
  "${EXTRA_ARGS[@]}"
