#!/usr/bin/env bash
set -e

# Default parameters
IMAGE_PATH="data/image.png"
CHECKPOINT_PATH=""
IMAGE_PATH_SET=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --image_path|-i)
      IMAGE_PATH="$2"
      IMAGE_PATH_SET=1
      shift 2
      ;;
    --checkpoint_path|-c)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    *)
      if [ -z "$IMAGE_PATH_SET" ] && [[ "$1" != -* ]]; then
        IMAGE_PATH="$1"
        IMAGE_PATH_SET=1
        shift
      else
        EXTRA_ARGS+=("$1")
        shift
      fi
      ;;
  esac
done

if [ ! -f "$IMAGE_PATH" ]; then
  echo "Error: Image file not found at '$IMAGE_PATH'"
  echo "Usage: ./inference.sh [IMAGE_PATH] [--checkpoint_path CHECKPOINT_PATH] [--disable_cache]"
  exit 1
fi

CMD=("uv" "run" "python" "main.py" "infer" "--image_path" "$IMAGE_PATH")

if [ -n "$CHECKPOINT_PATH" ]; then
  CMD+=("--checkpoint_path" "$CHECKPOINT_PATH")
fi

echo "=========================================="
echo " Running VLM Inference"
echo "=========================================="
echo " Image Path     : $IMAGE_PATH"
if [ -n "$CHECKPOINT_PATH" ]; then
  echo " Checkpoint Path: $CHECKPOINT_PATH"
fi
echo "=========================================="

START_TIME=$(date +%s.%N)

"${CMD[@]}" "${EXTRA_ARGS[@]}"

END_TIME=$(date +%s.%N)
DURATION=$(python -c "print(f'{float($END_TIME) - float($START_TIME):.3f}')")

echo ""
echo "=========================================="
echo " Generation completed in ${DURATION} seconds"
echo "=========================================="
