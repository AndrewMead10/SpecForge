#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export HF_HOME=${HF_HOME:-$ROOT_DIR/cache/huggingface}

# Qwen3.5-27B uses the new qwen3_5 architecture and requires trust_remote_code.
# Keep the explicit draft config: the repo's pinned transformers version does not
# auto-recognize qwen3_5 configs yet, so use the SGLang backend rather than HF.
# Adjust NUM_GPUS/TP_SIZE to match your box. For this model, tp4 is the practical minimum.
NUM_GPUS=${1:-4}
TP_SIZE=${2:-4}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3.5-27B \
    --trust-remote-code \
    --draft-model-config $ROOT_DIR/configs/qwen3.5-27b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3.5-27b-eagle3-sharegpt \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --model-download-dir $ROOT_DIR/cache/huggingface \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang
