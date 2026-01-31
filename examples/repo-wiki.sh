#!/bin/bash

ROOT_DIR=/mnt/nas/wangjiaping/temp/SpecForge
cd $ROOT_DIR
# pip install -e .

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export PYTHONPATH=$ROOT_DIR
export NCCL_DEBUG=ERROR

# support tp4/tp8 train eagle3 for Qwen3-30B-A3B

# export TOKENIZERS_PARALLELISM=false

NUM_GPUS=8
# TP_SIZE=16

TARGET_MODEL_PATH=/mnt/nas/wangjiaping/pretrained_models/Qwen3-Coder-480B-A35B-Instruct
TRAIN_DATA_PATH=/mnt/nas/wangjiaping/dataset/repo-wiki/data_for_SpecForge_test_128k.jsonl

export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_inductor_cache"
export TRITON_CACHE_DIR="/tmp/triton_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # Prepare hidden states
export TORCH_NCCL_TIMEOUT_SEC=7200

# export NNODES=${WORLD_SIZE}
# export NODE_RANK=${RANK}
# export MASTER_ADDR=${MASTER_ADDR}
# export MASTER_PORT=${MASTER_PORT}



# torchrun \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
# torchrun \
#     --nproc_per_node $NUM_GPUS \
#     scripts/prepare_hidden_states.py \
#     --target-model-path $TARGET_MODEL_PATH \
#     --enable-aux-hidden-states \
#     --data-path $TRAIN_DATA_PATH \
#     --chat-template repo-wiki \
#     --tp-size 8 \
#     --batch-size 1 \
#     --max-length 131072 \
#     --output-path $ROOT_DIR/outputs/repo-wiki-480b/train_hidden_states \
#     --sglang-mem-fraction-ratio 0.65



# offline training
BUILD_DATASET_NUM_PROC=0

# LOG_INTERNAL=400
# SAVE_INTERNAL=992


# export NNODES=${WORLD_SIZE}
# export NODE_RANK=${RANK}
# export MASTER_ADDR=${MASTER_ADDR}
# export MASTER_PORT=${MASTER_PORT}


# torchrun \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
torchrun \
    --nproc_per_node $NUM_GPUS \
    --rdzv-conf timeout=180000 \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $TARGET_MODEL_PATH \
    --train-hidden-states-path /mnt/nas/wangjiaping/repo-wiki-project/SpecForge/outputs/repo-wiki-480b/train_hidden_states \
    --draft-model-config $ROOT_DIR/configs/qwen3-coder-480B-A35B-instruct-eagle3.json \
    --train-data-path $TRAIN_DATA_PATH \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/repo-wiki \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 131072 \
    --chat-template repo-wiki \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size 1 \
    --report-to tensorboard \
    --sp-ring-size 8 \
    --sp-ulysses-size 1 \
    --attention-backend usp \
    --dist-timeout 300


# online training
# torchrun \
#     --standalone \
#     --nproc_per_node $NUM_GPUS \
#     $ROOT_DIR/scripts/train_eagle3.py \
#     --target-model-path $TARGET_MODEL_PATH \
#     --draft-model-config $ROOT_DIR/configs/qwen3-30B-A3B-eagle3.json \
#     --train-data-path $TRAIN_DATA_PATH \
#     --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
#     --output-dir $ROOT_DIR/outputs/repo-wiki \
#     --num-epochs 10 \
#     --batch-size 1 \
#     --learning-rate 1e-4 \
#     --max-length 32768 \
#     --chat-template repo-wiki \
#     --cache-dir $ROOT_DIR/cache \
#     --embedding-key model.embed_tokens.weight \
#     --tp-size 1 \
#     --report-to tensorboard \
#     --save-interval $LOR_INTERNAL \
#     --log-interval $SAVE_INTERNAL \
#     --sp-ring-size 2 \
#     --sp-ulysses-size 4 \
#     --attention-backend usp