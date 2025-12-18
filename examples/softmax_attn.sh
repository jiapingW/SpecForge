SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export PYTHONPATH=$ROOT_DIR
export CUDA_HOME=/usr/local/cuda-12.6
export NCCL_DEBUG=WARN


MODEL_PATH=/mnt/nas/wangjiaping/pretrained_models/Qwen3-8B
TRAIN_DATA_PATH=/mnt/nas/wangjiaping/projects/SpecForge/cache/dataset/ultrachat_train.jsonl
EVAL_DATA_PATH=/mnt/nas/wangjiaping/projects/SpecForge/cache/dataset/ultrachat_test.jsonl
# HIDDEN_STATE_PATH=/disk3/wjp/projects/specforge_output/nes_1w


# regenerate data
# python scripts/regenerate_train_data.py \
#     --model /disk3/wjp/pretrained_models/Qwen2.5-7B-Instruct \
#     --concurrency 128 \
#     --max-tokens 98304 \
#     --server-address localhost:30010 localhost:30020 localhost:30030 localhost:30040 \
#     --temperature 0.8 \
#     --input-file-path /disk3/wjp/projects/SpecForge/cache/dataset/ultrachat_train.jsonl \
#     --output-file-path /disk3/wjp/projects/SpecForge/cache/dataset/ultrachat_train_regen.jsonl




# torchrun \
#     --standalone \
#     --nproc_per_node $NUM_GPUS \
#     scripts/prepare_hidden_states.py \
#     --target-model-path $MODEL_PATH \
#     --enable-aux-hidden-states \
#     --data-path $DATA_PATH \
#     --output-path $HIDDEN_STATE_PATH \
#     --chat-template qwen \
#     --tp-size 1 \
#     --max-length 16384 \
#     --batch-size 8 \
#     --quantization fp8 \



# online训练
NPROC_PER_NODE=8
torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --standalone \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3.json \
    --train-data-path $TRAIN_DATA_PATH \
    --save-interval 25000 \
    --eval-data-path $EVAL_DATA_PATH \
    --eval-interval 25000 \
    --output-dir $ROOT_DIR/outputs/linear_attn_with_norm \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend linear_attn \
    --tp-size 1 \
    --report-to tensorboard \
    --target-model-backend hf