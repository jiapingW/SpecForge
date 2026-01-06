SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# support tp1 train eagle3 for Qwen3-VL-8B-Instruct
# NUM_GPUS=${1:-1}

NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=6
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-VL-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen3-vl-8b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/allava4v_train-1w.jsonl \
    --output-dir $ROOT_DIR/outputs/Qwen3-VL-8B-eagle3 \
    --build-dataset-num-proc 0 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 8192 \
    --chat-template qwen2-vl \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.language_model.embed_tokens.weight \
    --tp-size 1 \
    --is-vlm \
    --min-pixels 50176 \
    --max-pixels 802816 \
    --target-model-backend hf \
    --dist-timeout 2000 \
    --report-to tensorboard \
    --profile \
    --profile-start-step 10 \
    --profile-num-steps 10 \
    --profile-record-shapes