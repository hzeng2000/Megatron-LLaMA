#!/bin/bash

# Runs the "345M" parameter model
# run the script in respected node with $NODE_RANK
# e.g. 
# node0: bash pretrain_gpt_distributed.sh 0 
# node1: bash pretrain_gpt_distributed.sh 1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_TIMEOUT=10
GPUS_PER_NODE=8
# Change for multinode config
# MASTER_ADDR=localhost
MASTER_ADDR=11.11.4.3
MASTER_PORT=12345
NNODES=2
NODE_RANK=$1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# CHECKPOINT_PATH=/WORK/PUBLIC/zhaijd_work/dataset/gpt345/release/mp_rank_00/model_optim_rng.pt
CHECKPOINT_PATH=/WORK/PUBLIC/zhaijd_work/dataset/gpt345/release/mp_rank_00
# TENSORBOARD_LOGS_PATH=/WORK/PUBLIC/zhaijd_work/qi/Megatron-LM/workspace/logs
VOCAB_FILE=/home/fit/zhaijd/WORK/dataset/gpt345/gpt2-vocab.json
MERGE_FILE=/home/fit/zhaijd/WORK/dataset/gpt345/gpt2-merges.txt
DATA_PATH=/home/fit/zhaijd/WORK/qi/data/oscar-en-10k-meg-gpt_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max-restart 2 \
"

GPT_ARGS="
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 8 \
    --global-batch-size 64 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
# srun -N $NNODES -n $WORLD_SIZE --gres=gpu:$GPUS_PER_NODE --ntasks-per-node=$GPUS_PER_NODE \
#     python pretrain_gpt.py \
#     $GPT_ARGS \
#     $DATA_ARGS \
#     $OUTPUT_ARGS \
#     --distributed-backend nccl \
#     --save $CHECKPOINT_PATH \
#     --load $CHECKPOINT_PATH