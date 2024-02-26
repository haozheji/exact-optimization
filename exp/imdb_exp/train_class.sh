#!/bin/bash

# names and paths
init_model_name=$1
init_model_path=$2
dataset_name=imdb/class
dataset_path=exp/imdb_exp/data/imdb_class
tb_path=tb_logs
exp_name=${init_model_name}_imdb/class

# general config
dev=0,1
port=1234
train_bsz=32
eval_bsz=32
max_len=512
lr=1e-5
wm_steps=0
eps=5
grad_accum=4
wd=0
ZERO_STAGE=2


OUTPUT=models/$exp_name

if [ -d "$OUTPUT" ]; then
    echo "Warning: Directory '$OUTPUT' already exists."
else 
    mkdir -p $OUTPUT
fi

# training commands
deepspeed --include localhost:$dev --master_port $port \
src/class_stage/train.py \
   --model_name_or_path $init_model_path \
   --data_name_path $dataset_name:$dataset_path \
   --data_output_path $dataset_path \
   --output_dir $OUTPUT\
   --enable_tensorboard \
   --tensorboard_name_path $exp_name:$tb_path \
   --per_device_train_batch_size $train_bsz \
   --per_device_eval_batch_size $eval_bsz \
   --max_seq_len $max_len \
   --learning_rate $lr \
   --num_warmup_steps $wm_steps \
   --num_train_epochs $eps \
   --gradient_accumulation_steps $grad_accum \
   --weight_decay $wd \
   --gradient_checkpointing \
   --print_loss \
   --zero_stage $ZERO_STAGE \
   --deepspeed 2>&1 | tee -a $OUTPUT/training.log
