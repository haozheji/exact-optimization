#!/bin/bash

# names and paths
init_model_name=$1
init_model_path=$2
dataset_name=tldr/pref
dataset_path=exp/tldr_exp/data/tldr
tb_path=tb_logs
exp_name=${init_model_name}_tldr/rm


dev=0,1,2,3
port=1980
train_bsz=8
eval_bsz=8
max_len=650
max_gen_len=75
lr=1e-5
wm_steps=0
eps=1
grad_accum=4
wd=0
ZERO_STAGE=2

OUTPUT=models/$exp_name


if [ "$OUTPUT" == "" ]; then
    OUTPUT=models/$exp_name
fi

if [ -d "$OUTPUT" ]; then
    echo "Warning: Directory '$OUTPUT' already exists."
else 
    mkdir -p $OUTPUT
fi

# training commands ==================================


deepspeed --include localhost:$dev --master_port $port \
src/rm_stage/train.py \
   --model_name_or_path $init_model_path \
   --data_name_path $dataset_name:$dataset_path \
   --data_output_path $dataset_path \
   --output_dir $OUTPUT\
   --enable_tensorboard \
   --tensorboard_name_path $exp_name:$tb_path \
   --per_device_train_batch_size $train_bsz \
   --per_device_eval_batch_size $eval_bsz \
   --max_seq_len $max_len \
   --max_gen_len $max_gen_len \
   --learning_rate $lr \
   --num_warmup_steps $wm_steps \
   --num_train_epochs $eps \
   --gradient_accumulation_steps $grad_accum \
   --weight_decay $wd \
   --gradient_checkpointing \
   --print_loss \
   --zero_stage $ZERO_STAGE \
   --deepspeed 2>&1 | tee -a $OUTPUT/training.log
