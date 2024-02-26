#!/bin/bash

# names and paths
init_model_name=$1
init_model_path=$2
model_type=$3
dataset_name=$4
dataset_path=$5
# options: exo-pref/exo-rw
loss_type=$6
num_contrastive=$7
tb_path=tb_logs

dataset_abbr=$( echo $dataset_name | cut -d'/' -f1 )

# general config
dev=0,1,2,3
port=1484
train_bsz=4
eval_bsz=4
max_len=512
max_gen_len=200
lr=1e-6
wm_steps=0
eps=1
grad_accum=4
wd=0
ZERO_STAGE=2
num_save_checkpoint=40
save_step_interval=-1
max_iter_step=1400

# alignment config
beta_r=0.5
beta_pi=0.5
temp=0.8


exp_name=${init_model_name}_${dataset_abbr}/${model_type}_${loss_type}_nc${num_contrastive}

OUTPUT=models/$exp_name

if [ -d "$OUTPUT" ]; then
    echo "Warning: Directory '$OUTPUT' already exists."
else 
    mkdir -p $OUTPUT
fi

# training commands
deepspeed --include localhost:$dev --master_port $port \
src/align_stage/train.py \
   --model_name_or_path $init_model_path \
   --ref_name_or_path $init_model_path \
   --beta_r $beta_r \
   --beta_pi $beta_pi \
   --num_contrastive $num_contrastive \
   --temp $temp \
   --max_iter_step $max_iter_step \
   --save_step_interval $save_step_interval \
   --num_save_checkpoint $num_save_checkpoint \
   --loss_type $loss_type \
   --data_name_path $dataset_name:$dataset_path \
   --data_output_path $dataset_path \
   --output_dir $OUTPUT \
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
