# args:
# - 1: device ids
# - 2: data_path
# - 3: model_path

device_ids=$1
data_path=$2
model_path=$3

temp=0.8
ref_temp=0.8

    
deepspeed --include "localhost:$device_ids" \
    --master_port 1234 src/align_stage/calc_logprobs.py \
                    --data_path $data_path \
                    --model_path $model_path \
                    --temperature ${temp} \
                    --eval_num 512 \
                    --max_length 512 \
                    --max_new_tokens 500 \
                    --kernel_inject \
                    --batch_size 32 \
                    --overwrite 

    