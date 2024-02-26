# args:
# - 1: device ids
# - 2: model_path

ckpts=(1 2 3 4 5 6 7 8 9 10) 

temp=0.8
ref_temp=0.8

for i in "${ckpts[@]}"
do
    # generate
    model_path="${2%/}"/ckpt${i}

    deepspeed --include "localhost:$1" \
          --master_port 1234 src/align_stage/inference.py \
                            --split test --model_path $model_path \
                            --prompt_num 512 \
                            --return_num 4 \
                            --seed 42 \
                            --temp $temp \
                            --data_name_path tldr/sft:exp/tldr_exp/data/tldr \
                            --max_length 650 \
                            --max_new_tokens 75 \
                            --batch_size 16 \
    

done