deepspeed --include "localhost:$1" \
          --master_port 1234 src/sft_stage/inference.py \
                            --split $3 --model_path $4 \
                            --data_name_path $2 \
                            --prompt_num 512 \
                            --temp 0.8 \
                            --max_len 650 \
                            --max_new_tokens 75 \
                            --return_num 2 \
                            --batch_size 32 \