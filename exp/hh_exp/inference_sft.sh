deepspeed --include "localhost:$1" \
          --master_port 1234 src/sft_stage/inference.py \
                            --split $3 --model_path $4 \
                            --data_name_path $2 \
                            --prompt_num -1 \
                            --temp 0.8 \
                            --max_len 512 \
                            --max_new_tokens 200 \
                            --return_num 2 \
                            --batch_size 64 \