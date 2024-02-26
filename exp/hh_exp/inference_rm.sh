deepspeed --include "localhost:$1" \
          --master_port 1234 src/rm_stage/inference.py \
          --data_path $2 \
          --split $3 \
          --model_path $4 \
          --batch_size 64 \
          --max_length 512 \
          --eval_num -1 \
          --overwrite \
          --mode $5 \