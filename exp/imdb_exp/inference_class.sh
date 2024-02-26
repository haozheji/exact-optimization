
deepspeed --include "localhost:$1" \
    --master_port 1234 src/class_stage/inference.py \
    --data_path $2 \
    --split $3 \
    --model_path $4 \
    --batch_size 64 \
    --mode $5 \
    --max_length 512 \
    --overwrite \
    --kernel_inject \
    --eval_num -1 \
    