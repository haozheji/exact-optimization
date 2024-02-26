from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModel, set_seed
from transformers.pipelines.pt_utils import KeyDataset
import datasets
import argparse
import json
import os
import deepspeed
from deepspeed.accelerator import get_accelerator
import torch
import tqdm
import sys
from torch.utils.data import Dataset

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from data import SFTDataset
from utils.data.data_utils import get_raw_dataset, ListDataset
from utils.utils import get_tokenizer

DTYPE_MAP = {"fp32": torch.float, "fp16": torch.float16}

# get env variable from deepspeed cmdline
world_size = int(os.getenv('WORLD_SIZE', '1'))

def get_data_path(args):
    return args.data_name_path.split(":")[1]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, required=True)
    parser.add_argument("--data_name_path", type=str, default="imdb/sft:imdb_exp/data/imdb_prefix10", help="name:path")
    parser.add_argument("--local_rank", type=int, help="deepspeed cmdline var")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="fp16", help="fp32|fp16")
    parser.add_argument("--split", type=str, default="test", help="split to evaluate on (train/test)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--return_num", type=int, default=4)
    parser.add_argument("--prompt_num", type=int, default=-1)
    parser.add_argument("--kernel_inject", action="store_true", help="true only for gpt-2")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # prepare save path
    if args.prompt_num == -1:
        save_path = get_data_path(args).rstrip("/") + f"_sft"#_t{args.temp}_new{args.max_new_tokens}_return{args.return_num}"
    else:
        save_path = get_data_path(args).rstrip("/") + f"_sft"#_t{args.temp}_num{args.prompt_num}_new{args.max_new_tokens}_return{args.return_num}"
    os.makedirs(save_path, exist_ok=True)

    # set seed
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    get_accelerator().manual_seed_all(args.seed)

    # load tokenizer
    tokenizer = get_tokenizer(args.model_path, fast_tokenizer=False, inference=True)
    # set model_max_length
    # see: https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/pipelines/text_generation.py#L177
    tokenizer.model_max_length = args.max_length

    # create data for inference
    raw_dataset = get_raw_dataset(args.data_name_path, args.seed, args.local_rank)
    if args.split == "test":
        inference_samples = raw_dataset.get_eval_data()
    elif args.split == "train":
        inference_samples = raw_dataset.get_train_data()
    
    if args.prompt_num == -1:
        args.prompt_num = len(inference_samples)
    inference_samples = inference_samples[:args.prompt_num]
    print(f"number of samples: {len(inference_samples)}")

    _inference_samples = []
    if args.return_num > args.batch_size:
        _return_num = args.batch_size
        _batch_size = 1
        for sample in inference_samples:
            for _ in range(args.return_num // args.batch_size):
                _inference_samples.append(sample)
    else:
        _return_num = args.return_num
        _batch_size = args.batch_size // args.return_num
        _inference_samples = inference_samples

    inference_dataset = SFTDataset(_inference_samples,
                                   raw_dataset,
                                   tokenizer,
                                   args.max_length,
                                   args.max_new_tokens,
                                   inference=True) # only this matters
    
    pipe = pipeline(task="text-generation", 
                    model=args.model_path, 
                    tokenizer=tokenizer, 
                    device=args.local_rank)
    
    # kernel inject is only needed for gpt-2
    # See supported models for automatic tensor paralleism
    # https://www.deepspeed.ai/tutorials/automatic-tensor-parallelism/#t5-11b-inference-performance-comparison
    pipe.model = deepspeed.init_inference(
        pipe.model,
        mp_size=world_size,
        dtype=DTYPE_MAP[args.dtype],
        replace_with_kernel_inject=args.kernel_inject
    )

    data = []
    count = 0
    for o in tqdm.tqdm(pipe(inference_dataset, 
                            do_sample=True, 
                            handle_long_generation="hole",
                            max_new_tokens=args.max_new_tokens, 
                            top_k=args.top_k, 
                            temperature=args.temp,
                            batch_size=_batch_size,
                            num_return_sequences=_return_num,
                            return_full_text=False,
                            eos_token_id=tokenizer.eos_token_id, # or customized 
                            ), total=len(inference_samples)):

        #generated_token_ids
        data.extend([oi['generated_text'] for oi in o])
        
    res = []
    for i in range(0, len(data), args.return_num):
        res.append({"prompt": inference_samples[i // args.return_num]["prompt"], "completions": data[i:i+args.return_num]})
        
    print(f"Save inference results in {save_path}")
    with open(os.path.join(save_path, args.split + ".json"), "w") as f:
        json.dump(res, f, indent=4)

if __name__ == "__main__":
    main()    

