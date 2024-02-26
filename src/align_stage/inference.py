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
import time
from torch.utils.data import Dataset

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from data import DPODataset
from utils.data.data_utils import get_raw_dataset, ListDataset
from utils.utils import get_tokenizer
from utils.perf import Performance

DTYPE_MAP = {"fp32": torch.float, "fp16": torch.float16}

# get env variable from deepspeed cmdline
world_size = int(os.getenv('WORLD_SIZE', '1'))

def get_data_path(args):
    return args.data_name_path.split(":")[1]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, required=True)
    parser.add_argument("--data_name_path", type=str, default="", help="name:path")
    parser.add_argument("--local_rank", type=int, help="deepspeed cmdline var")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="fp16", help="fp32|fp16")
    parser.add_argument("--split", type=str, default="test", help="split to evaluate on (train/test)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--return_num", type=int, default=4)
    parser.add_argument("--prompt_num", type=int, default=512)
    parser.add_argument("--kernel_inject", action="store_true", help="true only for gpt-2")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # prepare save path
    save_path = os.path.join(get_data_path(args).rstrip("/") + "_" + "infer_res", args.model_path)
    os.makedirs(save_path, exist_ok=True)

    # set seed
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    get_accelerator().manual_seed_all(args.seed)

    # load tokenizer
    tokenizer = get_tokenizer(args.model_path, fast_tokenizer=False, inference=True)

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
    inference_dataset = DPODataset(inference_samples,
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
    times = []
    count = 0

    torch.cuda.synchronize()
    start = time.time()
    for prompt, o in tqdm.tqdm(zip(inference_dataset, pipe(inference_dataset, 
                            do_sample=True, 
                            handle_long_generation="hole",
                            max_new_tokens=args.max_new_tokens, 
                            top_k=args.top_k, 
                            temperature=args.temp,
                            batch_size=args.batch_size,
                            num_return_sequences=args.return_num,
                            eos_token_id=tokenizer.eos_token_id, # or customized 
                            return_tensors=True)), total=len(inference_samples)):

        return_ids = [oi["generated_token_ids"] for oi in o]
        real_gen_len = max(len(x) for x in return_ids)

        # log average time per token
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) / real_gen_len)

        # postprocessing
        prompt_len = len(prompt)

        instance = {"prompt": inference_samples[count]["prompt"], "completions": []}
        for sequence in return_ids:
            text = tokenizer.decode(
                        sequence,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
            instance["completions"].append(text[prompt_len:])
        
        count += 1
        data.append(instance)

        # start time
        torch.cuda.synchronize()
        start = time.time()

    Performance.print_perf_stats(map(lambda t: t, times), pipe.model.config, args.dtype, args.batch_size)

    print(f"Save inference results in {save_path}")
    with open(os.path.join(save_path, args.split + ".json"), "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()    

