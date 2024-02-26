from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM, set_seed, default_data_collator
from transformers.pipelines.pt_utils import KeyDataset
from transformers import Pipeline, TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper
import datasets
import argparse
import json
import os
import deepspeed
from deepspeed.accelerator import get_accelerator
import torch
import tqdm
import sys
from torch.utils.data import Dataset, SequentialSampler, DataLoader

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.pipe_utils import LogprobsPipeline
from utils.data.data_utils import get_raw_dataset, ListDataset
from utils.utils import get_tokenizer, to_device, print_rank_0, save_json
from utils.model.model_utils import create_hf_model

DTYPE_MAP = {"fp32": torch.float, "fp16": torch.float16}



# get env variable from deepspeed cmdline
world_size = int(os.getenv('WORLD_SIZE', '1'))

def load_data(data_path, split="test"):
    data = json.load(open(os.path.join(data_path, split + ".json"), "r"))
    return data



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, required=True)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--local_rank", type=int, help="deepspeed cmdline var")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="fp16", help="fp32|fp16")
    parser.add_argument("--split", type=str, default="test", help="split to evaluate on (train/test)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval_num", type=int, default=512)
    parser.add_argument("--mc_num", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true", help="overwrite cache")
    parser.add_argument("--kernel_inject", action="store_true", help="true only for gpt-2")

    return parser.parse_args()

def main():
    args = parse_args()

    # prepare save path
    model_ckpt = args.model_path.rstrip("/").split("/")[-1]
    save_path = os.path.join(args.data_path, f"logprobs_{model_ckpt}.json")
    
    print("destination: ", save_path)
    
    has_model_cache = False
    if os.path.isfile(save_path):
        print_rank_0(f"saved logprobs found: {save_path} !!!!!!!!!")
        has_model_cache = True
    
    if args.overwrite:
        print_rank_0("overwrite")
    
    if has_model_cache and not args.overwrite:
        print_rank_0("finish")
        exit()

    # set seed
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    # load tokenizer
    tokenizer = get_tokenizer(args.model_path, fast_tokenizer=False, inference=False) 
    # `inference` mode will use left padding!
    # we want right padding when evaluating kl

    # load model
    print_rank_0("init model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    # create data for evaluation
    data = load_data(args.data_path, split=args.split)
    data = data[:args.eval_num]
    texts_pairs_flat = []
    num_cands = len(data[0]["completions"])
    if args.mc_num != -1:
        num_cands = args.mc_num
        _data = []
        for line in data:
            _data.append({"prompt": line["prompt"], "completions": line["completions"][:num_cands]})
        data = _data
    #assert(args.batch_size % num_cands == 0), f"batch size {args.batch_size} must be dividable by {num_cands}"
    for texts in data:
        for text in texts["completions"]:
            texts_pairs_flat.append((texts["prompt"], text))
    
    eval_dataset = ListDataset(texts_pairs_flat)


    # create pipe
    model_pipe = LogprobsPipeline(model=model,
                                tokenizer=tokenizer,
                                device=args.local_rank)
    
    model_pipe.model = deepspeed.init_inference(
        model_pipe.model,
        mp_size=world_size,
        dtype=DTYPE_MAP[args.dtype],
        replace_with_kernel_inject=args.kernel_inject   
    )


    model_logprobs_seq = []

    for logprob_seq in tqdm.tqdm(model_pipe(eval_dataset,
                                            padding="max_length",
                                            truncation=True,
                                            max_new_tokens=args.max_new_tokens,
                                            max_length=args.max_length,
                                            temperature=args.temperature,
                                            batch_size=args.batch_size), total=len(eval_dataset), desc="evaluating model"):
        
        assert(type(logprob_seq) == float), logprob_seq
        model_logprobs_seq.append(logprob_seq)

    assert(len(model_logprobs_seq) == len(texts_pairs_flat))
    if args.local_rank == 0:
        model_logprobs_group = []
        for i in range(0, len(model_logprobs_seq), num_cands):
            model_logprobs_group.append(model_logprobs_seq[i:i+num_cands])
        res = []
        for logprobs, texts in zip(model_logprobs_group, data):
            texts["logprobs"] = logprobs
            res.append(texts)
        print(f"save logprobs to {save_path}")
        save_json(res, save_path)
                                        


    


if __name__ == "__main__":
    main()