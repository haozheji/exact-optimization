from transformers import pipeline, AutoTokenizer, set_seed, AutoConfig, AutoModel
from transformers.pipelines.pt_utils import KeyDataset
import datasets
import argparse
import json
import os
import deepspeed
from deepspeed.accelerator import get_accelerator
import torch
import tqdm
import numpy as np
import sys
from torch.utils.data import Dataset, SequentialSampler, DistributedSampler, DataLoader

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from data import PrefDataset
from model import RewardModel
from utils.pipe_utils import RewardModelPipeline
from utils.data.data_utils import get_raw_dataset, ListDataset
from utils.utils import get_tokenizer, to_device
from utils.ds_utils import get_eval_ds_config

DTYPE_MAP = {"fp32": torch.float, "fp16": torch.float16}

# get env variable from deepspeed cmdline
world_size = int(os.getenv('WORLD_SIZE', '1'))

def load_data(data_path, split="test"):
    data = json.load(open(os.path.join(data_path, split + ".json"), "r"))
    res = []
    for line in data:
        res.append(line)
    return res

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="", help="path to generated data")
    parser.add_argument("--local_rank", type=int, help="deepspeed cmdline var")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="fp16", help="fp32|fp16")
    parser.add_argument("--split", type=str, default="test", help="split to evaluate on (train/test)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--eval_num", type=int, default=-1)
    parser.add_argument("--mode", type=str, default="label", help="eval|label")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--kernel_inject", action="store_true", help="true only for gpt-2")

    return parser.parse_args()

def main():
    args = parse_args()

    # prepare save path
    if args.mode == "eval":
        save_path = os.path.join(args.data_path, "eval")
    elif args.mode == "label":
        save_path = args.data_path.rstrip("/").replace("sft", "rw")
    os.makedirs(save_path, exist_ok=True)

    # set seed 
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    get_accelerator().manual_seed_all(args.seed)

    # load tokenizer
    tokenizer = get_tokenizer(args.model_path, fast_tokenizer=False)

    # load model
    model_config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModel.from_config(model_config)
    model = RewardModel(model, tokenizer)
    model.load_state_dict(torch.load(os.path.join(args.model_path, "pytorch_model.bin"), map_location='cpu'))
    
    # load data
    samples = load_data(args.data_path, args.split)
    if args.eval_num == -1:
        args.eval_num = len(samples)
    samples = samples[:args.eval_num]


    texts_flat = []
    if "completions" in samples[0]:
        return_num = len(samples[0]["completions"])
        for s in samples:
            for text in s["completions"]:
                texts_flat.append(s["prompt"] + text)

        
    else: # pref dataset
        return_num = 2
        for s in samples:
            texts_flat.append(s["prompt"] + s["chosen"])
            texts_flat.append(s["prompt"] + s["rejected"])

    dataset = ListDataset(texts_flat)

    if not os.path.isfile(os.path.join(save_path, args.split + ".json")) or args.overwrite:

        pipe = RewardModelPipeline(model=model,
                                    tokenizer=tokenizer, 
                                    device=args.local_rank)
        
        # init by deepspeed
        pipe.model = deepspeed.init_inference(
            pipe.model,
            mp_size=world_size,
            dtype=DTYPE_MAP[args.dtype],
            replace_with_kernel_inject=args.kernel_inject 
        )

        rewards = []
        for s in tqdm.tqdm(pipe(dataset, 
                                padding=True,
                                truncation=True,
                                max_length=args.max_length, 
                                batch_size=args.batch_size), total=len(dataset)):
            
            rewards.append(s[0])
    
    else:
        print("found saved data!")
        with open(os.path.join(save_path, args.split + ".json"), "r") as f:
            saved_data = json.load(f)
        rewards = []
        for line in saved_data:
            for st in line["scores_texts"]:
                rewards.append(st[0])

    data = []
    total_num = 0
    mean_rewards = 0
    for i in range(len(samples)):
        instance = {"prompt": samples[i]["prompt"]}
        cand_scores = []
        for j in range(return_num):
            cand_id = i * return_num + j
            cand_scores.append(rewards[cand_id])
        cand_scores = np.array(cand_scores)

        if args.mode == "label":
            instance["completions"] = []
            instance["rewards"] = []
            for j in range(return_num):
                instance["completions"].append(texts_flat[i * return_num + j][len(instance["prompt"]):])
                instance["rewards"].append(cand_scores[j])

            data.append(instance)
        
        elif args.mode == "eval":
            instance["scores_texts"] = []
            expected_rewards = 0
            total_num += 1
            for j in range(return_num):
                instance["scores_texts"].append((cand_scores[j], texts_flat[i * return_num + j]))
                expected_rewards += cand_scores[j]
            expected_rewards /= return_num
            #print(expected_rewards)
            mean_rewards += expected_rewards
                
            
            data.append(instance)

    if args.mode == "eval":
        print("Mean reward: ", mean_rewards / total_num)
        with open(os.path.join(save_path, "reward.json"), "w") as f:
            json.dump({"mean reward": mean_rewards / total_num}, f, indent=4)

    print(f"Save inference results in {save_path}")
    with open(os.path.join(save_path, args.split + ".json"), "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()


