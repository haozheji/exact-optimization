import os
import argparse
from collections import defaultdict
import json
import tiktoken
import tqdm
import random

def clean(text):
    text = text.replace("<br />", "\n")
    return text

def random_test_set(data_path, seed = 42):
    random.seed(seed)
    print(f"randomly blend test set with seed = {seed}")
    data = json.load(open(os.path.join(data_path, "test.json"), "r"))
    pos, neg = data[:len(data) // 2], data[len(data) // 2:]
    random.shuffle(pos)
    random.shuffle(neg)
    res = []
    for p, n in zip(pos, neg):
        res.append(p)
        res.append(n)
    with open(os.path.join(data_path, "test.json"), "w") as f:
        json.dump(res, f, indent=4)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="aclImdb", help="raw data dir")
    parser.add_argument("--save_name", type=str, default="imdb", help="path to save data file")
    parser.add_argument("--prefix_len", type=int, default=10)

    args = parser.parse_args()
    sft_save_dir = args.save_name + "_prefix" + str(args.prefix_len)
    cls_save_dir = args.save_name + "_class"

    tokenizer = tiktoken.get_encoding("gpt2")

    splits = ["train", "test"]
    labels  = ["pos", "neg"]


    dataset = {"train": [], "test": []}
    prefix_dataset = {"train": [], "test": []}
    for split in splits:
        for label in labels:
            data_dir = os.path.join(args.raw, split, label)
            suffixes = os.listdir(data_dir)
            for suffix in tqdm.tqdm(suffixes, desc=f"{split}-{label}"):
                data_path = os.path.join(data_dir, suffix)
                data = open(data_path, "r").read().rstrip()
                data = clean(data)
                toks = tokenizer.encode(data)
                prefix_toks = toks[:args.prefix_len]
                cont_toks = toks[args.prefix_len:]
                prefix = tokenizer.decode(prefix_toks)
                cont = tokenizer.decode(cont_toks)
                example = {"label": label, "text": data}
                prefix_example = {"prompt": prefix, "chosen": cont}
                dataset[split].append(example)
                prefix_dataset[split].append(prefix_example)
            
        print(f"{split} has {len(dataset[split])} examples")
    
    
    os.makedirs(sft_save_dir, exist_ok=True)
    os.makedirs(cls_save_dir, exist_ok=True)

    for split in splits:
        with open(os.path.join(sft_save_dir, split + ".json"), "w") as f:
            json.dump(prefix_dataset[split], f, indent=4)
        
    for split in splits:
        with open(os.path.join(cls_save_dir, split + ".json"), "w") as f:
            json.dump(dataset[split], f, indent=4)
        
    random_test_set(sft_save_dir)


if __name__ == "__main__":
    main()
