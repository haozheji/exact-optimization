import json
import sys
import os
import tqdm

import torch

def load_json(filename):
    return json.load(open(filename, "r"))

model_logprobs_path = sys.argv[1]
ref_logprobs_path = sys.argv[2]
eval_proposal = "model"

ref_logprobs = load_json(ref_logprobs_path)
ref_logprobs = [x["logprobs"] for x in ref_logprobs]

model_logprobs = load_json(model_logprobs_path)
model_logprobs = [x["logprobs"] for x in model_logprobs]


kls = []
for ms, rs in tqdm.tqdm(zip(model_logprobs, ref_logprobs)):
    rewards = torch.FloatTensor([m - r for m, r in zip(ms, rs)])
    N = rewards.size(0)

    kl = rewards.mean()
    kls.append(kl.item())

mean_kl = sum(kls) / len(kls)


print(mean_kl)
