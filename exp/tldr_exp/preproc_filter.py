import json
import os
import sys

raw_path = sys.argv[-1]

out_path = "tldr_filtered"

os.makedirs(out_path, exist_ok=True)

splits = ["test", "train"]
for split in splits:
    data_path = os.path.join(raw_path, split + ".jsonl")
    data = []
    with open(data_path, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    

    res = []
    for line in data:
        prompt = "SUBREDDIT: r/{0}\n".format(line["subreddit"]) \
                + "TITLE: {0}\n".format(line["title"]) \
                + "POST: {0}\n".format(line["post"]) \
                + "TL;DR:"
                
        
        res.append({"prompt": prompt, "chosen": line["summary"]})
    
    with open(os.path.join(out_path, split + ".json"), "w") as f:
        json.dump(res, f, indent=4)
