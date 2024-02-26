import json
import os
import sys

raw_path = sys.argv[-1]

out_path = "tldr"

os.makedirs(out_path, exist_ok=True)

splits = ["test", "train"]
for split in splits:
    data_path = os.path.join(raw_path, split + ".json")
    data = []
    with open(data_path, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    res = []
    for line in data:
        prompt = "SUBREDDIT: r/{0}\n".format(line["info"]["subreddit"]) \
                + "TITLE: {0}\n".format(line["info"]["title"]) \
                + "POST: {0}\n".format(line["info"]["post"]) \
                + "TL;DR:"
                
        summaries = [line["summaries"][0]["text"], line["summaries"][1]["text"]]
        choice = line["choice"]
        chosen = " " + summaries[choice].strip()
        rejected = " " + summaries[0 if choice == 1 else 1].strip()
        res.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    
    with open(os.path.join(out_path, split + ".json"), "w") as f:
        json.dump(res, f, indent=4)
