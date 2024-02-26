import sys
import os 
import json

data_path = sys.argv[1]
out_path = data_path.rstrip("/") + "_p2r"
os.makedirs(out_path, exist_ok=True)

splits = ["train", "test"]

print(f"Save to {out_path}")
for split in splits:
    res = []
    filename = os.path.join(data_path, split + ".json")
    data = json.load(open(filename, "r"))

    res = []
    for line in data:
        temp = {}
        temp["prompt"] = line["prompt"]
        temp["completions"] = [line["chosen"], line["rejected"]]
        temp["rewards"] = [1, 0]
        res.append(temp)
    
    with open(os.path.join(out_path, split + ".json"), "w") as f:
        json.dump(res, f, indent=4)

