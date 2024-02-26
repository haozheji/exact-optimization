import json 
import os
import sys

src_root_dir = sys.argv[-1]
tgt_root_dir = "hh"
os.makedirs(tgt_root_dir, exist_ok=True)

def read(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def write(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4) 

src_filenames = [os.path.join(src_root_dir, "train.jsonl"), os.path.join(src_root_dir, "test.jsonl")]
tgt_filenames = [os.path.join(tgt_root_dir, "train.json"), os.path.join(tgt_root_dir, "test.json")]

os.makedirs("/".join(tgt_filenames[0].split("/")[:-1]), exist_ok=True)

for src_filename, tgt_filename in zip(src_filenames, tgt_filenames):
    data = read(src_filename)
    res = []
    for line in data:
        chosen = line["chosen"]
        rejected = line["rejected"]
        chosen_segs = chosen.split("\n\n")
        rejected_segs = rejected.split("\n\n")

        chosen_segs_join = []
        assert(chosen_segs[0] == ""), chosen_segs
        temp = ""
        human_prompts = ["Human: ", "Humans: "]
        assistant_prompt = "Assistant: "
        for seg in chosen_segs[1:]:
            if seg[:len(human_prompts[0])] == human_prompts[0] or \
                seg[:len(human_prompts[1])] == human_prompts[1]:
                chosen_segs_join.append(temp)
                temp = seg.replace("Humans: ", "Human: ")
                continue
                   
            
            if seg[:len(assistant_prompt)] == assistant_prompt:
                chosen_segs_join.append(temp)
                seg = seg.replace("Human: ", " ").replace("Humans: ", " ")
                temp = assistant_prompt + seg[len(assistant_prompt):].lstrip()
                continue

            
            temp = temp + "\n\n" + seg

        chosen_segs_join.append(temp)
        chosen_segs = chosen_segs_join


        # reject
        rejected_segs_join = []
        assert(rejected_segs[0] == ""), rejected_segs
        temp = ""
        human_prompts = ["Human: ", "Humans: "]
        assistant_prompt = "Assistant: "
        for seg in rejected_segs[1:]:
            if seg[:len(human_prompts[0])] == human_prompts[0] or \
                seg[:len(human_prompts[1])] == human_prompts[1]:
                rejected_segs_join.append(temp)
                temp = seg.replace("Humans: ", "Human: ")
                continue
            
            if seg[:len(assistant_prompt)] == assistant_prompt:
                rejected_segs_join.append(temp)
                temp = seg.replace("Human: ", " ").replace("Humans: ", " ")
                continue

            temp = temp + "\n\n" + seg

        rejected_segs_join.append(temp)
        rejected_segs = rejected_segs_join



        template = "Assistant:"
        prompt = []
        chosen_response = ""
        rejeted_response = ""
        for chosen_seg, rejected_seg in zip(chosen_segs, rejected_segs):
            print(chosen_seg)
            print(rejected_seg)
            if chosen_seg == rejected_seg:
                prompt.append(chosen_seg)
            else:
                assert(template == chosen_seg[:len(template)]), chosen_segs
                assert(template == rejected_seg[:len(template)]), rejected_segs

                chosen_response = chosen_seg#.split("Human:")[0].split("Humans:")[0].strip()
                rejeted_response = rejected_seg#.split("Human:")[0].split("Humans:")[0].strip()

                break

        # filter same response
        if chosen_response.strip() == "" or rejeted_response.strip() == "":
            continue

        assert(chosen_response[:len(template)] == template), chosen_response + "@@@"
        assert(rejeted_response[:len(template)] == template), rejeted_response + "@@@"

        chosen_response = chosen_response[len(template):]
        rejeted_response = rejeted_response[len(template):]

        # filter empty response
        if chosen_response.strip() == "" or rejeted_response.strip() == "":
            continue

        prompt.append(template)

        res.append({"prompt": "\n\n".join(prompt), "chosen": chosen_response, "rejected": rejeted_response})

    write(res, tgt_filename)

