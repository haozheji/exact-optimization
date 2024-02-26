import os
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset
import re
import json


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, seed, local_rank):
        self.seed = seed
        self.local_rank = local_rank

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


        

class ImdbRewardDataset(PromptRawDataset):

    def __init__(self, seed, local_rank, dataset_name, dataset_path):
        super().__init__(seed, local_rank)
        self.dataset_name = "imdb/rw"
        self.dataset_name_clean = "imdb_rw"
        self.raw_datasets = {"train": json.load(open(os.path.join(dataset_path, "train.json"), "r")),
                             "test": json.load(open(os.path.join(dataset_path, "test.json"), "r"))}
        
    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return sample['prompt']        

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_prompt_and_completions(self, sample):
        return [sample['prompt'] + x for x in sample["completions"]]
    
    def get_completions(self, sample):
        return sample["completions"]
    
    def get_rewards(self, sample):
        return sample['rewards']
    




class ImdbClassDataset(PromptRawDataset):

    def __init__(self, seed, local_rank, dataset_name, dataset_path):
        super().__init__(seed, local_rank)
        self.dataset_name = "imdb/class"
        self.dataset_name_clean = "imdb_class"
        self.raw_datasets = {"train": json.load(open(os.path.join(dataset_path, "train.json"), "r")),
                             "test": json.load(open(os.path.join(dataset_path, "test.json"), "r"))}
    
    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']
    
    def get_text(self, sample):
        return sample["text"]
    
    def get_label(self, sample):
        return sample["label"]

class ImdbPrefDataset(PromptRawDataset):

    def __init__(self, seed, local_rank, dataset_name, dataset_path):
        super().__init__(seed, local_rank)
        self.dataset_name = "imdb/pref"
        self.dataset_name_clean = "imdb_pref"
        self.raw_datasets = {"train": json.load(open(os.path.join(dataset_path, "train.json"), "r")),
                             "test": json.load(open(os.path.join(dataset_path, "test.json"), "r"))}
    
    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return sample['prompt']        

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return sample['chosen']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class ImdbSFTDataset(PromptRawDataset):

    def __init__(self, seed, local_rank, dataset_name, dataset_path):
        super().__init__(seed, local_rank)
        self.dataset_name = "imdb/sft"
        self.dataset_name_clean = "imdb_sft"
        self.raw_datasets = {"train": json.load(open(os.path.join(dataset_path, "train.json"), "r")) if os.path.isfile(os.path.join(dataset_path, "train.json")) else None,
                             "test": json.load(open(os.path.join(dataset_path, "test.json"), "r"))}

    def get_train_data(self):
        return self.raw_datasets['train']
        

    def get_eval_data(self):
        return self.raw_datasets['test']
        

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return sample['prompt']
        

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return sample['chosen']
        

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']
    
    def get_rejected(self, sample):
        return None
    
    def get_prompt_and_rejected(self, sample):
        return None
        
# ============================================================
# Anthropic

class HHSFTDataset(PromptRawDataset):

    def __init__(self, seed, local_rank, dataset_name, dataset_path):
        super().__init__(seed, local_rank)
        self.dataset_name = "hh/sft"
        self.dataset_name_clean = "hh_sft"
        self.raw_datasets = {"train": json.load(open(os.path.join(dataset_path, "train.json"), "r")) if os.path.isfile(os.path.join(dataset_path, "train.json")) else None,
                             "test": json.load(open(os.path.join(dataset_path, "test.json"), "r"))}
        
    def get_train_data(self):
        return self.raw_datasets['train']
        

    def get_eval_data(self):
        return self.raw_datasets['test']
        

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return sample['prompt']# + "\n\n"

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return sample['chosen']
        

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']
    
    def get_rejected(self, sample):
        return None
    
    def get_prompt_and_rejected(self, sample):
        return None


class HHPrefDataset(PromptRawDataset):

    def __init__(self, seed, local_rank, dataset_name, dataset_path):
        super().__init__(seed, local_rank)
        self.dataset_name = "hh/pref"
        self.dataset_name_clean = "hh_pref"
        self.raw_datasets = {"train": json.load(open(os.path.join(dataset_path, "train.json"), "r")),
                             "test": json.load(open(os.path.join(dataset_path, "test.json"), "r"))}
    
    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return sample['prompt']# + "\n\n"    

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return sample['chosen']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']
    

class HHRewardDataset(PromptRawDataset):

    def __init__(self, seed, local_rank, dataset_name, dataset_path):
        super().__init__(seed, local_rank)
        self.dataset_name = "hh/rw"
        self.dataset_name_clean = "hh_rw"
        self.raw_datasets = {"train": json.load(open(os.path.join(dataset_path, "train.json"), "r")),
                             "test": json.load(open(os.path.join(dataset_path, "test.json"), "r"))}
        
    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return sample['prompt']

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_prompt_and_completions(self, sample):
        return [sample['prompt'] + x for x in sample["completions"]]
    
    def get_completions(self, sample):
        return sample["completions"]
    
    def get_rewards(self, sample):
        return sample['rewards']

# ======================================
## tldr


class TldrSFTDataset(PromptRawDataset):

    def __init__(self, seed, local_rank, dataset_name, dataset_path):
        super().__init__(seed, local_rank)
        self.dataset_name = "tldr/sft"
        self.dataset_name_clean = "tldr_sft"
        self.raw_datasets = {"train": json.load(open(os.path.join(dataset_path, "train.json"), "r")) if os.path.isfile(os.path.join(dataset_path, "train.json")) else None,
                             "test": json.load(open(os.path.join(dataset_path, "test.json"), "r"))}
        
    def get_train_data(self):
        return self.raw_datasets['train']
        

    def get_eval_data(self):
        return self.raw_datasets['test']
        

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return sample['prompt']# + "\n\n"

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return sample['chosen']
        

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']
    
    def get_rejected(self, sample):
        return None
    
    def get_prompt_and_rejected(self, sample):
        return None

class TldrPrefDataset(PromptRawDataset):

    def __init__(self, seed, local_rank, dataset_name, dataset_path):
        super().__init__(seed, local_rank)
        self.dataset_name = "tldr/pref"
        self.dataset_name_clean = "tldr_pref"
        self.raw_datasets = {"train": json.load(open(os.path.join(dataset_path, "train.json"), "r")),
                             "test": json.load(open(os.path.join(dataset_path, "test.json"), "r"))}
    
    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return sample['prompt']# + "\n\n"    

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return sample['chosen']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']
    

class TldrRewardDataset(PromptRawDataset):

    def __init__(self, seed, local_rank, dataset_name, dataset_path):
        super().__init__(seed, local_rank)
        self.dataset_name = "tldr/rw"
        self.dataset_name_clean = "tldr_rw"
        self.raw_datasets = {"train": json.load(open(os.path.join(dataset_path, "train.json"), "r")),
                             "test": json.load(open(os.path.join(dataset_path, "test.json"), "r"))}
        
    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return sample['prompt']

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_prompt_and_completions(self, sample):
        return [sample['prompt'] + x for x in sample["completions"]]
    
    def get_completions(self, sample):
        return sample["completions"]
    
    def get_rewards(self, sample):
        return sample['rewards']