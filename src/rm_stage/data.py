from torch.utils.data import Dataset
from utils.data.data_utils import BaseDataset
import torch

from transformers import Pipeline

class PrefDataset(BaseDataset):
    def __init__(self, samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, vectors=None):
        super().__init__(samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, vectors=vectors)
    
    def vectorize(self, sample):
        
        # chosen
        self.tokenizer.truncation_side = "right"
        chosen_response = self.get_chosen(sample)
        chosen_response += self.tokenizer.eos_token
        trunc_chosen_response_token = self.tokenizer.encode(chosen_response,
                                max_length=self.max_gen_len,
                                truncation=True)
        # check whether append <s>
        if trunc_chosen_response_token[0] == self.tokenizer.bos_token_id:
            trunc_chosen_response_token = trunc_chosen_response_token[1:]

        trunc_chosen_response_len = len(trunc_chosen_response_token)
        trunc_chosen_response = self.tokenizer.decode(trunc_chosen_response_token)
        

        # rejected
        rejected_response = self.get_rejected(sample)
        rejected_response += self.tokenizer.eos_token
        trunc_rejected_response_token = self.tokenizer.encode(rejected_response,
                                max_length=self.max_gen_len,
                                truncation=True)
        # check whether append <s>
        if trunc_rejected_response_token[0] == self.tokenizer.bos_token_id:
            trunc_rejected_response_token = trunc_rejected_response_token[1:]

        trunc_rejected_response_len = len(trunc_rejected_response_token)
        trunc_rejected_response = self.tokenizer.decode(trunc_rejected_response_token)

        # prompt
        prompt = self.get_prompt(sample)
        
        chosen_sentence = prompt + trunc_chosen_response
        reject_sentence = prompt + trunc_rejected_response
        self.tokenizer.truncation_side = "left"
        chosen_token = self.tokenizer(chosen_sentence,
                                         max_length=self.max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")

        reject_token = self.tokenizer(reject_sentence,
                                         max_length=self.max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
        
        chosen_real_len = chosen_token["attention_mask"].sum(-1, keepdim=True)
        chosen_token["class_pos"] = chosen_real_len - 2 if chosen_token["input_ids"][0][
                    chosen_real_len -1].item() == self.tokenizer.eos_token_id else chosen_real_len - 1

        reject_real_len = reject_token["attention_mask"].sum(-1, keepdim=True)
        reject_token["class_pos"] = reject_real_len - 2 if reject_token["input_ids"][0][
            reject_real_len - 1].item() == self.tokenizer.eos_token_id else reject_real_len - 1
        
        return chosen_token, reject_token
    
    def collate(self, data):
        batch = {}
        for k in ["input_ids", "attention_mask", "class_pos"]:
            batch[k] = torch.cat([x[0][k] for x in data] +
                                 [x[1][k] for x in data], dim=0)

        return batch





