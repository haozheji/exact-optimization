from torch.utils.data import Dataset
from utils.data.data_utils import BaseDataset
import torch

DTYPE_MAP = {"fp32": torch.float, "fp16": torch.float16}

# TODO: Deprecate this file
# use data.py from rm_stage

class DPODataset(BaseDataset):
    def __init__(self, samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, inference=False, vectors=None, dtype="fp32"):
        super().__init__(samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, inference=inference, vectors=vectors)
        self.dtype = DTYPE_MAP[dtype]
        if not inference:
            self.num_cands = 2

    
    def vectorize(self, sample, inference=False):
        if inference:
            prompt = self.get_prompt(sample)
            chosen = self.get_chosen(sample)
            assert(chosen.strip() != ""), chosen
            tokens = self.tokenizer.encode(prompt + chosen)
            chosen_tokens = self.tokenizer.encode(chosen)

            if self.tokenizer.bos_token_id == chosen_tokens[0]:
                chosen_tokens = chosen_tokens[1:]

            assert(tokens[-len(chosen_tokens):] == chosen_tokens), {"tokens": tokens, "chosen tokens": chosen_tokens}
            prompt_tokens = tokens[:-len(chosen_tokens)]
            prompt_tokens = prompt_tokens[-self.max_seq_len:]
            assert(len(prompt_tokens) <= self.max_seq_len)

            trunc_prompt = self.tokenizer.decode(prompt_tokens)
            #print(trunc_prompt)
            return trunc_prompt
        
        # chosen
        chosen_response = self.get_chosen(sample)
        chosen_response += self.tokenizer.eos_token
        self.tokenizer.truncation_side = "right"
        trunc_response_token = self.tokenizer.encode(chosen_response,
                                max_length=self.max_gen_len,
                                truncation=True)
        # check whether append <s>
        if trunc_response_token[0] == self.tokenizer.bos_token_id:
            trunc_response_token = trunc_response_token[1:]
        
        trunc_response_len = len(trunc_response_token)
        trunc_chosen_response = self.tokenizer.decode(trunc_response_token)

        prompt = self.get_prompt(sample)
        
        chosen_sentence = prompt + trunc_chosen_response

        self.tokenizer.truncation_side = "left"
        chosen_token = self.tokenizer(chosen_sentence,
                                max_length=self.max_seq_len,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt")
        
        #chosen_token["energy_labels"] = torch.FloatTensor([1.0]).unsqueeze(0)
        pad_len = (chosen_token["attention_mask"].eq(0)).sum()
        prompt_len = self.max_seq_len - pad_len - trunc_response_len
        chosen_token["prompt_lens"] = torch.LongTensor([prompt_len]).unsqueeze(0)


        # rejected
        rejected_response = self.get_rejected(sample)
        rejected_response += self.tokenizer.eos_token
        self.tokenizer.truncation_side = "right"
        trunc_response_token = self.tokenizer.encode(rejected_response,
                                max_length=self.max_gen_len,
                                truncation=True)
        # check whether append <s>
        if trunc_response_token[0] == self.tokenizer.bos_token_id:
            trunc_response_token = trunc_response_token[1:]
        
        trunc_response_len = len(trunc_response_token)
        trunc_rejected_response = self.tokenizer.decode(trunc_response_token)

        prompt = self.get_prompt(sample)
        
        rejected_sentence = prompt + trunc_rejected_response

        self.tokenizer.truncation_side = "left"
        rejected_token = self.tokenizer(rejected_sentence,
                                max_length=self.max_seq_len,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt")
        
        #rejected_token["energy_labels"] = torch.FloatTensor([0.0]).unsqueeze(0)
        pad_len = (rejected_token["attention_mask"].eq(0)).sum()
        prompt_len = self.max_seq_len - pad_len - trunc_response_len
        rejected_token["prompt_lens"] = torch.LongTensor([prompt_len]).unsqueeze(0)


        

        return chosen_token, rejected_token
    
    def collate(self, data):
        batch = {}
        for k in ["input_ids", "attention_mask", "prompt_lens"]:
            batch[k] = torch.cat([x[0][k] for x in data] +
                                 [x[1][k] for x in data], dim=0)
        
        return batch


class ExactDataset(BaseDataset):
    def __init__(self, samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, inference=False, vectors=None, dtype="fp32"):
        super().__init__(samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, inference=inference, vectors=vectors)
        self.dtype = DTYPE_MAP[dtype]
        if not inference:
            self.num_cands = len(self.get_prompt_and_completions(samples[0]))
    
    def vectorize(self, sample, inference=False):
        if inference:
            prompt = self.get_prompt(sample)
            chosen = self.get_completions(sample)[0]
            assert(chosen.strip() != ""), chosen
            tokens = self.tokenizer.encode(prompt + chosen)
            chosen_tokens = self.tokenizer.encode(chosen)

            if self.tokenizer.bos_token_id == chosen_tokens[0]:
                chosen_tokens = chosen_tokens[1:]

            assert(tokens[-len(chosen_tokens):] == chosen_tokens), {"tokens": tokens, "chosen tokens": chosen_tokens}
            prompt_tokens = tokens[:-len(chosen_tokens)]
            prompt_tokens = prompt_tokens[-self.max_seq_len:]
            assert(len(prompt_tokens) <= self.max_seq_len)

            trunc_prompt = self.tokenizer.decode(prompt_tokens)
            
            return trunc_prompt

        
        prompt = self.get_prompt(sample)
        rewards = self.get_rewards(sample)
        completions = self.get_completions(sample)

        inputs = []
        for completion, reward in zip(completions, rewards):
            self.tokenizer.truncation_side = "right"
            completion += self.tokenizer.eos_token
            trunc_completion_token = self.tokenizer.encode(completion,
                                                        max_length=self.max_gen_len,
                                                        truncation=True)
            
            # check whether append <s>
            if trunc_completion_token[0] == self.tokenizer.bos_token_id:
                trunc_completion_token = trunc_completion_token[1:]
            
            trunc_completion_len = len(trunc_completion_token)
            trunc_completion = self.tokenizer.decode(trunc_completion_token)

            sentence = prompt + trunc_completion
            self.tokenizer.truncation_side = "left"
            token = self.tokenizer(sentence,
                                    max_length=self.max_seq_len,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt")
            
            token["energy_labels"] = torch.FloatTensor([reward]).unsqueeze(0)
            pad_len = (token["attention_mask"].eq(0)).sum()
            prompt_len = self.max_seq_len - pad_len - trunc_completion_len
            token["prompt_lens"] = torch.LongTensor([prompt_len]).unsqueeze(0)
            
            inputs.append(token)

        return inputs
    
    def collate(self, data):
        batch = {}
        for k in ["input_ids", "attention_mask", "energy_labels", "prompt_lens"]:
            temp_features = []
            for group in data:
                temp_features.extend([x[k] for x in group])
            
            batch[k] = torch.cat(temp_features, dim=0)
        
        return batch


