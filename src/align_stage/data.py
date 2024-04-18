from torch.utils.data import Dataset
from utils.data.data_utils import BaseDataset
import torch

DTYPE_MAP = {"fp32": torch.float, "fp16": torch.float16}


class ExactDataset(BaseDataset):
    def __init__(self, samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, inference=False, vectors=None, dtype="fp32"):
        super().__init__(samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, inference=inference, vectors=vectors)
        self.dtype = DTYPE_MAP[dtype]
        if not inference:
            self.num_cands = len(self.get_prompt_and_completions(samples[0]))
    
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


