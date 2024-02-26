from torch.utils.data import Dataset
from utils.data.data_utils import BaseDataset

class SFTDataset(BaseDataset):
    def __init__(self, samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, inference=False, vectors=None, skip_pad=True):
        super().__init__(samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, vectors=vectors, inference=inference, skip_pad=skip_pad)
    
    def vectorize(self, sample, inference=False, skip_pad=True):
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
                                
        # squeeze bsz dim
        # as the default collator will stack this dim
        chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(0)
        chosen_token["attention_mask"] = chosen_token[
                    "attention_mask"].squeeze(0)
        
        labels = chosen_token["input_ids"].clone()
        pad_len = (chosen_token["attention_mask"].eq(0)).sum()
        if skip_pad:
            labels[:-(trunc_response_len + pad_len)] = -100
            labels[-pad_len:] = -100
        return {
            "input_ids": chosen_token["input_ids"],
            "attention_mask": chosen_token["attention_mask"],
            "labels": labels,
        }
    
    def collate(self, data):
        pass
        # use transformers.default_data_collator

    

        
