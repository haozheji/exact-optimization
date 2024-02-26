from torch.utils.data import Dataset
from utils.data.data_utils import BaseDataset
import torch

from transformers import Pipeline

class ClassDataset(BaseDataset):
    def __init__(self, samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, vectors=None):
        super().__init__(samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, vectors=vectors)
    
    def vectorize(self, sample):

        sentence = self.get_text(sample)
        label = 1.0 if self.get_label(sample) == "pos" else 0.0

        tokens = self.tokenizer(sentence,
                                max_length=self.max_seq_len,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt")
        real_len = tokens["attention_mask"].sum(-1, keepdim=True)
        tokens["class_pos"] = real_len - 2 if tokens["input_ids"][0][
                    real_len -1].item() == self.tokenizer.eos_token_id else real_len - 1

        tokens["labels"] = torch.FloatTensor([label])
        
        return tokens
    
    def collate(self, data):
        batch = {}
        for k in ["input_ids", "attention_mask", "class_pos", "labels"]:
            batch[k] = torch.cat([x[k] for x in data], dim=0)

        return batch


