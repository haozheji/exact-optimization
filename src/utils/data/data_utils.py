import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
#from datasets import load_dataset
import numpy as np
import os
import hashlib
from itertools import chain
import tqdm

from deepspeed.accelerator import get_accelerator

from utils.data import raw_datasets
#from . import raw_datasets


def get_raw_dataset(dataset_name_path, seed, local_rank):
    '''
        dataset_name_path: should be in this format "name:path"
    '''
    assert(":" in dataset_name_path), "dataset_name_path: should be in this format `name:path`"
    dataset_name, dataset_path = dataset_name_path.split(":")

    # ==== tldr ====
    if "tldr/rw" == dataset_name:
        return raw_datasets.TldrRewardDataset(seed, local_rank, 
                                                dataset_name, dataset_path)

    if "tldr/pref" == dataset_name:
        return raw_datasets.TldrPrefDataset(seed, local_rank, 
                                                dataset_name, dataset_path)

    if "tldr/sft" == dataset_name:
        return raw_datasets.TldrSFTDataset(seed, local_rank, 
                                                dataset_name, dataset_path)

    # ==== hh ====
    if "hh/rw" == dataset_name:
        return raw_datasets.HHRewardDataset(seed, local_rank, 
                                                dataset_name, dataset_path)

    if "hh/pref" == dataset_name:
        return raw_datasets.HHPrefDataset(seed, local_rank, 
                                                dataset_name, dataset_path)

    elif "hh/sft" == dataset_name:
        return raw_datasets.HHSFTDataset(seed, local_rank, 
                                                dataset_name, dataset_path)
    # ==== imdb ====
    elif "imdb/class" == dataset_name:
        return raw_datasets.ImdbClassDataset(seed, local_rank, 
                                                dataset_name, dataset_path)

    elif "imdb/sft" == dataset_name:
        return raw_datasets.ImdbSFTDataset(seed, local_rank, 
                                                dataset_name, dataset_path)
    
    elif "imdb/pref" == dataset_name:
        return raw_datasets.ImdbPrefDataset(seed, local_rank, 
                                                dataset_name, dataset_path)
    
    elif "imdb/rw" == dataset_name:
        return raw_datasets.ImdbRewardDataset(seed, local_rank, 
                                                dataset_name, dataset_path)
    
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )

def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx

# for training
class BaseDataset(Dataset):
    def __init__(self, samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, vectors=None, **kwargs):
        super().__init__()
        self.samples = samples
        self.tokenizer = tokenizer
        self.raw_dataset = raw_dataset
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        if vectors is not None:
            self.vectors = vectors
        else:
            self.vectors = []
            for sample in tqdm.tqdm(self.samples, desc="vectorizing"):
                self.vectors.append(self.vectorize(sample, **kwargs))

    def get_prompt(self, sample):
        return self.raw_dataset.get_prompt(sample)

    # (pair) pref dataset method
    def get_chosen(self, sample):
        return self.raw_dataset.get_chosen(sample)

    def get_prompt_and_chosen(self, sample):
        return self.raw_dataset.get_prompt_and_chosen(sample)
    
    def get_rejected(self, sample):
        return self.raw_dataset.get_rejected(sample)
    
    def get_prompt_and_rejected(self, sample):
        return self.raw_dataset.get_prompt_and_rejected(sample)

    # oracle / reward dataset method
    def get_prompt_and_completions(self, sample):
        return self.raw_dataset.get_prompt_and_completions(sample)
    
    def get_completions(self, sample):
        return self.raw_dataset.get_completions(sample)
    
    def get_rewards(self, sample):
        return self.raw_dataset.get_rewards(sample)
    
    # class dataset method
    def get_text(self, sample):
        return self.raw_dataset.get_text(sample)
    
    def get_label(self, sample):
        return self.raw_dataset.get_label(sample)

    # base dataset method
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.vectors[idx]
    
    def vectorize(self, sample):
        pass 
        # need to be implemented
    
    def collate(self, data):
        pass 
        # need to be implemented

# for inference
# simple
class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def create_dataset(dataset_cls, local_rank, dataset_name, seed, tokenizer,
                   max_seq_len, max_gen_len, **kwargs):
    '''
        dataset_cls: customized Dataset class (inheritage of `BaseDataset`)
                    , must implement the following methods:
            - __len__
            - __getitem__
            - vectorize
            - collate
    '''

    raw_dataset = get_raw_dataset(dataset_name, seed, local_rank)
    train_samples = raw_dataset.get_train_data()
    train_dataset = dataset_cls(train_samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, **kwargs)
    
    eval_samples = raw_dataset.get_eval_data()
    eval_dataset = dataset_cls(eval_samples, raw_dataset, tokenizer, max_seq_len, max_gen_len, **kwargs)
    
    return train_dataset, eval_dataset


def load_dataset(local_rank,
                 dataset_cls,
                 data_path,
                 output_path,
                 seed,
                 tokenizer,
                 max_seq_len,
                 max_gen_len,
                 reload=False,
                 exp_type="",
                 **kwargs):
    '''
        This function is suppposed to be used in the training script
    '''

    # cache name
    os.makedirs(output_path, exist_ok=True)
    fname = data_path
    data_type = fname.split(":")[0]
    # get last model name (potential)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].split("/")[-1]
    fname = f"{exp_type}_{data_type}_seed{seed}_model-{tokenizer_name}_seqlen{max_seq_len}_genlen{max_gen_len}"
    fname = "_".join(fname.split("/"))
    
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    # distribute create cache flag to all devices
    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).to(
        get_accelerator().current_device_name())
    torch.distributed.all_reduce(buf_create_cache)    

    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        train_dataset, eval_dataset = create_dataset(
            dataset_cls, local_rank, data_path, seed, tokenizer,
            max_seq_len, max_gen_len, **kwargs)

        torch.save(train_dataset.vectors, train_fname)
        torch.save(eval_dataset.vectors, eval_fname)
    
    torch.distributed.barrier()

    train_vectors = torch.load(train_fname)
    eval_vectors = torch.load(eval_fname)
    raw_dataset = get_raw_dataset(data_path, seed, local_rank)
    train_dataset = dataset_cls(raw_dataset.get_train_data(), raw_dataset, tokenizer, max_seq_len, max_gen_len, vectors=train_vectors)
    eval_dataset = dataset_cls(raw_dataset.get_eval_data(), raw_dataset, tokenizer, max_seq_len, max_gen_len, vectors=eval_vectors)
    return train_dataset, eval_dataset