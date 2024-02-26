# TL;DR experiment

Follow the guidelines below to reproduce the experimental results on the TL;DR dataset.

## Preprocess TL;DR dataset

Download the TL;DR preference dataset [summarize_from_feedback](https://huggingface.co/datasets/openai/summarize_from_feedback) and the TL;DR filtered dataset [openai-tldr-filtered](https://huggingface.co/datasets/UCL-DARK/openai-tldr-filtered) which is the same filtered version of the TL;DR dataset used in [*Learning to summarize from human feedback*](https://arxiv.org/abs/2009.01325). This can be simply done using Huggingface datasets:
```python
from datasets import load_dataset

tldr_dataset = load_dataset("openai/summarize_from_feedback", 'comparisons')
tldr_dataset['train'].to_json('summarize_from_feedback/train.json')
tldr_dataset['validation'].to_json('summarize_from_feedback/test.json')

tldr_filtered_dataset = load_dataset("UCL-DARK/openai-tldr-filtered")
tldr_filtered_dataset['train'].to_json('openai-tldr-filtered/train.jsonl')
tldr_filtered_dataset['test'].to_json('openai-tldr-filtered/test.jsonl')
```

Run the followiung command to preprocess the data:

```bash
cd exp/tldr_exp
python preproc_pref.py /path/to/summarize_from_feedback
python preproc_filter.py /path/to/openai-tldr-filtered
```

Create a `data` folder under `exp/tldr_exp` and move the processed data `tldr` and `tldr_filtered` to `data`.

## Create reward dataset

Besides directly learning from the preference dataset, we can also train a reward model to fit the preference and construct a reward dataset. 

### Train the reward model 

The reward model is trained on the preference dataset `tldr`. The following command is used to train the reward model initialized from `pythia-2.8b` with checkpoint saved in `/local/path/to/pythia-2.8b`.

```bash
bash exp/tldr_exp/train_rm.sh pythia-2.8b /local/path/to/pythia-2.8b
```

### Train the SFT model

The SFT model is trained on the chosen reponses in `tldr_filtered` data given the dialogue history. The following command is used to train the SFT model initialized from `pythia-2.8b` with checkpoint saved in `/local/path/to/pythia-2.8b`.

```bash
bash exp/tldr_exp/train_sft.sh pythia-2.8b /local/path/to/pythia-2.8b
```

### Generate from the SFT model

Inference on the (whole) train and test set using the learned SFT model. Default sampling 2 completions given each prompt. The following commands are used to conduct inference on the train/test split using the trained SFT model saved in `models/pythia-2.8b_tldr/sft` (by default) with device ids `0,1,2,3`.

```bash
# train set
bash exp/tldr_exp/inference_sft.sh 0,1,2,3 tldr/sft:exp/tldr_exp/data/tldr train models/pythia-2.8b_tldr/sft

# test set
bash exp/tldr_exp/inference_sft.sh 0,1,2,3 tldr/sft:exp/tldr_exp/data/tldr test models/pythia-2.8b_tldr/sft
```

The generated data will be saved in `exp/tldr_exp/data/tldr_sft`.

### Compute reward by reward model

Label the preference on the SFT generated data using the reward model. The following commands are used to conduct inference on the train/test split of the SFT generated dataset using the reward model saved in `models/pythia-2.8b_tldr/rm` with device ids `0,1,2,3`.

```bash
# train set
bash exp/tldr_exp/inference_rm.sh 0,1,2,3 exp/tldr_exp/data/tldr_sft train models/pythia-2.8b_tldr/rm label

# test set
bash exp/tldr_exp/inference_rm.sh 0,1,2,3 exp/tldr_exp/data/tldr_sft test models/pythia-2.8b_tldr/rm label
```

The inference results will be saved in `exp/imdb_exp/data/tldr_rw`.

## Alignment

First, convert the preference dataset to the same format of the reward dataset by running the following command:

```bash
python src/utils/data/pref_to_rw.py exp/tldr_exp/data/tldr
```

The results will be saved in `exp/tldr_exp/data/tldr_p2r`.

### EXO & DPO

Train the policy using the EXO algorithm, run commands:

```bash
# Any causal HuggingFace model (`AutoModelForCausalLM` class)
INIT_MODEL_NAME=pythia-2.8b
# local path to the SFT model
INIT_MODEL_PATH=models/pythia-2.8b_tldr/sft
# local path to the training data, e.g., tldr_rw / tldr_p2r.
DATA_PATH=exp/tldr_exp/data/tldr_rw
# supported loss type: exo-pref / exo-rw / dpo-pref / dpo-rw
LOSS_TYPE="exo-rw"
# number of contrastive samples, should not be greater than the number of completion candidates in the dataset.
NUM_CONTRASTIVE=2

bash exp/tldr_exp/train_exo.sh $INIT_MODEL_NAME $INIT_MODEL_PATH $DATA_PATH $LOSS_TYPE $NUM_CONTRASTIVE
```

Other hyperparameters for training can be specified in `exp/tldr_exp/train_exo.sh`. 

To train the policy using the DPO algorithm, simply change the `LOSS_TYPE` to either `dpo-pref` or `dpo-rw`.

The model checkpoints will be saved in `models/align_${LOSS_TYPE}_nc${$NUM_CONTRASTIVE}` by default.

### Inference

To conduct inference using the checkpoints saved during training (default use the first 10 checkpoints), run the following command to decode using the model checkpoints saved in, e.g., `models/pythia-2.8b_tldr/align_exo-rw_nc2` with device ids `0,1,2,3`.

```bash
bash exp/tldr_exp/inference_align.sh 0,1,2,3 models/pythia-2.8b_tldr/align_exo-rw_nc2
```

The inference results will be saved in `exp/tldr_exp/data/tldr_infer_res/models/pythia-2.8b_tldr/align_exo-rw_nc2/`.

### Evaluation

To evaluate the reward of the generated samples saved in, e.g., `exp/tldr_exp/data/tldr_infer_res/models/pythia-2.8b_tldr/align_exo-rw_nc2/ckpt1`, run the following command:

```bash
bash exp/tldr_exp/inference_rm.sh \
0,1,2,3 \
exp/tldr_exp/data/tldr_infer_res/models/pythia-2.8b_tldr/align_exo-rw_nc2/ckpt1 \
test \
models/pythia-2.8b_tldr/rm \
eval
```