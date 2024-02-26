# Anthropic-HH experiment

Follow the guidelines below to reproduce the experimental results on the Anthropic-HH dataset.

## Preprocess HH dataset

Download the HH dataset [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf).  This can be simply done using Hugginface datasets:
```python
from datasets import load_dataset

hh_dataset = load_dataset('Anthropic/hh-rlhf', data_dir='helpful-base')
hh_dataset['train'].to_json('hh-rlhf/train.jsonl')
hh_dataset['test'].to_json('hh-rlhf/test.jsonl')

```

Run the followiung command to preprocess the data:

```bash
cd exp/hh_exp 
python preproc.py /path/to/hh-rlhf
```

Create a `data` folder under `exp/hh_exp` and move the processed data `hh` to `data`.

## Create reward dataset

Besides directly learning from the preference dataset, we can also train a reward model to fit the preference and construct a reward dataset. 

### Train the reward model 

The reward model is trained on the preference dataset `hh`. The following command is used to train the reward model initialized from `pythia-2.8b` with checkpoint saved in `/local/path/to/pythia-2.8b`.

```bash
bash exp/hh_exp/train_rm.sh pythia-2.8b /local/path/to/pythia-2.8b
```

### Train the SFT model

The SFT model is trained on the chosen reponses in `hh` data given the dialogue history. The following command is used to train the SFT model initialized from `pythia-2.8b` with checkpoint saved in `/local/path/to/pythia-2.8b`.

```bash
bash exp/hh_exp/train_sft.sh pythia-2.8b /local/path/to/pythia-2.8b
```

### Generate from the SFT model

Inference on the (whole) train and test set using the learned SFT model. Default sampling 2 completions given each prompt. The following commands are used to conduct inference on the train/test split using the trained SFT model saved in `models/pythia-2.8b_hh/sft` (by default) with device ids `0,1,2,3`.

```bash
# train set
bash exp/hh_exp/inference_sft.sh 0,1,2,3 hh/sft:exp/hh_exp/data/hh train models/pythia-2.8b_hh/sft

# test set
bash exp/hh_exp/inference_sft.sh 0,1,2,3 hh/sft:exp/hh_exp/data/hh test models/pythia-2.8b_hh/sft
```

The generated data will be saved in `exp/hh_exp/data/hh_sft`.

### Compute reward by reward model

Label the preference on the SFT generated data using the reward model. The following commands are used to conduct inference on the train/test split of the SFT generated dataset using the reward model saved in `models/pythia-2.8b_hh/rm` with device ids `0,1,2,3`.

```bash
# train set
bash exp/hh_exp/inference_rm.sh 0,1,2,3 exp/hh_exp/data/hh_sft train models/pythia-2.8b_hh/rm label

# test set
bash exp/hh_exp/inference_rm.sh 0,1,2,3 exp/hh_exp/data/hh_sft test models/pythia-2.8b_hh/rm label
```

The inference results will be saved in `exp/hh_exp/data/hh_rw`.

## Alignment

First, convert the preference dataset to the same format of the reward dataset by running the following command:

```bash
python src/utils/data/pref_to_rw.py exp/hh_exp/data/hh
```

The results will be saved in `exp/hh_exp/data/hh_p2r`.

### EXO & DPO

Train the policy using the EXO algorithm, run commands:

```bash
# Any causal HuggingFace model (`AutoModelForCausalLM` class)
INIT_MODEL_NAME=pythia-2.8b
# local path to the SFT model
INIT_MODEL_PATH=models/pythia-2.8b_hh/sft
# local path to the training data, e.g., hh_rw / hh_p2r.
DATA_PATH=exp/hh_exp/data/hh_rw
# supported loss type: exo-pref / exo-rw / dpo-pref / dpo-rw
LOSS_TYPE="exo-rw"
# number of contrastive samples, should not be greater than the number of completion candidates in the dataset.
NUM_CONTRASTIVE=2

bash exp/hh_exp/train_exo.sh $INIT_MODEL_NAME $INIT_MODEL_PATH $DATA_PATH $LOSS_TYPE $NUM_CONTRASTIVE
```

Other hyperparameters for training can be specified in `exp/hh_exp/train_exo.sh`. 

To train the policy using the DPO algorithm, simply change the `LOSS_TYPE` to either `dpo-pref` or `dpo-rw`.

The model checkpoints will be saved in `models/align_${LOSS_TYPE}_nc${$NUM_CONTRASTIVE}` by default.


### Inference

To conduct inference using the checkpoints saved during training (default use the first 10 checkpoints), run the following command to decode using the model checkpoints saved in, e.g., `models/pythia-2.8b_hh/align_exo-rw_nc2` with device ids `0,1,2,3`.

```bash
bash exp/hh_exp/inference_align.sh 0,1,2,3 models/pythia-2.8b_hh/align_exo-rw_nc2
```

The inference results will be saved in `exp/hh_exp/data/hh_infer_res/models/pythia-2.8b_hh/align_exo-rw_nc2/`.

### Evaluation

To evaluate the reward of the generated samples saved in, e.g., `exp/hh_exp/data/hh_infer_res/models/pythia-2.8b_hh/align_exo-rw_nc2/ckpt1`, run the following command:

```bash
bash exp/hh_exp/inference_rm.sh \
0,1,2,3 \
exp/hh_exp/data/hh_infer_res/models/pythia-2.8b_hh/align_exo-rw_nc2/ckpt1 \
test \
models/pythia-2.8b_hh/rm \
eval
```