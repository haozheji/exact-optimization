# IMDB experiment

Follow the guidelines below to reproduce the experimental results on the IMDB dataset.

## Preprocess IMDB dataset

Download the raw IMDB dataset [aclImdb](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz). Run the following command to extract the file and preprocess the data:

```bash
tar -xvzf aclImdb_v1.tar.gz
cd exp/imdb_exp 
python preproc.py --raw /path/to/aclImdb
```

Create a `data` folder under `exp/imdb_exp` and move the processed data `imdb_class` and `imdb_prefix10` to `data`.

## Create preference dataset

### Train the oracle model

In the IMDB experiment, we train a binary sentiment classifier on the `imdb_class` data which contains the original sentiment label of the IMDB dataset. The logodds of this classifier is used as the oracle reward. The following command is used to train a sentiment classifier initialized from `gpt2-large` with checkpoint saved in `/local/path/to/gpt2`.

```bash
bash exp/imdb_exp/train_class.sh gpt2-large /local/path/to/gpt2
```

### Train the SFT model

The SFT model is trained on the `imdb_prefix10` data to complete the IMDB review given the first 10 bpe tokens of the review as prompt. The following command is used to train the SFT model initialized from `gpt2-large` with checkpoint saved in `/local/path/to/gpt2`.

```bash
bash exp/imdb_exp/train_sft.sh gpt2-large /local/path/to/gpt2
```

### Generate from the SFT model

Inference on the (whole) train and test set using the learned SFT model. Default sampling 2 completions given each prompt. The following commands are used to conduct inference on the train/test split using the trained SFT model saved in `models/gpt2-large_imdb/sft` (by default) with device ids `0,1`.

```bash
# train set
bash exp/imdb_exp/inference_sft.sh 0,1 imdb/sft:exp/imdb_exp/data/imdb_prefix10 train models/gpt2-large_imdb/sft

# test set
bash exp/imdb_exp/inference_sft.sh 0,1 imdb/sft:exp/imdb_exp/data/imdb_prefix10 test models/gpt2-large_imdb/sft
```

The generated data will be saved in `exp/imdb_exp/data/imdb_prefix10_sft`.

### Label preference by oracle reward model

Label the preference on the SFT generated data using the oracle reward model. The following commands are used to conduct inference on the train/test split of the SFT generated dataset using the oracle reward model saved in `models/gpt2-large_imdb/class` with device ids `0,1`.

```bash
# train set
bash exp/imdb_exp/inference_class.sh 0,1 exp/imdb_exp/data/imdb_prefix10_sft train models/gpt2-large_imdb/class label_pref

# test set
bash exp/imdb_exp/inference_class.sh 0,1 exp/imdb_exp/data/imdb_prefix10_sft test models/gpt2-large_imdb/class label_pref
```

The inference results will be saved in `exp/imdb_exp/data/imdb_prefix10_pref`.

## Create reward dataset

Besides directly learning from the preference dataset, we can also train a reward model to fit the preference and construct a reward dataset.

### Train the reward model

The reward model is trained on the preference dataset `imdb_prefix10_pref`. The following command is used to train the reward model initialized from `gpt2-large` with checkpoint saved in `/local/path/to/gpt2`.

```bash
bash exp/imdb_exp/train_rm.sh gpt2-large /local/path/to/gpt2 
```

### Compute reward by reward model

Label the preference on the SFT generated data using the reward model. The following commands are used to conduct inference on the train/test split of the SFT generated dataset using the reward model saved in `models/gpt2-large_imdb/rm` with device ids `0,1`.

```bash
# train set
bash exp/imdb_exp/inference_rm.sh 0,1 exp/imdb_exp/data/imdb_prefix10_sft train models/gpt2-large_imdb/rm label

# test set
bash exp/imdb_exp/inference_rm.sh 0,1 exp/imdb_exp/data/imdb_prefix10_sft test models/gpt2-large_imdb/rm label
```

The inference results will be saved in `exp/imdb_exp/data/imdb_prefix10_rw`.

## Alignment

First, convert the preference dataset to the same format of the reward dataset by running the following command:

```bash
python src/utils/data/pref_to_rw.py exp/imdb_exp/data/imdb_prefix10_pref
```

The results will be saved in `exp/imdb_exp/data/imdb_prefix10_pref_p2r`.

### EXO & DPO

Train the policy using the EXO algorithm, run commands:

```bash
# Any causal HuggingFace model (`AutoModelForCausalLM` class)
INIT_MODEL_NAME=gpt2-large
# local path to the SFT model
INIT_MODEL_PATH=models/gpt2-large_imdb/sft
# local path to the training data, e.g., imdb_prefix10_rw / imdb_prefix10_pref_p2r.
DATA_PATH=exp/imdb_exp/data/imdb_prefix10_rw
# supported loss type: exo-pref / exo-rw / dpo-pref / dpo-rw
LOSS_TYPE="exo-rw"
# number of contrastive samples, should not be greater than the number of completion candidates in the dataset.
NUM_CONTRASTIVE=2

bash exp/imdb_exp/train_exo.sh $INIT_MODEL_NAME $INIT_MODEL_PATH $DATA_PATH $LOSS_TYPE $NUM_CONTRASTIVE
```

Other hyperparameters for training can be specified in `exp/imdb_exp/train_exo.sh`. 

To train the policy using the DPO algorithm, simply change the `LOSS_TYPE` to either `dpo-pref` or `dpo-rw`.

The model checkpoints will be saved in `models/align_${LOSS_TYPE}_nc${$NUM_CONTRASTIVE}` by default.


### Inference

To conduct inference using the checkpoints saved during training (default use the first 10 checkpoints), run the following command to decode using the model checkpoints saved in, e.g., `models/gpt2-large_imdb/align_exo-rw_nc2/` with device ids `0,1`.

```bash
bash exp/imdb_exp/inference_align.sh 0,1 models/gpt2-large_imdb/align_exo-rw_nc2
```

The inference results will be saved in `exp/imdb_exp/data/imdb_prefix10_infer_res/models/gpt2-large_imdb/align_exo-rw_nc2/`.

To calculate the log probabilities on the generated samples by the saved model checkpoint, e.g., `ckpt1`, run the following command:

```bash
# path to the generated data to be evaluated
EVAL_DATA_PATH="exp/imdb_exp/data/imdb_prefix10_infer_res/models/gpt2-large_imdb/align_exo-rw_nc2/ckpt1"
# path to the model checkpoint 
MODEL_PATH="models/gpt2-large_imdb/align_exo-rw_nc2/ckpt1"

bash exp/imdb_exp/calc_logprobs.sh 0,1 $EVAL_DATA_PATH $MODEL_PATH
```

The result will be saved in `logprobs_ckpt1.json` under the same directory of the generated samples.

To calculate the log probabilities using the SFT model, simply change `MODEL_PATH="models/gpt2-large_imdb/sft`. The result will be saved in `logprobs_sft.json`.

### Evaluation

To evaluate the oracle reward of the generated samples saved in, e.g., `exp/imdb_exp/data/imdb_prefix10_infer_res/models/gpt2-large_imdb/align_exo-rw_nc2/ckpt1`, run the following command:

```bash
bash exp/imdb_exp/inference_class.sh \
0,1 \
exp/imdb_exp/data/imdb_prefix10_infer_res/models/gpt2-large_imdb/align_exo-rw_nc2/ckpt1 \
test \
models/gpt2-large_imdb/class \
eval
```

To evaluate the KL divergence $KL(\pi_\theta\|\pi_{\textrm{SFT}})$ between the trained policy and the SFT policy, run the following command:

```bash
python src/align_stage/eval_kl.py \
exp/imdb_exp/data/imdb_prefix10_infer_res/models/gpt2-large_imdb/align_exo-rw_nc2/ckpt1/logprobs_ckpt1.json \
exp/imdb_exp/data/imdb_prefix10_infer_res/models/gpt2-large_imdb/align_exo-rw_nc2/ckpt1/logprobs_sft.json
```

