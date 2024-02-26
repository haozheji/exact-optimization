import math
import torch
from torch import nn
from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from torch.nn import CrossEntropyLoss, NLLLoss, LogSoftmax, Softmax, LogSigmoid





def dpo_loss(ref_logits, 
            model_logits, 
            attention_mask, 
            y_ids, 
            prompt_lens, 
            energy_labels, 
            N=2, 
            beta=1.0, 
            beta_model=1.0,
            loss_type="dpo"):
    
    # prepare
    bsz = ref_logits.size(0)
    logsigmoid = LogSigmoid()
    logsm = LogSoftmax(-1)
    device = ref_logits.device

    for i in range(attention_mask.size(0)):
        attention_mask[i, :prompt_lens[i][0]] = 0

    model_logprobs = torch.gather(logsm(model_logits)[:, :-1, :], 2, y_ids[:, 1:].unsqueeze(2)).squeeze(2) * attention_mask[:, 1:]
    ref_logprobs = torch.gather(logsm(ref_logits)[:, :-1, :], 2, y_ids[:, 1:].unsqueeze(2)).squeeze(2) * attention_mask[:, 1:]


    estimated_rewards_prefix = (model_logprobs - ref_logprobs).sum(1, keepdim=True)
    

    loss = 0.
    count = 0
    for estimated_rewards_prefix_group, energy_labels_group in zip(estimated_rewards_prefix.split(N), 
                                                                    energy_labels.split(N)):
        
        # prepare label
        if "rw" in loss_type:
            energy_labels_group = (energy_labels_group / beta).softmax(0)
        

        # num_contrastive * num_draw
        log_est_rewards_prefix_draw = (beta_model * estimated_rewards_prefix_group).log_softmax(0)

        loss = loss + ( - energy_labels_group * log_est_rewards_prefix_draw ).sum(0).mean()
        count += 1
        

    return loss / count


def exact_loss(ref_logits, 
                model_logits, 
                attention_mask, 
                y_ids, 
                prompt_lens, 
                energy_labels, 
                N=2, 
                beta=1.0, 
                beta_model=1.0,
                loss_type="exo-rw"):

    log_epsilon = -10 

    # prepare
    bsz = ref_logits.size(0)
    logsigmoid = LogSigmoid()
    logsm = LogSoftmax(-1)
    device = ref_logits.device

    for i in range(attention_mask.size(0)):
        attention_mask[i, :prompt_lens[i][0]] = 0

    model_logprobs = torch.gather(logsm(model_logits)[:, :-1, :], 2, y_ids[:, 1:].unsqueeze(2)).squeeze(2) * attention_mask[:, 1:]
    ref_logprobs = torch.gather(logsm(ref_logits)[:, :-1, :], 2, y_ids[:, 1:].unsqueeze(2)).squeeze(2) * attention_mask[:, 1:]


    estimated_rewards_prefix = beta_model * (model_logprobs - ref_logprobs).sum(1, keepdim=True)
    

    loss = 0.
    count = 0
    for estimated_rewards_prefix_group, energy_labels_group, has_loss_mask_group in zip(estimated_rewards_prefix.split(N), 
                                                                                        energy_labels.split(N), 
                                                                                        attention_mask[:, 1:].bool().split(N)):
        
        # prepare label
        if "rw" in loss_type:
            energy_labels_group = (energy_labels_group / beta).log_softmax(0)
        elif "pref" in loss_type:
            if energy_labels_group[0][0] > energy_labels_group[1][0]:
                energy_labels_group[0][0] = 0.
                energy_labels_group[1][0] = log_epsilon
            else:
                energy_labels_group[1][0] = 0.
                energy_labels_group[0][0] = log_epsilon
            
        last_ids = torch.zeros(N, 1).long().to(device)

        log_est_rewards_prefix_draw = estimated_rewards_prefix_group.gather(1, last_ids).log_softmax(0)

        loss = loss + (log_est_rewards_prefix_draw.exp() * (log_est_rewards_prefix_draw - energy_labels_group)).sum(0).mean()
        count += 1
    
    return loss / count
