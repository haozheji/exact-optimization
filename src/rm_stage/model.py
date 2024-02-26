import torch
from torch import nn


class RewardModel(nn.Module):
    def __init__(self,
                 base_model,
                 tokenizer,
                 compute_fp32_loss=False):
        
        super().__init__()
        self.config = base_model.config
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        

        self.rwtransformer = base_model
        

        self.PAD_ID = tokenizer.pad_token_id
        self.compute_fp32_loss = compute_fp32_loss
    
    def gradient_checkpointing_enable(self):
        self.rwtransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtransformer.gradient_checkpointing_disable()
    
    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                class_pos=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False,
                single=False):

        if single:
            return self.forward_single(input_ids=input_ids,
                                  past_key_values=past_key_values,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask,
                                  inputs_embeds=inputs_embeds,
                                  use_cache=use_cache)


        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        
        hidden_states = transformer_outputs[0] 
        rewards = self.v_head(hidden_states).squeeze(-1) # 2bsz x seq

        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs] # bs x seq
        rejected_rewards = rewards[bs:]

        chosen_end_scores = []
        rejected_end_scores = []

        if class_pos != None:
            chosen_class_pos = class_pos[:bs]
            rejected_class_pos = class_pos[bs:]

            c_last_rewards = torch.gather(chosen_rewards, dim=1, index=chosen_class_pos).squeeze(1)
            r_last_rewards = torch.gather(rejected_rewards, dim=1, index=rejected_class_pos).squeeze(1)

            if self.compute_fp32_loss:
                c_last_rewards = c_last_rewards.float()
                r_last_rewards = r_last_rewards.float()

            loss = -torch.nn.functional.logsigmoid(c_last_rewards -
                                                        r_last_rewards).mean()
            
            return {
                "loss": loss,
                "chosen_end_scores": c_last_rewards,
                "rejected_end_scores": r_last_rewards,
            }

        else:
            loss = 0.
            for i in range(bs):
                chosen_id = chosen_ids[i]
                rejected_id = rejected_ids[i]
                chosen_reward = chosen_rewards[i]
                rejected_reward = rejected_rewards[i]

                # find the last ind of chosen
                c_inds = (chosen_id == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(
                    c_inds
                ) > 0 else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence

                # find the last ind of rejected
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[0].item() if len(
                    r_inds) > 0 else seq_len

                c_last_reward = chosen_reward[c_ind - 1]
                r_last_reward = rejected_reward[r_ind - 1]

                chosen_end_scores.append(c_last_reward)
                rejected_end_scores.append(r_last_reward)

                if self.compute_fp32_loss:
                    c_last_reward = c_last_reward.float()
                    r_last_reward = r_last_reward.float()

                loss += -torch.nn.functional.logsigmoid(c_last_reward -
                                                        r_last_reward)
            
            loss = loss / bs
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)

            return {
                "loss": loss,
                "chosen_end_scores": chosen_end_scores,
                "rejected_end_scores": rejected_end_scores,
            }
        
    def forward_single(self, 
                       input_ids=None, 
                       attention_mask=None,
                       past_key_values=None,
                       position_ids=None,
                       head_mask=None,
                       inputs_embeds=None,
                       return_value_only=False,
                       #prompt_length=0,
                       use_cache=False):
        
        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        
        hidden_states = transformer_outputs[0] # bsz x seq x 1
        rewards = self.v_head(hidden_states).squeeze(-1) # bsz x seq
        if return_value_only:
            return rewards
        else:
            #assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = rewards.size(0)
            seq_len = input_ids.size(1)

            chosen_end_scores = []
            for i in range(bs):
                input_id = input_ids[i]
                reward = rewards[i]

                c_inds = (input_id == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(
                    c_inds
                ) > 0 else seq_len
                chosen_end_scores.append(reward[c_ind - 1])
            
            return {
                "values": rewards,
                "chosen_end_scores": torch.stack(chosen_end_scores)
            }

