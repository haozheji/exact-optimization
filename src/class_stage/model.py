import torch
from torch import nn


class ClassifierModel(nn.Module):
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
            self.c_head = nn.Linear(self.config.n_embd, 1, bias=False)
        
        self.clstransformer = base_model

        self.PAD_ID = tokenizer.pad_token_id
        self.compute_fp32_loss = compute_fp32_loss
    
    def gradient_checkpointing_enable(self):
        self.clstransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.clstransformer.gradient_checkpointing_disable()
    
    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                class_pos=None,
                labels=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):

        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.clstransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        
        hidden_states = transformer_outputs[0] 
        logodds = self.c_head(hidden_states).squeeze(-1) # 2bsz x seq

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0]
        seq_len = input_ids.shape[1]


        last_logodds = []
        logsigmoid = torch.nn.LogSigmoid()
        

        if class_pos != None:
            last_logodds = torch.gather(logodds, dim=1, index=class_pos).squeeze(1)
            
            if self.compute_fp32_loss:
                last_logodds = last_logodds.float()

            if labels is not None:
                labels = labels.to(logodds.dtype)
                # Binary classification
                loss = - labels * logsigmoid(last_logodds) - \
                        (1.0 - labels) * logsigmoid(- last_logodds)
                loss = loss.mean()
            

        else:
            loss = 0.
            for i in range(bs):
                input_id = input_ids[i]
                logodd = logodds[i]
                

                # find the last ind of chosen
                inds = (input_id == self.PAD_ID).nonzero()
                ind = inds[0].item() if len(
                    inds
                ) > 0 else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence

                last_logodd = logodd[ind - 1]

                last_logodds.append(last_logodd)

                if self.compute_fp32_loss:
                    last_logodd = last_logodd.float()

                if labels is not None:
                    labels = labels.to(logodds.dtype)
                    loss += - labels[i] * logsigmoid(last_logodd) \
                            - (1.0 - labels[i]) * logsigmoid(- last_logodd)
            
            loss = loss / bs
            last_logodds = torch.stack(last_logodds)
        
        if labels is not None:
            return {
                "loss": loss,
                "logodds": last_logodds,
            }
        
        else:
            return {
                "logodds": last_logodds,
            }