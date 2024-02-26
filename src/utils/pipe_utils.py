import torch
from transformers import Pipeline

class ClassifierModelPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        forward_kwargs = {}
        if "padding" in kwargs:
            preprocess_kwargs["padding"] = kwargs["padding"]
        if "max_length" in kwargs:
            preprocess_kwargs["max_length"] = kwargs["max_length"]

        if "return_list" in kwargs:
            postprocess_kwargs["return_form"] = kwargs["return_form"]
        
        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(self, text, **kwargs):
        tokens = self.tokenizer(text, 
                            return_tensors="pt",
                            truncation=True,
                            **kwargs)
        
        real_len = tokens["attention_mask"].sum(-1, keepdim=True)
        tokens["class_pos"] = real_len - 2 if tokens["input_ids"][0][
                    real_len -1].item() == self.tokenizer.eos_token_id else real_len - 1
        
        return tokens

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs, return_list=True):
        if return_list:
            return model_outputs["logodds"].tolist()
        return model_outputs["logodds"]




class RewardModelPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        forward_kwargs = {}
        if "padding" in kwargs:
            preprocess_kwargs["padding"] = kwargs["padding"]
        if "max_length" in kwargs:
            preprocess_kwargs["max_length"] = kwargs["max_length"]

        if "return_list" in kwargs:
            postprocess_kwargs["return_form"] = kwargs["return_form"]
        
        if "single" in kwargs:
            forward_kwargs["single"] = kwargs["single"]
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, text, **kwargs):
        return self.tokenizer(text, 
                            return_tensors="pt",
                            truncation=True,
                            **kwargs)

    def _forward(self, model_inputs, single=True):
        return self.model(**model_inputs, single=single)

    def postprocess(self, model_outputs, return_list=True):
        if return_list:
            return model_outputs["chosen_end_scores"].tolist()
        return model_outputs["chosen_end_scores"]


    
class LogprobsPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        forward_kwargs = {}
        if "padding" in kwargs:
            preprocess_kwargs["padding"] = kwargs["padding"]
        if "max_length" in kwargs:
            preprocess_kwargs["max_length"] = kwargs["max_length"]
        if "max_new_tokens" in kwargs:
            preprocess_kwargs["max_new_tokens"] = kwargs["max_new_tokens"]
        if "truncation" in kwargs:
            preprocess_kwargs["truncation"] = kwargs["truncation"]
        
        if "temperature" in kwargs:
            postprocess_kwargs["temperature"] = kwargs["temperature"]


        return preprocess_kwargs, forward_kwargs, postprocess_kwargs
    
    def preprocess(self, prompt_text_pair, **kwargs):
        prompt, text = prompt_text_pair

        max_new_tokens = kwargs.pop("max_new_tokens")
        self.tokenizer.truncation_side = "right"
        trunc_token = self.tokenizer.encode(text + self.tokenizer.eos_token, 
                                            truncation=True, 
                                            max_length=max_new_tokens)
        if trunc_token[0] == self.tokenizer.bos_token_id:
            trunc_token = trunc_token[1:]

        trunc_token_len = len(trunc_token)
        trunc_text = self.tokenizer.decode(trunc_token)

        self.tokenizer.truncation_side = "left"
        inputs = self.tokenizer(prompt + trunc_text,
                                        return_tensors="pt",
                                        **kwargs)
        
        pad_len = (inputs["attention_mask"].eq(0)).sum()
        max_length = kwargs["max_length"]
        #print(f"max: {max_length}, pad: {pad_len}, trunc: {trunc_token_len}")
        inputs["prompt_len"] = torch.LongTensor([[max_length - pad_len - trunc_token_len]])
        #print(inputs["prompt_len"])
        #print(inputs)
        return inputs
    
    def _forward(self, inputs):
        prompt_lens = inputs.pop("prompt_len")
        outputs = self.model(**inputs)
        return {"input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "logits": outputs["logits"],
                "prompt_lens": prompt_lens}
    
    def postprocess(self, inputs, temperature=1.0):
        # size: 1 x ...
        # pre-initialize something
        logsm = torch.nn.LogSoftmax(-1)
        

        input_ids, attention_mask, logits, prompt_lens = inputs["input_ids"], \
                                                         inputs["attention_mask"], \
                                                         inputs["logits"], \
                                                         inputs["prompt_lens"]

        labels = input_ids[..., 1:]#.contiguous()
        shifted_mask = attention_mask[..., 1:].bool()#.contiguous()
        logits = logits[..., :-1, :]#.contiguous()
        
        logprobs_warp = torch.gather(logsm(logits / temperature), 2, labels.unsqueeze(2)).squeeze(2)

        shifted_mask[0][:prompt_lens[0] - 1] = 0
        
        return logprobs_warp.masked_fill(shifted_mask.eq(0), 0).sum().item()