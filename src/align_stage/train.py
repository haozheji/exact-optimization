import argparse
import os
import math
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import GPTNeoXLayer
from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.data.data_utils import load_dataset
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config, get_eval_ds_config
from utils.model.model_utils import create_hf_model, causal_lm_model_to_fp32_loss 
from utils.perf import print_throughput

from data import ExactDataset
from loss import dpo_loss, exact_loss

DTYPE_MAP = {"fp32": torch.float, "fp16": torch.float16}

def save_model(args, model, tokenizer, sub_folder):
    print_rank_0('saving model ...', args.global_rank)
    model = convert_lora_to_linear_layer(model)
    
    if args.global_rank == 0:
        save_hf_format(model, tokenizer, args, sub_folder=sub_folder, model_name_or_path=args.model_name_or_path)
    if args.zero_stage == 3:
        # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
        save_zero_three_model(model,
                                args.global_rank,
                                args.output_dir,
                                zero_stage=args.zero_stage)

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_name_path',
                        default=None,
                        help='Data name and path separated by colon')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--ref_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--num_save_checkpoint",
        type=int,
        default=1,
        help="Number of checkpoint to be saved"
    )
    parser.add_argument(
        "--beta_r",
        type=float,
        default=1.0,
        help="temperature of the reward"
    )
    parser.add_argument(
        "--beta_pi",
        type=float,
        default=1.0,
        help="coefficient of PoE: \pi ^ beta_pi * \pi_sft ^ ( 1 - beta_pi )"
    )
    parser.add_argument(
        "--label_smooth",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--num_contrastive",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="temperature of sft policy"
    )
    parser.add_argument(
        "--save_step_interval",
        type=int,
        default=-1,
        help="interval of saving steps, -1 for no saving"
    )
    parser.add_argument(
        "--max_iter_step",
        type=int,
        default=-1,
        help="maximal iteration step, -1 for no stop"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="exo-pref",
        help="exo-pref|exo-rw|dpo-pref|dpo-rw"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=512,
        help="The maximum generation (response) length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
        'If specified, loss is calculated in fp32.')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_name_path',
                        type=str,
                        default="exp_name:tb_path",
                        help="experiment name and tensorboard path separated by colon")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')

    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    parser.add_argument("--kernel_inject", action="store_true", help="true only for gpt-2")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
    
    args.global_rank = torch.distributed.get_rank()

    tb_name, tb_path = args.tensorboard_name_path.split(":")
    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype=args.dtype,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=tb_path,
                                    tb_name=tb_name)
    
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
    

    set_random_seed(args.seed)

    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=False, # better when set to slow
                                  add_special_tokens=None)
    
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            dropout=args.dropout,
                            torch_dtype=DTYPE_MAP[args.dtype])
    
    ref_model = AutoModelForCausalLM.from_pretrained(args.ref_name_or_path, torch_dtype=DTYPE_MAP[args.dtype])
    
    # prepare lora
    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)


    exp_type = "align"
    dataset_cls = ExactDataset
    train_dataset, eval_dataset = load_dataset(args.local_rank,
        dataset_cls,
        args.data_name_path,
        args.data_output_path,
        args.seed,
        tokenizer,
        args.max_seq_len,
        args.max_gen_len,
        exp_type=exp_type)
        
    assert(args.num_contrastive <= train_dataset.num_cands), f"num_contrastive {args.num_contrastive} should not be larger than num_cands {train_dataset.num_cands} of the dataset"
    assert(train_dataset.num_cands % args.num_contrastive == 0), f"num_cands {train_dataset.num_cands} of the dataset must be dividable by num_contrastive {args.num_contrastive}"
    

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=train_dataset.collate,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=eval_dataset.collate,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    
    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    
    ref_engine = deepspeed.init_inference(
            ref_model,
            tensor_parallel={"tp_size": 1,},
            dtype=DTYPE_MAP[args.dtype],
            replace_with_kernel_inject=args.kernel_inject,
        )
    ref_model = ref_engine.module

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    
    if args.save_step_interval != -1:
        save_step_interval = args.save_step_interval
    else:
        if len(train_dataloader) < args.num_save_checkpoint:
            print_rank_0("number of training batches < number of save checkpoints, default to save at every step.", args.global_rank)
            
            save_step_interval = 1
        else:
            save_step_interval = len(train_dataloader) // args.num_save_checkpoint 
    if args.max_iter_step == -1:
        args.max_iter_step = len(train_dataloader)
    
    ckpt_count = 1
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        import time
        for step, batch in enumerate(train_dataloader):
            start = time.time()
            
            batch = to_device(batch, device)
            prompt_lens = batch.pop("prompt_lens")
            energy_labels = batch.pop("energy_labels", None)

            
            outputs = model(**batch, use_cache=False)
            with torch.no_grad():
                
                ref_outputs = ref_model(**batch)
                logits = ref_outputs.logits.detach().clone()
                del ref_outputs
                torch.cuda.empty_cache()
                
            if "dpo" in args.loss_type:
                loss = dpo_loss(logits / args.temp, 
                                outputs.logits / args.temp, 
                                batch["attention_mask"], 
                                batch["input_ids"], 
                                prompt_lens,
                                energy_labels, 
                                N=args.num_contrastive, 
                                beta=args.beta_r, 
                                beta_model=args.beta_pi,
                                loss_type=args.loss_type)
            
            elif "exo" in args.loss_type:
                loss = exact_loss(logits / args.temp, 
                                    outputs.logits / args.temp, 
                                    batch["attention_mask"], 
                                    batch["input_ids"], 
                                    prompt_lens,
                                    energy_labels, 
                                    N=args.num_contrastive, 
                                    beta=args.beta_r, 
                                    beta_model=args.beta_pi,
                                    loss_type=args.loss_type)

            if args.print_loss:
                print(
                    f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
                )
            model.backward(loss)
            model.step()
            end = time.time()
            if torch.distributed.get_rank() == 0:
                print_throughput(model.module, args, end - start,
                                 args.global_rank)
            
            if (step + 1) % save_step_interval == 0:
                

                save_model(args, model, tokenizer, f"ckpt{ckpt_count}")
                ckpt_count += 1

            if (step + 1) % args.max_iter_step == 0:
                print_rank_0(f"Finished iteration {args.max_iter_step}, stop!", args.global_rank)
                exit()
        
    

if __name__ == "__main__":
    main()
