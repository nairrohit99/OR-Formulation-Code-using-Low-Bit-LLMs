# import all packages here
# bitsandbytes needed to load quantized models

import torch
from datasets import load_dataset
from transformers import pipeline, logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from transformers import DataCollatorForLanguageModeling
import bitsandbytes
import accelerate

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from trl import SFTTrainer
from transformers import TrainingArguments


import gc

# It helps in focusing only on critical issues by hiding less severe log messages
# Alternative Levels: Other verbosity levels include set_verbosity_debug(), set_verbosity_info(), and set_verbosity_warning().
logging.set_verbosity_error()

# set device to GPU:0
torch.cuda.set_device(0)

model_name = "../models/Qwen3-4B-Base-bnb-4bit"

# create pipeline for text generation
# https://huggingface.co/docs/transformers/main_classes/pipelines
# pipe = pipeline(
#     "text-generation",
#     model=model_name,
#     device_map="auto",
#     trust_remote_code=True
# )

# setting the prompts for both training and inference

train_prompt_template = """{} {}"""

inference_prompt_template = """{}"""

# load the dataset
from datasets import load_dataset

dataset = load_dataset("../datasets/OR-Instruct-Data-3K", split="train[1:]")

# load the tokenizer in a similar way
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    inputs = examples["prompt"]
    outputs = examples["completion"]
    texts = []
    for question,  response in zip(inputs, outputs):
        # Append the EOS token to the response if it's not already there
        if not response.endswith(tokenizer.eos_token):
            response += tokenizer.eos_token
        text = train_prompt_template.format(question,response)
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

print(dataset)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,   
    device_map="auto",  
    torch_dtype=torch.bfloat16,
    trust_remote_code=True             
)

model.config.use_cache = False
model.config.pretraining_tp = 1

from peft import LoraConfig, get_peft_model

# LoRA config
peft_config = LoraConfig(
    lora_alpha=32,                           # Scaling factor for LoRA
    lora_dropout=0.05,                       # Add slight dropout for regularization
    r=64,                                    # Rank of the LoRA update matrices
    bias="none",                             # No bias reparameterization
    task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Target modules for LoRA
)

# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

# training_arguments = TrainingArguments(
#     output_dir="output_1",
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=2,
#     optim="paged_adamw_32bit",
#     num_train_epochs=3,
#     logging_steps=0.2,
#     warmup_steps=10,
#     logging_strategy="steps",
#     learning_rate=2e-4,
#     fp16=False,
#     bf16=False,
#     group_by_length=True,
#     # set report_to to wandb or tensorboard to visualise
#     report_to="none",
#     lr_scheduler_type="cosine",
#     gradient_checkpointing=True
# )

training_arguments = TrainingArguments(
    output_dir="output_2",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,

    # Use integer for logging_steps (0.2 is invalid)
    logging_steps=50,  
    logging_strategy="steps",

    # Add TensorBoard logging directory
    logging_dir="./runs/lora_tensorboard",  

    warmup_steps=10,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,

    # Enable TensorBoard reporting
    report_to=["tensorboard"], 

    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
)

import trl

# Initialiseiing the SFT trainer 
# note all the arguments

trainer = SFTTrainer(
    model=model,
    # dataset_text_field="text",
    args=training_arguments,
    train_dataset=dataset,
    peft_config=peft_config,
    data_collator=data_collator
)

# training chalu

gc.collect()
torch.cuda.empty_cache()
model.config.use_cache = False
trainer.train()