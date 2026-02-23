# %%
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, AutoModelForTokenClassification, AutoTokenizer

import torch
from torch.quantization import quantize_dynamic, prepare, convert


import numpy as np

import auxiliary as aux
from auxiliary import *

#import importlib
#importlib.reload(aux)
#from auxiliary import *


# %% fine-tune on German dataset
# setup BERT
modelname= "bert-base-german-cased"
tokenizer = BertTokenizerFast.from_pretrained(modelname)
model = BertForTokenClassification.from_pretrained(modelname, num_labels=3)  # if using BIO

#%% load and prepare data
dat = aux.APCData(training=True, csvfile='../data/copyright/DWDS_APC_main_redux.csv',tokenizer=tokenizer,language='german')
dat.generate_biolabels_dataset()                 # about 7s with dataframes, about 4s(?) with datasets, far less with caching
dat.merge_apc_annotations()                 # about 0.03s

# %%

train_dataset, val_dataset, test_dataset = dat.tokenize_and_split(#cache_dir='./tokenization_cache',
                                                                         random_state=5,
                                                                         max_length=128, 
                                                                         overlap_size=32,
                                                                         num_proc=12)  # Get train/val/test splits        # 23s

# %%
dat.tokenized_dataset


# %%  old implementations
#train_dataset, val_dataset, test_dataset = dat.train_test_split_ds(tokenizer)               # 18.25s
#train_dataset, val_dataset, test_dataset = dat.train_test_split_ds_threaded(tokenizer,max_workers=4)      # 8.5s

# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    weight_decay=0.01,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# %%
trainer.train()


# %%

tunedmodel = trainer.model

output_model_path = '../models/BERTfixedtrain_3epochs'

tunedmodel.config.id2label = {
    0: "B-APC",
    1: "I-APC",
    2: "O"
}

tunedmodel.config.label2id = {
    "B-APC": 0,
    "I-APC": 1,
    "O": 2
}


tunedmodel.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path) # Always save the tokenizer with the model!

print(f"Fine-tuned model and tokenizer saved to: {output_model_path}")



# %%
tunedmodel.eval()

# Trainer for evaluation only
eval_trainer = Trainer(
    model=tunedmodel,
    args=training_args, # You might create new args for evaluation or re-use parts
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
eval_results = eval_trainer.evaluate()

# %%

eval_results



###########################
# train English model on BNC-data

# %%

modelname= "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(modelname)
model = BertForTokenClassification.from_pretrained(modelname, num_labels=3)  # if using BIO

#%% load and prepare data
dat = aux.APCData(training=True, csvfile='../data/copyright/BNC_combined_we_you_redux.csv',tokenizer=tokenizer,language='english')
dat.generate_biolabels_dataset()                 # about 7s with dataframes, about 4s(?) with datasets, far less with caching
dat.merge_apc_annotations()                 # about 0.03s

# %%

train_dataset, val_dataset, test_dataset = dat.tokenize_and_split(#cache_dir='./tokenization_cache',
                                                                         random_state=5,
                                                                         max_length=128, 
                                                                         overlap_size=32,
                                                                         num_proc=12)  # Get train/val/test splits        # 23s



# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results-tmp",
    weight_decay=0.01,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    overwrite_output_dir=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# %%
trainer.train()


# %%

tunedmodel = trainer.model

output_model_path = '../models/BERT-EN-manualfixes_v1'
tunedmodel.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path) # Always save the tokenizer with the model!

print(f"Fine-tuned model and tokenizer saved to: {output_model_path}")



# %%
tunedmodel.eval()

# Example: If you need to use it with a Trainer for evaluation only
eval_trainer = Trainer(
    model=tunedmodel,
    args=training_args, # You might create new args for evaluation or re-use parts
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
eval_results = eval_trainer.evaluate()

# %%

eval_results



# %%

output_model_path = '../models/BERT-EN-3epochs'
tokenizer = AutoTokenizer.from_pretrained(output_model_path)
tunedmodel = AutoModelForTokenClassification.from_pretrained(output_model_path)


inference_args = TrainingArguments(
    output_dir="./inference_results", # Required, but won't save much for predict
    per_device_eval_batch_size=16,
    do_train=False,
    do_eval=False,
    do_predict=True,
    # Consider fp16=True if your model supports it for faster inference on GPU
    # no_cuda_empty_cache=True, # Often helpful
)

inference_trainer = Trainer(
    model=tunedmodel,
    args=inference_args,
    processing_class=tokenizer
    # train_dataset=None, # No training
    # eval_dataset=None,  # No evaluation
    # compute_metrics=None # No metrics computation during predict, if not needed
)

#%%

df = string_in_apc_df_out("Ok let's test a more complex sentence. If you linguists are clever, you'll figure out what I'm doing. But us clever people should try to not be haughty. And definitely, we academics have a certain responsibility. I think you deans of study know this very well.", inference_trainer, tokenizer, language='english', inclprons=True, num_proc=6)
display(df)



###########################
# train English model on BNC-data

# %%

modelname= "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(modelname)
model = BertForTokenClassification.from_pretrained(modelname, num_labels=3)  # if using BIO

#%% load and prepare data
dat = aux.APCData(training=True, csvfile='../data/copyright/BNC_combined_we_you_redux.csv',tokenizer=tokenizer,language='english')
dat.generate_biolabels_dataset()                 # about 7s with dataframes, about 4s(?) with datasets, far less with caching
dat.merge_apc_annotations()                 # about 0.03s

# %%

train_dataset, val_dataset, test_dataset = dat.tokenize_and_split(#cache_dir='./tokenization_cache',
                                                                         random_state=5,
                                                                         max_length=128, 
                                                                         overlap_size=32,
                                                                         num_proc=12)  # Get train/val/test splits        # 23s



# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results-tmp",
    weight_decay=0.01,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    overwrite_output_dir=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# %%
trainer.train()


# %%

tunedmodel = trainer.model

output_model_path = '../models/BERT-EN-manualfixes_v1'
tunedmodel.config.id2label = {
    0: "B-APC",
    1: "I-APC",
    2: "O"
}

tunedmodel.config.label2id = {
    "B-APC": 0,
    "I-APC": 1,
    "O": 2
}
tunedmodel.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path) # Always save the tokenizer with the model!

print(f"Fine-tuned model and tokenizer saved to: {output_model_path}")



# %%
tunedmodel.eval()

# Example: If you need to use it with a Trainer for evaluation only
eval_trainer = Trainer(
    model=tunedmodel,
    args=training_args, # You might create new args for evaluation or re-use parts
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
eval_results = eval_trainer.evaluate()

# %%

eval_results



# %%

output_model_path = '../models/BERT-EN-3epochs'
tokenizer = AutoTokenizer.from_pretrained(output_model_path)
tunedmodel = AutoModelForTokenClassification.from_pretrained(output_model_path)


inference_args = TrainingArguments(
    output_dir="./inference_results", # Required, but won't save much for predict
    per_device_eval_batch_size=16,
    do_train=False,
    do_eval=False,
    do_predict=True,
    # Consider fp16=True if your model supports it for faster inference on GPU
    # no_cuda_empty_cache=True, # Often helpful
)

inference_trainer = Trainer(
    model=tunedmodel,
    args=inference_args,
    processing_class=tokenizer
    # train_dataset=None, # No training
    # eval_dataset=None,  # No evaluation
    # compute_metrics=None # No metrics computation during predict, if not needed
)

#%%

df = string_in_apc_df_out("Ok let's test a more complex sentence. If you linguists are clever, you'll figure out what I'm doing. But us clever people should try to not be haughty. And definitely, we academics have a certain responsibility. I think you deans of study know this very well.", inference_trainer, tokenizer, language='english', inclprons=True, num_proc=6)
display(df)
