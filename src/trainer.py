# %%
from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import classification_report

from auxiliary import *

# setup BERT
modelname= "bert-base-german-cased"
tokenizer = BertTokenizerFast.from_pretrained(modelname)
model = BertForTokenClassification.from_pretrained(modelname, num_labels=3)  # if using BIO


#%% load and prepare data
dat = DataPrep()
dat.load_csv('../data/copyright/DWDS_APC_main_redux.csv')
dat.generate_biolabels_df()
dat.merge_apc_annotations()

train_dataset, val_dataset = dat.train_test_split_ds_mp(tokenizer)


# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    weight_decay=0.01,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
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
df = dat.get_df(subset='instance')
df.head()

# %% 
dat.df_index(14).tokenized_clean

# %% 

trainer.predict()