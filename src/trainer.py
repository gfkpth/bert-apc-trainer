# %%
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, AutoModelForTokenClassification, AutoTokenizer

import numpy as np

import auxiliary as aux
from auxiliary import *

import importlib
importlib.reload(aux)
from auxiliary import *

# %%
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


#########################################
# CORE PART OF trainer.py ends here
# below is a testing ground

# %% testing

# %% load model
output_model_path = '../models/BERTfull_GER-APC'

tokenizer = AutoTokenizer.from_pretrained(output_model_path)

tunedmodel = AutoModelForTokenClassification.from_pretrained(output_model_path)

#%%

teststring = """
Das ist ein interessanter Text, den wir Linguisten sehr mögen. Natürlich könnte ich auch anders gestrickt sein, aber tatsächlich ist er für mich Syntaktiker besonders spannend.
Leider ist es aber auch nicht so leicht, sich Texte auszudenken. Später können wir das mal automatisch machen, aber wie ich euch Computerlinguisten einschätze, wisst ihr das sicherlich schon.
Erstmal bleibe ich bei der guten, alten Handarbeit. Das kann mir niemand verdenken, auch nicht ihr lachhaften Neider. Ihr langweiligen Spießer der Moderne, da wundert ihr euch, was? 
Das ist schon ein schwerer Satz, mit dem kann man euch Philister auch etwas quälen. Und du Idiot da in der Ecke brauchst gar nicht zu lachen. 
Für mich ernsthaften Akademiker sind das sehr ernste Fragen. Eigentlich haben wir Tiere nur im Zoo gesehen. Da wollen die mich Idiot nennen, ich glaub's gar nicht.
"""

# %%

test = APCData(training=False,language='german',strinput=teststring,tokenizer=tokenizer)
testset = test.prepare_for_inference(max_length=128)

# %% Inferencing option 1

# Assuming you have:
# fine_tuned_model (your reloaded model or trainer.model after training)
# test_dataset (your tokenized Hugging Face Dataset for testing)
# training_args (or a new TrainingArguments instance for inference config, e.g., batch size)
# compute_metrics (if you want to evaluate metrics during prediction)

# If you only want predictions (no evaluation), you can make a minimal TrainingArguments:
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

# %%

# Run prediction
predictions_output = inference_trainer.predict(testset)
# Debug your predictions_output structure
print("Type:", type(predictions_output))
print("Length:", len(predictions_output) if hasattr(predictions_output, '__len__') else 'No length')

if isinstance(predictions_output, list):
    print("First few shapes:", [np.array(p).shape for p in predictions_output[:5]])
    print("Sample prediction:", predictions_output[0] if len(predictions_output) > 0 else 'Empty')

#%%
test.import_predictions(predictions_output)

outputtable = test.generate_output_table()

print(outputtable)

resultdf = pd.DataFrame(outputtable)
display(resultdf)
resultdf.to_csv('test-output.csv')

# %% try extended output

output2df = pd.DataFrame(test.generate_output_table(include_personal_pronouns=True))
output2df.to_csv('test-output-withpronouns.csv')


# %%

test5 = APCData(training=False,language='german',strinput='Eure Probleme finden wir Hausmeister interessant.',tokenizer=tokenizer)
testset = test5.prepare_for_inference(max_length=128)

test5.import_predictions(inference_trainer.predict(testset))
output2df = pd.DataFrame(test5.generate_output_table(include_personal_pronouns=True))
output2df.to_csv('test5-output-withpronouns.csv')

# %% Testing 

df = string_in_apc_df_out('Wir Linguisten lachen gerne. Dagegen seid ihr Clowns die reinsten Griesgrame.', inference_trainer, tokenizer, language='german', inclprons=True, num_proc=6)
display(df)

# predictions_output will be a PredictionOutput object (or namedtuple) with:
# .predictions: numpy array of raw logits (batch_size, sequence_length, num_labels)
# .label_ids: numpy array of true labels (batch_size, sequence_length) - will be None if not provided in dataset
# .metrics: dictionary of computed metrics if compute_metrics was passed to Trainer

# Example: To get the predicted label IDs:
# predicted_logits = predictions_output.predictions
# predicted_labels = np.argmax(predicted_logits, axis=2)

# You'll then need to map these label IDs back to your string labels if you want:
# e.g., using dat.inttolabel


# %% or the pipeline

from transformers import pipeline

# Assuming fine_tuned_model and reloaded_tokenizer are available
# For token classification, specify the task
# You need to pass the model and tokenizer objects directly
token_classifier = pipeline(
    "token-classification",
    model=tunedmodel,
    tokenizer=tokenizer,
    # If your 'O' label is 2 and you don't want it in results, specify aggregation_strategy
    # aggregation_strategy="simple" or "first" or "average"
    # Example: To get proper entity spans:
    # aggregation_strategy="first" # or "average", etc.
)

# %%
result = token_classifier(teststring)
print(result)

#%%
# Run inference on a single string or a list of strings
example_text_1 = "This is a sentence with an APC word."
example_text_2 = "Another sentence for testing APCs."

results_1 = token_classifier(example_text_1)
results_2 = token_classifier([example_text_1, example_text_2]) # Process multiple at once

# The 'results' will be a list of dictionaries, where each dictionary
# represents a token and its predicted label, score, etc.
# Example structure:
# [[{'entity': 'O', 'score': 0.99, 'index': 1, 'word': 'This', 'start': 0, 'end': 4},
#   {'entity': 'O', 'score': 0.98, 'index': 2, 'word': 'is', 'start': 5, 'end': 7},
#   ...
#   {'entity': 'B-APC', 'score': 0.95, 'index': 7, 'word': 'APC', 'start': 23, 'end': 26},
#   {'entity': 'I-APC', 'score': 0.92, 'index': 8, 'word': 'word', 'start': 27, 'end': 31},
#   ...
# ]]

# Note: The pipeline uses model.config.id2label for outputting entity names.
# Make sure your model's config.id2label is correctly set up.
# If not, you might need to set it before saving:
# model.config.id2label = {v: k for k, v in dat.labeltoint.items()}
# model.config.label2id = dat.labeltoint
# Then save the model.