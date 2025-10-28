# %%
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, AutoModelForTokenClassification, AutoTokenizer, BitsAndBytesConfig



import numpy as np

import auxiliary as aux
from auxiliary import *


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


# %%

print(test.aligned_predictions_data)

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





# %%



# %% experimenting with quantized models ()

# GERMAN

# --- Paths --- 
fp_model_path = "../models/BERTfull_GER-APC"      # fine-tuned model 
quant_model_path = "../models/BERTfull_GER-APC_int8.pt"  

# --- Load fine-tuned model --- 
tokenizer = BertTokenizerFast.from_pretrained(fp_model_path) 
model = BertForTokenClassification.from_pretrained(fp_model_path) 
model.eval()  

# --- Move to CPU (quantization runs on CPU) --- 
model.cpu()  

# --- Fuse modules (helps quantization, optional for BERT) --- 
# BERT isn’t a convnet, so we skip fuse_model()  
# --- Prepare for static quantization --- 
model.qconfig = torch.quantization.get_default_qconfig("fbgemm") 
print("Using qconfig:", model.qconfig)  

# Skip embedding layers (set None or use float_qparams_weight_only)
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        module.qconfig = float_qparams_weight_only_qconfig
        
# Insert observers to collect stats 
torch.quantization.prepare(model, inplace=True)  

# --- Calibrate --- 
# Run a few representative batches to gather activation statistics 


def calibrate(model, tokenizer):     
	sentences =  get_sample_sents('../data/copyright/DWDS_APC_main_redux.csv',500,proportionate=False)
	for sent in sentences:         
		inputs = tokenizer(sent, return_tensors="pt", truncation=True, padding=True, max_length=128)         
		with torch.no_grad():             
			_ = model(**inputs)  
			

# %%
calibrate(model, tokenizer)  
			
# --- Convert to quantized version --- 

torch.quantization.convert(model, inplace=True)  
print("Model quantized successfully ✅")  

# --- Save to disk --- 
torch.save(model.state_dict(), quant_model_path) 
print(f"Quantized model saved to: {quant_model_path}")