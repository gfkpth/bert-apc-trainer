# APC detector

This project documents the creation of a deep-learning model for detecting adnominal pronoun constructions like *we linguists*. These consist of collocations of personal pronouns and (potentially further modified) nominals in a complex nominal phrase (for further references and theoretical and typological discussion see, e.g., [Höhn 2024](https://doi.org/10.1515/lingty-2023-0080)).

Automatic extraction of adnominal pronoun constructions (and other types of adnominal person marking) is very difficult because even prompting based on part-of-speech in appropriately tagged corpora yields many false positives. Recall improves for treebanks (like [Universal Dependencies](https://universaldependencies.org/)), but those are typically much smaller in size (and currently lack a universal annotation standard for adnominal person marking, cf. [Hoehn 2021](https://aclanthology.org/2021.udw-1.6/)).

I trained the transformer model [`bert-base-german-cased`](https://huggingface.co/google-bert/bert-base-german-cased) for token classification on a manually annotated dataset of sentences containing adnominal pronoun constructions. The token-level labels are automatically generated on the basis of raw data available in the dataset. 

The resulting model has a size of about 415 MB and achieves 99.7% accuracy on initial evaluation.

# Usage

Due to the space restrictions on GitHub, I cannot upload the trained model here. So for now this will only work if you get the model directly from me. I will update when I find time make the model available alternatively. 

For running the streamlit demo app (ensure that you have loaded an appropriate python environment):

```shell
pip install -r requirements.txt
cd src
streamlit run streamlit-ui.py 
```

Using the convenience function `string_in_apc_df_out` in `auxiliary.py` in python:

```python
def string_in_apc_df_out(string, trainer, tokenizer, language='german' , inclprons=True, num_proc=4):
```

Arguments: 
string
:   the input string (can be a complete file, currently this is automatically sentence-tokenized and split into context triplets)

trainer
:   a huggingface `Trainer` object

tokenizer
:   the matching tokenizer for the model provided to the `Trainer`

language
:   language of the data (currently only German is supported)

inclprons
:   bool indicating whether the returned dataframe should have separate rows for all personal pronouns (currently hard-coded) in addition to instances identified as APCs


Here is a minimal working example for its use:

```python
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification, AutoTokenizer

# load model
model_path = '../models/BERTfull_GER-APC'
tokenizer = AutoTokenizer.from_pretrained(model_path)
tunedmodel = AutoModelForTokenClassification.from_pretrained(model_path)

# create trainer
inference_args = TrainingArguments(
    output_dir="./inference_results",
    per_device_eval_batch_size=16,
    do_train=False,
    do_eval=False,
    do_predict=True
)

inference_trainer = Trainer(
    model=tunedmodel,
    args=inference_args,
    processing_class=tokenizer
)

# call convenience function 
df = string_in_apc_df_out('Komplexe Sätze sind für uns Linguisten eine Freude.', inference_trainer, tokenizer, language='german', inclprons=True, num_proc=6)
display(df)
```

# Overview

- [src/](src/): contains the source code
  - [EDA.ipynb](src/EDA.ipynb): some basic EDA on the original data (due to copyright restrictions the original training data cannot be shared)
  - [auxiliary.py](src/auxiliary.py): contains some helper functions and the APCData class, which 
  - [trainer.py](src/trainer.py): the code for training the 


## Training dataset

The original training dataset for German (about 45000 rows with slightly over 6000 manually annotated instances of APCs) was collected on the basis of several searches from [DWDS](https://www.dwds.de/) and manually annotated for the actual presence of APCs as well as several other linguistic properties. A similar dataset was also collected from the British National Corpus for English. 
I am very grateful to the annotators involved in different parts of creating the original training datasets: Andrea Schröter, Maya Galvez Wimmelmann and Carolin Kuna.

Unfortunately, due to copyright restrictions of the source corpora (or the rights provided by the original licensors for the corpus sources), these annotated datasets cannot be published. The hope is that the present model can be used to generate a similar dataset from freely available text or corpora.


