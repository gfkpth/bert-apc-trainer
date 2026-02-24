# BERT-APC training pipeline

This project is a training pipeline for fine-tuning pre-trained transformer models (currently used with BERT) for detecting adnominal pronoun constructions like *we linguists*. These constructions are collocations of personal pronouns and (potentially further modified) nominals in a complex nominal phrase (for further references and theoretical and typological discussion see, e.g., [Höhn 2024](https://doi.org/10.1515/lingty-2023-0080)).

## Background

Automatic extraction of adnominal pronoun constructions/APCs (and other types of adnominal person marking) is difficult with simple pattern-matching approaches even on POS-tagged corpora because of a large number of false positives. Recall improves with treebanks (like [Universal Dependencies](https://universaldependencies.org/)), but those are typically much smaller in size and currently lack a universal annotation standard for adnominal person marking, cf. [Hoehn 2021](https://aclanthology.org/2021.udw-1.6/), so annotation is not necessarily reliable.


# Quick Start

### Training a New Model

```python
from src.auxiliary import APCData

# Initialize with config
apc = APCData(model_config="german-bilou", config_file="src/config.yaml")

# Load and preprocess data
apc.load_data("../data/copyright/DWDS_APC_main_redux.csv")
apc.prepare_training_data()
apc.save_datasets()

# Train
apc.setup_trainer()
apc.run_trainer()

# Push to Hub with evaluation results
results = apc.evaluate_model()
apc.push_model_to_hub(
    commit_message="Initial release",
    revision="bilou-v1",
    eval_results=results
)
```

Evaluating a Trained Model

```python
from src.auxiliary import APCData

# Load from Hub or local path
apc = APCData.from_pretrained(
    "gfkpth/bert-apc-detector-ger",  # or "../models/bert-apc-detector-ger"
    model_config="german-bilou"
)

# Load preprocessed datasets
apc.load_datasets()

# Evaluate and update model card
results = apc.evaluate_model()
apc.push_model_to_hub(
    commit_message="Updated metrics",
    revision="bilou-v1.1",
    eval_results=results
)
```

## Configuration

All training parameters are defined in `src/config.yaml`:

- training: Global settings (learning rate, batch size, epochs, etc.)
- german-bilou, german-bio, english-bilou: Model-specific configs

Each config specifies:

- modelname: Base HuggingFace model ID
- datacsv: Path to training data
- label_scheme: BILOU or BIO labeling
- hub_model_id: Target HuggingFace repo ID
- hf_revision: Revision/branch for organizing versions


## Repository Structure


- [src/](src/): contains the source code
  - [trainer.py](src/trainer.py): main training and upload workflows
  - [auxiliary.py](src/auxiliary.py): Core APCData class with:
    - Data loading and preprocessing (tokenization, chunking)
    - Training pipeline (via HuggingFace Trainer)
    - Evaluation on test set
    - Hub integration (push/pull models)
    - Model card generation
    - from_pretrained() classmethod for loading trained models
  - config.yaml: Configuration for training/inference parameters

- [data/](data/): contains two test files for the streamlit application (training data or test output can also be expected here)
- [notebook/](notebook/): 
  - [EDA_BNC.ipynb](notebook/EDA_BNC.ipynb): some basic EDA on the original data (due to copyright restrictions the original training data cannot be shared)
  - [EDA_DWDS.ipynb](notebook/EDA_DWDS.ipynb): some basic EDA on the original data (due to copyright restrictions the original training data cannot be shared)
  - [GER_nounlist.csv](notebook/GER_nounlist.csv): detailed overview of frequency distribution of head nouns within the German/DWDS data
  - [GER_nounlist-plain.txt](notebook/GER_nounlist-plain.txt): a plain text list of the nouns attested in APC contexts in the German/DWDS dataset
- [presentation](presentation/): a presentation of the project in its version from Jul 2025
- [assets](assets/): assets for the presentation



## Model Evaluation & Versioning

Models are versioned via HuggingFace Hub revisions:

- Metrics are automatically incorporated into model cards
- Each training run can create a new revision (e.g., bilou-v1, bilou-v1.1)

Evaluation results (accuracy, F1, precision, recall) are logged in:

- Model card on Hub (human-readable)
- Local evaluation log (evaluation_logs/evaluation_log.txt)


## Supported Label Schemes

- BILOU: Begin, Inside, Last, Unit, Outside (5 labels)
- BIO: Begin, Inside, Outside (3 labels)

Both are implemented as separate model configurations in config.yaml.


## Models


- [bert-apc-detector-ger](https://huggingface.co/gfkpth/bert-apc-detector-ger) - trained on German DWDS corpus
- [bert-apc-detector-en](https://huggingface.co/gfkpth/bert-apc-detector-en) - trained on annotated data from English BNC corpus

Each model includes:

- Trained weights and tokenizer
- Model card with evaluation metrics
- Revision history for different label schemes/versions


I trained the transformer model [`bert-base-german-cased`](https://huggingface.co/google-bert/bert-base-german-cased) for token classification on a manually annotated dataset of sentences containing adnominal pronoun constructions extracted from the Kernkorpus of [DWDS](https://www.dwds.de/). The token-level labels are automatically generated on the basis of raw data available in the dataset. 

The resulting model has a size of about 415 MB and achieves 99.7% accuracy on initial evaluation.

Similarly, I trained the English [`bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) model on a similarly annotated dataset extracted from the [British National Corpus](https://www.natcorp.ox.ac.uk/).



## Training dataset

The original training dataset for German (about 45000 rows with slightly over 6000 manually annotated instances of APCs) was collected on the basis of pattern-matching searches from [DWDS](https://www.dwds.de/) and manually annotated for the actual presence of APCs as well as several other linguistic properties. A similar dataset was also collected from the British National Corpus for English. 
I am very grateful to the annotators involved in different parts of the creation of the original training datasets: Andrea Schröter, Maya Galvez Wimmelmann and Carolin Kuna.

Unfortunately, due to copyright restrictions of the source corpora (or the rights provided by the original licensors for the corpus sources), these annotated datasets cannot be published. The hope is that the present model can be used to generate a similar dataset from freely available text or corpora.


# Changelog

## 0.2.0 (24 Feb 2026)

- major refactoring

## 0.1.0 (Jul 2025)

- initial version