# APC detector

This project documents the creation of a deep-learning model for detecting adnominal pronoun constructions like *we linguists*. These consist of collocations of personal pronouns and (potentially further modified) nominals in a complex nominal phrase (for further references and theoretical and typological discussion see, e.g., ).

Automatic extraction of adnominal pronoun constructions (and other types of adnominal person marking) is very difficult because even prompting based on part-of-speech in appropriately tagged corpora yields many false positives. Recall improves for treebanks (like [Universal Dependencies](https://universaldependencies.org/)), but those are typically much smaller in size (and currently lack a universal annotation standard for adnominal person marking, cf. [Hoehn 2021](https://aclanthology.org/2021.udw-1.6/)).


# Overview



## Base dataset

- sentences (+ context sentences left/right) manually annotated for whether they contain an APC (and several other linguistic properties)
  - German: DWDS Kernkorpus
  - English: BNC
- manually annotated datasets cannot be published due to copyright restrictions of the source corpora (or rather the rights provided by the original licensors for the corpus sources)


## Problem: base datasets not in public domain
  - hence no sharing of those on GitHUB
- aim: train ML or DL model based on the annotated data to identify whether a given string contains an actual APC
  - minor extension (probably still core requirement): applicable to larger documents
  - output: a or b? 
    a) csv of potential APCs (as identified by classical NLP heuristics?) annotated with model prediction for actual "APC-ness"
    b) more complete list including document identifiers, reference to location in document, (maybe also pos-tagged version of texts)?

## MVP

- NLP pipeline with trained DL model that can identify sentences with APCs in a given text with at least a decent accuracy (85%?)
- save output as csv 
- work for either English or German
- apply this to public domain texts to be able to publish new annotated dataset for possible future use (research, training etc.) 

## Possible future extensions

- do manual check on newly generated dataset(s) to make it/them gold standard (rather optional, only do after bootcamp)
- extend to other language (i.e. English or German)
- create basic interface with streamlit
- extended (linguistically informed) EDA on base data
- applying unsupervised learning to base data
- consider expanding to yet other languages with different structures (e.g. Spanish or Greek with definite articles "nosotros los lingüistas") - unlikely due to limited availability of annotated data
- output data could also be saved in SQL db (or in linguistically-oriented formats like CLDF?)
- include a re-training functionality to allow easier refinement of the dataset with more/new data
- compare to performance of LLM with targeted prompt(s) on test-set
- publicly available or local; for volume reasons local would be preferable, but time-limitations might preclude this


## Challenges

- re-formatting existing data to identify relevant APCs within training sentences (bracketing?) before training
- set up NLP pipeline to identify potential APCs (pronoun + noun with possible intervening material) and pre-annotate raw text into appropriate list of candidates (intermediate pos-tagging? structural parsing? latter presumably error prone)
- getting the initial NLP pipeline to work and pre-filter raw text data the way I need it
- find additional datasets for annotation (Project Gutenberg?)



# Work packages for initial bringup

## WP1: Data preparation

- consider issues of data imbalance
  - full dataset contains many more 0s than 1s (about 10:1 at best as I recall, verify in EDA) -> reflection of real-world distribution
  - are there reasonable ways to oversample and augment the data similar to image classification?
  - downsampling would massively decrease the dataset size
  - also question: upsampling for subcategories?
    - background: in German 1st plural APCs are much more common than 1st or 2nd singular (and I think also than 2nd plural), again reflecting real data patterns

- alternative NLP strategies
  - option 1: marking APCs by bracketting in training data, requires pre-identification of potential candidates (e.g. based on automatic pos-tagging)
  - option 2: token-level supervision, can get by with less pre-tagging (although some may still be useful? investigate)
- current database consists of triples of sentences (more or less), arranged in columns: hit, previous and following
  - what to put forward for training: 
    - only (possibly annotated) sentence containing the potential candidate 
    - or all three columns
  - if all three, in which format?
    - as distinct table columns (vectorised strings) 
    - or as one complete vectorised string
  - include annotations for person/number/case of pronoun?
    - if so: would need to be inferenced for new input in additional pre-processing step
      - how to deal with ambiguities
- evaluate need for/benefit of using SQL db, e.g., for intermediate tables

### Output: functionality for data pre-processing and input

- requirements
  - input single string or list of documents 
  - be adaptable for new input as well (with potential previous step of pre-selecting candidates?)
  - output table in necessary format and with appropriate pre-annotation for use in model training and predictions
- desirable:
  - implement both options with ability to choose and compare performance
  - implementation as class (but functions could be ok)


## WP2: Model training and selection

- what to optimise for: accuracy, precision, recall, F1?
  - accuracy: maximise overall correctness (minimise all errors: false positives and false negatives)
  - precision: minimise false positives
  - recall: minimise false negatives (don't loose true positives)
  - F1: balanced measure for false positives/negatives
- model selection
  - expecting best performance from pre-trained transformer-models like BERT
    - potential issue: big model size, high compute cost
  - custom neural networks for smaller size
  - maybe also experiment with simpler ML algorithms for comparison?

### Output

- function (or class) for training models
- function (or method) for reporting and logging results
    - to log (minimum)
    - accuracy (train/test)
    - precision, recall, F1 (test)
    - model name
    - model parameters (separate columns for easier sortability or single field for more flexibility?)


## WP3: NLP pipeline for extracting candidate table

- ability to run inferencing on raw text without previous annotation
- possibly token-level annotation could forgo this?
  - however, some prior annotation might still be helpful, e.g. annotating personal pronouns as possible candidates, reducing need to unnecessarily process strings/sentences that don't contain any relevant pronouns?
- if two different routes implemented in WP1, this might be considered here as well (unless results are very clearly in favour of one option over the other)


### Output

- additional module of preprocessing/preselecting candidate sentences as appropriate to chosen strategy
  


## WP4: Application to set of new texts

- choose variety of texts for inferencing
- manual verification for some appropriately small chunk of data to assess reliability of results on raw data
- optional: extend input/output functions to allow generation of full results including meta-information on documents as available/provided, index in document


## WP5: Streamlit interface for core functionality (optional)

- pick files or folders for data to inference
- maybe: 
  - pick model to use
  - pick "strategy" (type of marking candidates in source, if any)
- provide output file
- maybe generate some basic statistics about results (counts)



## WP6: Extend pipeline and training to other language (optional)

- apply pipeline to the other language with available annotated data not used so far
- compare metrics
- implement in interface


## WP7: Compare to LLM performance (optional)

- set up local LLM
- build basic prompt pipeline to achieve classification and sorting results
- compare performance on test sample 