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

## MVP:

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


## Challenges:

- re-formatting existing data to identify relevant APCs within training sentences (bracketing?) before training
- set up NLP pipeline to identify potential APCs (pronoun + noun with possible intervening material) and pre-annotate raw text into appropriate list of candidates (intermediate pos-tagging? structural parsing? latter presumably error prone)
- getting the initial NLP pipeline to work and pre-filter raw text data the way I need it
- find additional datasets for annotation (Project Gutenberg?)


# Project plan for initial bringup

## Overall concept

- Assess available annotated data for best preparation
- option 1: marking APCs by bracketting in training data, requires pre-identification of potential hits (e.g. based on automatic pos-tagging)
- option 2: token-level supervision, can get by with less pre-tagging (although some may still be useful? investigate)

## Work packages

### WP1: Data preparation


### WP2: Model training


### WP3: NLP pipeline for extracting candidate table