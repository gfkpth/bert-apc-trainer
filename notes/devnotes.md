

# Work packages for initial bringup

## WP1: Data preparation

- consider issues of data imbalance
  - full dataset contains many more 0s than 1s (about 10:1 at best as I recall, verify in EDA) -> reflection of real-world distribution
  - are there reasonable ways to oversample and augment the data similar to image classification?
  - downsampling would massively decrease the dataset size
  - also question: upsampling for subcategories?
    - background: in German 1st plural APCs are much more common than 1st or 2nd singular (and I think also than 2nd plural), again reflecting real data patterns
  - options:
    - class weights
    - use ChatGPT for synthetic data
  - NOTE: when using token-level classification, apparently the lack of balance is not an issue (in fact rather common for O to be much more common than B or I tags)
    - consequence: not an issue (at least for transformer-based token-level classification, which is probably the most effective way of doing this)
  

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

- DECISIONS:
  - German data
  - token-level supervised learning (BERT?)
  - no extra need for balancing dataset

### Output: functionality for data pre-processing and input

- requirements
  - input single string or list of documents 
  - be adaptable for new input as well (with potential previous step of pre-selecting candidates?)
  - output table in necessary format and with appropriate pre-annotation for use in model training and predictions
- desirable:
  - implement both options with ability to choose and compare performance
  - implementation as class (but functions could be ok)

- notes on implementation:
  - import csv
  - tokenize by word and create BIO labels
  - original data has one row per APC, so sentences with multiple APCs are doubled; a method is implemented to combine the BIO labels for those instances and remove the extra rows


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




# Other notes to self


https://huggingface.co/docs/transformers/main_classes/tokenizer

- Quick Overview of the Arguments
Argument	Description
truncation	If True, cuts input that exceeds max_length. Needed when working with long sequences.
padding	If "max_length", pads sequences up to max_length. If "longest" or True, pads dynamically.
max_length	Max number of tokens (after tokenization). Default for BERT is usually 512, but you can reduce it (e.g. to 64).
return_tensors	If set (e.g. "pt"), returns PyTorch (torch.Tensor) or TensorFlow tensors ("tf"). Useful for batching or model input.


- Choose max_length based on:

Average sentence + context length in your data:

Tokenize some examples and inspect their length.

If most are under 64 or 128 tokens, that’s a good safe cap.

Memory constraints:

Longer sequences → more memory use and slower training.

Try 64, 128, or 256 as starting points. BERT can handle up to 512, but rarely necessary.

Truncation side-effects:

If you truncate too short, some important tokens (maybe APCs or context) may get cut off.

Print some examples of truncated input to ensure no critical info is lost.

