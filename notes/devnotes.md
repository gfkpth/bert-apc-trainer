# Implementation log

- currently `tokenize_dataset` hangs for unknown reasons
  - before this problem started occurring, `tokenize_and_split_native` hung on trying to get the splits right

- basic version of APCData class is done
  - modified to work with datasets
  - now using BERT tokenisation directly for creation of BIO labels
  - pre-hit-after-collation and full tokenisation now working in batches
    - before: issues with "Token indices sequence length is longer than the specified maximum sequence length for this model (2866 > 512). Running this sequence through the model will result in indexing errors"
    - was not an issue when using initial word_tokenisation, but it might be that subsequent BERT tokenisation wasn't working as intended on that prior setup
    - implementation currently bugged
  - newly implemented attempt at method to integrate inference results, restoring table format
    - tentatively integrating option to also collect pronouns that are not part of APCs (useful for calculating ratio)

- initial training results from 2025-06-30: promising accuracy around 99.6% (both on validation and test)
  - this was on previous implementation with word-tokenisation, but not proper output processing to make results usable again

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
  - import raw text (for application to new data)
  - tokenize by word and create BIO labels
  - original data has one row per APC, so sentences with multiple APCs are doubled; a method is implemented to combine the BIO labels for those instances and remove the extra rows
  - train-val-test split
  - switch for determining whether instance is for training data (i.e. annotated) or for inference

- implementation change 2025-06-30: using datasets instead of dataframes as core data management tool in APCData class 


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




# Refactoring APCData for full BERT-tokenisation

```python
import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict
from nltk.tokenize import sent_tokenize # Keep for load_raw_text if it's external to BERT tokenizer
import multiprocessing as mp
import time
from functools import wraps
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from sklearn.model_selection import train_test_split # Assuming this is available

# Dummy timeit decorator for demonstration. Replace with your actual implementation.
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds to execute.")
        return result
    return wrapper


class APCData:
    def __init__(self, training=False, language="german", csvfile=None, strinput=None, tokenizer=None):
        # setting core properties of data
        self.language = language  # marks (main) language of dataset
        self.training = training  # markes whether data set is annotated as training data
        self.dataset = None  # Primary data storage
        self.df = None  # Keep for backward compatibility or if specific Pandas operations are truly needed
        self.tokenizer = tokenizer # Store the tokenizer for consistent use

        if csvfile:
            print(f'Loading csv file {csvfile} into dataset')
            self.dataset = self.load_csv(csvfile)
            self.df = self.dataset.to_pandas()  # Optionally keep a df for some operations, but minimize
        elif strinput:
            print(f'Loading raw string into dataset')
            self.dataset = self.load_raw_text(strinput)
            self.df = self.dataset.to_pandas()  # Optionally keep a df
        else:
            print('No data provided, starting with empty dataset')
            self.dataset = Dataset.from_dict({'ContextBefore': [], 'Hit': [], 'ContextAfter': [], 'instance': [], 'APC': []})
            self.df = pd.DataFrame(columns=['ContextBefore', 'Hit', 'ContextAfter', 'instance', 'APC'])

        # setting up mappings
        self.labeltoint = {
            'B-APC': 0,
            'I-APC': 1,
            'O': 2
        }
        self.inttolabel = {
            0: 'B-APC',
            1: 'I-APC',
            2: 'O'
        }

        self.textcols = ['ContextBefore', 'Hit', 'ContextAfter', 'APC']

    # INPUT METHODS

    # loading a csv (assumed to be annotated)
    def load_csv(self, path):
        try:
            dataset = load_dataset('csv', data_files=path, split='train')
            return dataset
        except Exception as e:
            print(f"Error loading CSV directly into Dataset: {e}. Falling back to Pandas.")
            df = pd.read_csv(path, index_col='ID', dtype={'instance': int})
            return Dataset.from_pandas(df, preserve_index=False)

    # loading raw data for inferencing
    def load_raw_text(self, text):
        if self.training:
            print('load_raw_text() is designed to load non-annotated, raw-text datasets. To use it, please instantiate the APCData object with `training=False`.')
            return
        else:
            sentences = sent_tokenize(text, language=self.language)
            tmplist = []
            for i in range(len(sentences)):
                context_before = sentences[i - 1] if i > 0 else ""
                hit = sentences[i]
                context_after = sentences[i + 1] if i < len(sentences) - 1 else ""
                tmplist.append({
                    'ContextBefore': context_before,
                    'Hit': hit,
                    'ContextAfter': context_after,
                    'instance': 0,
                    'APC': ''
                })
            return Dataset.from_list(tmplist)

    ########################
    # BASIC DATAFRAME OUTPUT

    def get_dataset(self, subset=None):
        if subset == 'instance':
            return self.dataset.filter(lambda x: x['instance'] == 1)
        else:
            return self.dataset

    def get_df(self, subset=None):
        if self.df is None:
            if self.dataset:
                print("Converting dataset to pandas DataFrame. Consider using get_dataset() directly for efficiency.")
                self.df = self.dataset.to_pandas()
            else:
                return None

        if subset == 'instance':
            return self.df[self.df.instance == 1]
        else:
            return self.df

    def df_index(self, index):
        return self.df.loc[index]

    ##################################
    # Generating and combining data

    @timeit
    def generate_biolabels_dataset(self):
        """
        Tokenize "Hit" column with the BERT tokenizer and create BIO labels for it.
        The labels are directly aligned with BERT's subword tokens.
        """
        if self.dataset is None:
            raise ValueError("No dataset available. Load data first.")
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Please provide a tokenizer when initializing APCData or set it before calling this method.")

        def process_example(example):
            hit_text = str(example['Hit']) if pd.notnull(example['Hit']) else ""
            apc_string = str(example['APC']) if pd.notnull(example['APC']) else ""

            # Tokenize the 'Hit' sentence with BERT tokenizer, getting offsets
            # This is crucial for aligning with the APC string
            tokenized_hit = self.tokenizer(
                hit_text,
                return_offsets_mapping=True,
                add_special_tokens=False, # We'll add special tokens later in tokenize_dataset if needed for full sequence
                truncation=False, # Don't truncate here, handle it in tokenize_dataset
                padding=False # Don't pad here, handle it in tokenize_dataset
            )

            # Initialize labels with 'O' for all tokens in the 'Hit' sentence
            labels = ["O"] * len(tokenized_hit['input_ids'])

            # If it's an instance with an APC, assign B-APC/I-APC labels
            if example['instance'] == 1 and apc_string:
                # Find the character start and end of the APC within the Hit text
                apc_start_char = hit_text.find(apc_string)
                apc_end_char = apc_start_char + len(apc_string)

                if apc_start_char != -1: # APC string found in Hit
                    is_inside_apc = False
                    for i, (char_start, char_end) in enumerate(tokenized_hit['offset_mapping']):
                        # Check if token overlaps with APC
                        # A token is part of APC if its span is completely within or overlaps with APC span
                        # For BIO, we want tokens *fully within* the APC span (or starting it).
                        # Adjust char_start and char_end for token if it's (0,0) due to special token or padding
                        if char_start is None or char_end is None: # Should not happen with add_special_tokens=False
                            continue

                        # Check for overlap: [token_start, token_end) vs [apc_start, apc_end)
                        if max(char_start, apc_start_char) < min(char_end, apc_end_char):
                            # This token is inside or overlaps with the APC span
                            if not is_inside_apc:
                                labels[i] = "B-APC" # First token of the APC
                                is_inside_apc = True
                            else:
                                labels[i] = "I-APC" # Subsequent tokens of the APC
                        else:
                            # If we were inside an APC and now we are outside, reset flag
                            if is_inside_apc:
                                is_inside_apc = False # End of APC for previous tokens
                            # labels[i] remains "O"

            # Store the BERT tokenized Hit sequence and its aligned labels
            example['bert_tok_Hit_input_ids'] = tokenized_hit['input_ids']
            example['bert_tok_Hit_attention_mask'] = tokenized_hit['attention_mask']
            example['bert_tok_Hit_offsets'] = tokenized_hit['offset_mapping']
            example['bert_biolabels'] = labels

            return example

        # Use .map() for efficient processing
        # Ensure num_proc is correctly handled (e.g., set to 1 if tokenizer isn't picklable or for debugging)
        # Using num_proc > 1 requires the tokenizer to be picklable, which it usually is.
        self.dataset = self.dataset.map(
            process_example,
            num_proc=mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1,
            desc="Generating BERT-aligned BIO labels for 'Hit'"
        )

        # Remove original word_tokenize-based tokenization and biolabels if no longer needed
        # self.dataset = self.dataset.remove_columns(['tok_ContextBefore', 'tok_Hit', 'tok_ContextAfter', 'tok_APC', 'biolabels'])
        print("Generated BERT-aligned BIO labels.")


    # The `generate_biolabels_single` function is no longer strictly needed if `generate_biolabels_dataset`
    # handles the BERT tokenization directly. You can remove it.
    # def generate_biolabels_single(...)


    def find_duplicate_hits(self):
        # This method still relies on 'Hit' text for duplication, which is fine.
        if self.df is None:
            return []
        else:
            self.df = self.dataset.to_pandas()

        step1 = self.df[self.df.instance == 1]
        step2 = step1[step1.duplicated(subset=['Hit'], keep=False)]
        groups = step2.sort_values('Hit').groupby('Hit').groups
        print(f'Found {len(groups)} groups of duplicate rows')
        return [tuple(indices) for indices in step2.sort_values('Hit').groupby('Hit').groups.values()]

    @timeit
    def merge_apc_annotations(self):
        if self.df is None:
            print("No data to merge.")
            return
        else:
            self.df = self.dataset.to_pandas() # Ensure df is up-to-date with dataset

        id_tuples = self.find_duplicate_hits()

        print("Merging BERT-aligned BIO-labels for duplicates")
        processed = 0
        for t in id_tuples:
            # We now merge 'bert_biolabels'
            consolidated_labels = list(self.df.loc[t[0]]['bert_biolabels']) # Make a mutable copy

            for i_idx in t[1:]: # Iterate through subsequent duplicate indices
                processed += 1
                tmp_bio = self.df.loc[i_idx]['bert_biolabels']

                # Ensure labels lists have the same length (should be if they are duplicates)
                if len(consolidated_labels) != len(tmp_bio):
                    print(f"Warning: Label length mismatch for duplicate rows {t[0]} and {i_idx}. Skipping merge for this pair.")
                    continue

                for count in range(len(consolidated_labels)):
                    # If the temporary label is not 'O', it's an APC tag, so use it.
                    # This ensures 'B-APC' or 'I-APC' overwrites 'O'.
                    if tmp_bio[count] != 'O':
                        consolidated_labels[count] = tmp_bio[count]

            # Update the first instance of the duplicate group with the consolidated labels
            self.df.at[t[0], 'bert_biolabels'] = consolidated_labels

        print(f"Processed {processed} duplicate rows")

        print("Removing now-redundant rows")
        self.df = self.df.drop_duplicates(subset='Hit', keep='first')

        # Convert back to Dataset after modification
        self.dataset = Dataset.from_pandas(self.df, preserve_index=False)
        print("Dataset updated after merging and removing duplicates.")

    def create_dataset_from_df(self):
        """Convert processed DataFrame to HuggingFace Dataset"""
        if self.df is None:
            raise ValueError("No dataframe available. Load data first.")
        self.dataset = Dataset.from_pandas(self.df)
        return self.dataset

    @timeit
    def tokenize_dataset(self, tokenizer, num_proc=None, batch_size=1000,
                        max_length=64, cache_dir=None):
        """
        General tokenization method that works for both training and inference data.
        Uses BERT-aligned labels if self.training is True.
        """
        if tokenizer.is_fast:
            effective_num_proc = None
            print("Using a 'fast' tokenizer. Setting num_proc to None for optimal performance.")
        else:
            effective_num_proc = num_proc if num_proc is not None else (min(4, mp.cpu_count() - 1) if mp.cpu_count() > 1 else 1)
            print(f"Processing with {effective_num_proc} processes.")

        if self.dataset is None:
            self.create_dataset_from_df()

        def tokenize_function(examples):
            input_ids_batch = []
            attention_mask_batch = []
            token_type_ids_batch = []
            offset_mapping_batch = []
            labels_batch = []
            hit_token_ranges = []
            original_text_full_batch = []

            for i in range(len(examples['Hit'])):
                context_before_text = str(examples['ContextBefore'][i]) if pd.notnull(examples['ContextBefore'][i]) else ""
                hit_text = str(examples['Hit'][i]) if pd.notnull(examples['Hit'][i]) else ""
                context_after_text = str(examples['ContextAfter'][i]) if pd.notnull(examples['ContextAfter'][i]) else ""

                full_text = context_before_text + hit_text + context_after_text
                original_text_full_batch.append(full_text)

                # Tokenize the full string with return_offsets_mapping and return_word_ids
                # `return_word_ids=True` is useful for mapping to original words, though `offset_mapping`
                # combined with `hit_char_start/end` is sufficient for this case.
                full_encoding = tokenizer(
                    full_text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors=None,
                    return_offsets_mapping=True,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    return_word_ids=True # Useful for debugging or more complex alignment
                )

                input_ids_batch.append(full_encoding['input_ids'])
                attention_mask_batch.append(full_encoding['attention_mask'])
                token_type_ids_batch.append(full_encoding['token_type_ids'])
                offset_mapping_batch.append(full_encoding['offset_mapping'])

                # Determine the character offsets of 'Hit' within the concatenated full_text
                cb_len = len(context_before_text)
                h_len = len(hit_text)
                hit_char_start = cb_len
                hit_char_end = cb_len + h_len

                # Find the token indices corresponding to the 'Hit' section
                hit_start_token_idx = -1
                hit_end_token_idx = -1

                for token_idx, (char_start, char_end) in enumerate(full_encoding['offset_mapping']):
                    if char_start is None: # Special tokens like [CLS], [SEP], padding
                        continue
                    if max(char_start, hit_char_start) < min(char_end, hit_char_end):
                        # This token overlaps with the 'Hit' original text span
                        if hit_start_token_idx == -1:
                            hit_start_token_idx = token_idx
                        hit_end_token_idx = token_idx + 1 # Exclusive end

                # If no hit tokens found (e.g., empty hit string), set range to 0,0
                if hit_start_token_idx == -1:
                    hit_start_token_idx = 0
                    hit_end_token_idx = 0
                hit_token_ranges.append((hit_start_token_idx, hit_end_token_idx))


                # Align labels for training data
                if self.training: # Assume 'bert_biolabels' is now present and BERT-aligned
                    bert_biolabels_hit = examples['bert_biolabels'][i]
                    # We need to map `bert_biolabels_hit` (which are for the 'Hit' segment only)
                    # to the full sequence's tokens.
                    full_labels = [-100] * len(full_encoding['input_ids'])
                    current_hit_label_idx = 0

                    for token_idx in range(len(full_encoding['input_ids'])):
                        if hit_start_token_idx <= token_idx < hit_end_token_idx:
                            # This token corresponds to the 'Hit' section
                            if current_hit_label_idx < len(bert_biolabels_hit):
                                full_labels[token_idx] = self.labeltoint[bert_biolabels_hit[current_hit_label_idx]]
                                current_hit_label_idx += 1
                            else:
                                # This can happen if truncation/padding logic for hit_token_ranges
                                # leads to more BERT tokens than original bert_biolabels_hit.
                                # This should be rare if max_length is sufficient and bert_biolabels_hit are correct.
                                full_labels[token_idx] = -100 # Default to ignore
                        else:
                            full_labels[token_idx] = -100 # Tokens outside 'Hit' are ignored for loss

                    labels_batch.append(full_labels)
                else: # For inference, labels are not needed
                    labels_batch.append([-100] * len(full_encoding['input_ids']))

            tokenized = {
                'input_ids': input_ids_batch,
                'attention_mask': attention_mask_batch,
                'token_type_ids': token_type_ids_batch,
                'offset_mapping': offset_mapping_batch,
                'hit_token_ranges': hit_token_ranges,
                'original_text_full': original_text_full_batch
            }

            if self.training:
                tokenized['labels'] = labels_batch

            return tokenized

        columns_to_remove = [col for col in self.dataset.column_names
                            if col not in (['instance', 'bert_biolabels'] if self.training else [])
                            and not col.startswith('bert_tok_Hit_')] # Keep the original context/hit/after columns for post-processing

        self.tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            num_proc=effective_num_proc,
            remove_columns=columns_to_remove, # This needs careful adjustment to keep relevant info
            desc="Tokenizing full sequence and aligning labels",
            cache_file_name=f"{cache_dir}/tokenized.arrow" if cache_dir else None if cache_dir else None
        )

        # Restore original text columns if removed for post-processing
        # The `remove_columns` can be tricky. It's often better to explicitly specify what to *keep*.
        # For post-processing, you need ContextBefore, Hit, ContextAfter.
        # Let's ensure these are kept, or accessed from self.dataset.
        # The `original_text_full` is derived, but the individual fields are still useful.

        return self.tokenized_dataset

    # (rest of your APCData class methods like tokenize_and_split_native, prepare_for_inference,
    # get_tokenized_dataset, save/load datasets, get_train/val/test_dataset are unchanged)


    def post_process_predictions(self, predictions, tokenizer):
        """
        Reconstructs text and extracts APCs from BERT model predictions.

        Args:
            predictions (np.array or torch.Tensor): The raw logits predictions from the BERT model.
                                                 Shape: (num_examples, sequence_length, num_labels).
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for encoding the input.

        Returns:
            list[dict]: A list of dictionaries, each representing an APC instance.
                        Format: {'ContextBefore': str, 'Hit': str, 'ContextAfter': str, 'instance': 1, 'APC': str}
        """
        if not hasattr(self, 'tokenized_dataset'):
            raise ValueError("Tokenized dataset not found. Please run prepare_for_inference() first.")
        if self.dataset is None:
             raise ValueError("Original dataset not found. Cannot retrieve ContextBefore/Hit/ContextAfter.")

        results = []

        # Convert predictions to labels (integers, then to string labels)
        predicted_label_ids = torch.argmax(torch.tensor(predictions), axis=2).tolist()

        # Iterate through the original dataset AND the tokenized dataset in parallel
        # This is important to get the original ContextBefore/Hit/ContextAfter strings
        for i, (original_example, tokenized_example) in enumerate(zip(self.dataset, self.tokenized_dataset)):
            input_ids = tokenized_example['input_ids']
            offset_mappings = tokenized_example['offset_mapping']
            hit_token_start_idx, hit_token_end_idx = tokenized_example['hit_token_ranges']
            original_full_text = tokenized_example['original_text_full']

            # Get the predicted BIO labels for this example
            actual_seq_len = sum(tokenized_example['attention_mask'])
            predicted_labels = [self.inttolabel[l_id] for l_id in predicted_label_ids[i][:actual_seq_len]]

            current_apc_start_char = -1
            current_apc_end_char = -1

            # Iterate through the tokens that correspond to the 'Hit' section based on hit_token_ranges
            # This ensures we only extract APCs from the 'Hit' part of the sentence
            for token_idx in range(hit_token_start_idx, hit_token_end_idx):
                if token_idx >= len(predicted_labels): # Safety check for truncated sequences
                    break

                label = predicted_labels[token_idx]
                char_start, char_end = offset_mappings[token_idx]

                # Skip special tokens or padding tokens that might have (0,0) or (None,None) offsets
                if char_start is None or char_end is None or (char_start == 0 and char_end == 0 and token_idx != 0):
                    continue

                # Ensure char_start and char_end are within the original full text bounds
                char_start = max(0, char_start)
                char_end = min(len(original_full_text), char_end)

                if label == 'B-APC':
                    if current_apc_start_char != -1: # Previous APC was open, close it and add to results
                        apc_string = original_full_text[current_apc_start_char:current_apc_end_char].strip()
                        if apc_string: # Only add if non-empty
                            results.append({
                                'ContextBefore': original_example['ContextBefore'],
                                'Hit': original_example['Hit'],
                                'ContextAfter': original_example['ContextAfter'],
                                'instance': 1,
                                'APC': apc_string
                            })
                    current_apc_start_char = char_start
                    current_apc_end_char = char_end
                elif label == 'I-APC':
                    if current_apc_start_char != -1: # Continue current APC
                        current_apc_end_char = char_end
                    # else: # I-APC without preceding B-APC, implies annotation error or model mistake, ignore for extraction
                elif label == 'O':
                    if current_apc_start_char != -1: # End of an APC
                        apc_string = original_full_text[current_apc_start_char:current_apc_end_char].strip()
                        if apc_string:
                            results.append({
                                'ContextBefore': original_example['ContextBefore'],
                                'Hit': original_example['Hit'],
                                'ContextAfter': original_example['ContextAfter'],
                                'instance': 1,
                                'APC': apc_string
                            })
                        current_apc_start_char = -1 # Reset for next APC
                        current_apc_end_char = -1

            # After iterating all tokens in the 'Hit' section, check if an APC is still open
            if current_apc_start_char != -1:
                apc_string = original_full_text[current_apc_start_char:current_apc_end_char].strip()
                if apc_string:
                    results.append({
                        'ContextBefore': original_example['ContextBefore'],
                        'Hit': original_example['Hit'],
                        'ContextAfter': original_example['ContextAfter'],
                        'instance': 1,
                        'APC': apc_string
                    })

        return results

    # The _align_labels_with_tokens method is no longer needed since generate_biolabels_dataset
    # now produces BERT-aligned labels directly. You can remove it.
    # def _align_labels_with_tokens(...)


# Example Usage (assuming you have a trained model and tokenizer)
if __name__ == '__main__':
    # --- Setup (Dummy for demonstration) ---
    # Create a dummy CSV file for demonstration
    dummy_csv_content = """ID,ContextBefore,Hit,ContextAfter,instance,APC
1,,Wir Linguisten lieben die Sprache.,Dennoch ist sie komplex.,1,Wir Linguisten
2,Ein Beispiel:,Ich mag Äpfel.,Sie sind gesund.,0,
3,In Berlin trafen sich,unsere Studenten und wir alle.,um ein Projekt zu besprechen.,1,wir alle
4,Der große,graue Elefant läuft langsam.,durch den Dschungel.,0,
5,Früher,die Deutschen lebten hier.,Heute sind es andere.,1,die Deutschen
6,Sie sagten:,ihr wart doch dabei.,Wir wissen es genau.,1,ihr wart
7,,Du und ich sind Freunde.,immer noch.,1,Du und ich
8,,Ich habe ihn,ihr Experten,gefragt.,.,1,ihr Experten
"""
    with open("dummy_data.csv", "w", encoding="utf-8") as f:
        f.write(dummy_csv_content)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

    # Dummy model for inference
    class DummyModel:
        def __init__(self, num_labels, seq_len):
            self.num_labels = num_labels
            self.seq_len = seq_len

        def __call__(self, input_ids, attention_mask=None, token_type_ids=None):
            # Simulate logits: random predictions
            # For demonstration, let's create a *somewhat* plausible output
            # where some APCs are "detected"
            logits = torch.full((input_ids.shape[0], input_ids.shape[1], self.num_labels), -10.0) # Default to O (large negative)
            
            # Simple simulation: for some examples, set a B-APC/I-APC sequence
            # This is highly simplified and won't match true BERT behavior
            
            # Example 0: "Wir Linguisten lieben die Sprache." -> "Wir Linguisten"
            # Assuming 'Wir' 'Linguisten' map to 2-3 tokens roughly around index 10-15 (after CLS, CB, SEP)
            if input_ids.shape[0] > 0:
                # Simulate "Wir Linguisten"
                # Find indices for a few tokens after CLS, SEP
                # This needs knowledge of token IDs. For simplicity, just pick a range.
                if input_ids[0].shape[0] > 15:
                    logits[0, 10, 0] = 10.0 # B-APC
                    logits[0, 11, 1] = 10.0 # I-APC
                    logits[0, 12, 1] = 10.0 # I-APC (if 'Linguisten' is split)

                # Example 2: "unsere Studenten und wir alle." -> "wir alle"
                if input_ids.shape[0] > 2 and input_ids[2].shape[0] > 20:
                    logits[2, 18, 0] = 10.0 # B-APC ('wir')
                    logits[2, 19, 1] = 10.0 # I-APC ('alle')

                # Example 4: "die Deutschen lebten hier." -> "die Deutschen"
                if input_ids.shape[0] > 4 and input_ids[4].shape[0] > 15:
                    logits[4, 10, 0] = 10.0 # B-APC ('die')
                    logits[4, 11, 1] = 10.0 # I-APC ('Deutschen')
                    logits[4, 12, 1] = 10.0 # I-APC (if split)

                # Example 5: "ihr wart doch dabei." -> "ihr wart"
                if input_ids.shape[0] > 5 and input_ids[5].shape[0] > 15:
                    logits[5, 10, 0] = 10.0 # B-APC ('ihr')
                    logits[5, 11, 1] = 10.0 # I-APC ('wart')

                # Example 6: "Du und ich sind Freunde." -> "Du und ich"
                if input_ids.shape[0] > 6 and input_ids[6].shape[0] > 15:
                    logits[6, 10, 0] = 10.0 # B-APC ('Du')
                    logits[6, 11, 1] = 10.0 # I-APC ('und')
                    logits[6, 12, 1] = 10.0 # I-APC ('ich')

                # Example 7: "ihr Experten"
                if input_ids.shape[0] > 7 and input_ids[7].shape[0] > 15:
                    logits[7, 10, 0] = 10.0 # B-APC ('ihr')
                    logits[7, 11, 1] = 10.0 # I-APC ('Experten')


            return type('Outputs', (object,), {'logits': logits})()

    dummy_model = DummyModel(num_labels=3, seq_len=64)


    print("\n--- Testing with CSV input (Training pipeline simulation) ---")
    # Pass tokenizer during initialization for generate_biolabels_dataset
    apc_data_csv = APCData(csvfile="dummy_data.csv", language="german", training=True, tokenizer=tokenizer)
    
    # This step now generates BERT-aligned BIO labels
    apc_data_csv.generate_biolabels_dataset()
    print("\nDataset after BERT-aligned BIO label generation:")
    print(apc_data_csv.get_dataset().column_names) # Check for 'bert_biolabels'
    # Check some labels:
    # print(apc_data_csv.get_dataset()[0]['Hit'])
    # print(apc_data_csv.get_dataset()[0]['bert_biolabels'])
    # print(tokenizer.convert_ids_to_tokens(apc_data_csv.get_dataset()[0]['bert_tok_Hit_input_ids']))

    # Demonstrate merging duplicates with new labels
    apc_data_csv.merge_apc_annotations() # This will now use 'bert_biolabels'
    print("\nDataset after merging duplicates:")
    print(apc_data_csv.get_dataset().column_names)
    print(apc_data_csv.get_df()[['Hit', 'APC', 'instance', 'bert_biolabels']])


    # Prepare for inference (tokenize full sequence)
    inference_dataset_csv = apc_data_csv.prepare_for_inference(tokenizer=tokenizer, max_length=64, num_proc=1)
    print("\nTokenized Dataset (CSV, ready for model):")
    print(inference_dataset_csv.column_names)
    # print(inference_dataset_csv[0]) # Inspect an example

    # Simulate predictions
    input_ids_batch = torch.tensor(inference_dataset_csv['input_ids'])
    attention_mask_batch = torch.tensor(inference_dataset_csv['attention_mask'])
    token_type_ids_batch = torch.tensor(inference_dataset_csv['token_type_ids'])

    with torch.no_grad():
        dummy_outputs = dummy_model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, token_type_ids=token_type_ids_batch)
        dummy_predictions = dummy_outputs.logits.cpu().numpy()

    # Post-process predictions
    results_list_csv = apc_data_csv.post_process_predictions(dummy_predictions, tokenizer)
    print("\nResults List (CSV Input):")
    for res in results_list_csv:
        print(res)


    print("\n--- Testing with raw string input (Inference pipeline) ---")
    raw_text = "Die jungen Leute feierten ausgelassen. Wir Linguisten sind gespannt. Meine Freunde und ich gehen ins Kino. Auch ihr Musiker seid willkommen."
    apc_data_str = APCData(strinput=raw_text, language="german", training=False, tokenizer=tokenizer)
    print(apc_data_str.get_dataset())

    # Prepare for inference
    inference_dataset_str = apc_data_str.prepare_for_inference(tokenizer=tokenizer, max_length=64, num_proc=1)
    print("\nTokenized Dataset (String, ready for model):")
    print(inference_dataset_str.column_names)
    # print(inference_dataset_str[0])

    # Simulate predictions
    input_ids_batch_str = torch.tensor(inference_dataset_str['input_ids'])
    attention_mask_batch_str = torch.tensor(inference_dataset_str['attention_mask'])
    token_type_ids_batch_str = torch.tensor(inference_dataset_str['token_type_ids'])

    with torch.no_grad():
        dummy_outputs_str = dummy_model(input_ids=input_ids_batch_str, attention_mask=attention_mask_batch_str, token_type_ids=token_type_ids_batch_str)
        dummy_predictions_str = dummy_outputs_str.logits.cpu().numpy()

    # Post-process predictions
    results_list_str = apc_data_str.post_process_predictions(dummy_predictions_str, tokenizer)
    print("\nResults List (Raw String Input):")
    for res in results_list_str:
        print(res)

    # Clean up dummy file
    import os
    os.remove("dummy_data.csv")
```