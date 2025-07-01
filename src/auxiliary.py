
import pandas as pd
import numpy as np

import multiprocessing as mp
from functools import partial
#from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

import torch

from datasets import load_dataset, Dataset, DatasetDict
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Time taken: {elapsed:.6f} seconds')
        return result
    return wrapper



# class for data input and pre-processing
class APCData:
    def __init__(self, training=False,language="german", csvfile=None, strinput=None, tokenizer=None):
        # setting core properties of data
        self.language=language              # marks (main) language of dataset
        self.training = training            # markes whether data set is annotated as training data
        self.tokenizer = tokenizer
        self.dataset = None  # Primary data storage
        self.df = None       # Keep for backward compatibility or if specific Pandas operations are truly needed
        
        if csvfile:
            print(f'Loading csv file {csvfile} into dataset')
            self.dataset = self.load_csv(csvfile)
            self.df = self.dataset.to_pandas() # Optionally keep a df for some operations, but minimize
        elif strinput:
            print(f'Loading raw string into dataset')
            self.dataset = self.load_raw_text(strinput)
            self.df = self.dataset.to_pandas() # Optionally keep a df
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
  
        self.textcols = ['ContextBefore', 'Hit', 'ContextAfter','APC']
        

    # INPUT METHODS
    
    # loading a csv (assumed to be annotated)
    def load_csv(self, path):
        # Directly load into a Hugging Face Dataset
        # If your CSV is large, load_dataset is more efficient than pd.read_csv then from_pandas
        try:
            dataset = load_dataset('csv', data_files=path, split='train')
            # Ensure correct types if necessary, though load_dataset often infers well
            # You might need to cast 'instance' column: dataset = dataset.cast_column('instance', Value('int32'))
            return dataset
        except Exception as e:
            print(f"Error loading CSV directly into Dataset: {e}. Falling back to Pandas.")
            df = pd.read_csv(path, index_col='ID', dtype={'instance': int})
            return Dataset.from_pandas(df, preserve_index=False) # preserve_index=False generally preferred for HF Dataset

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
            # Convert list of dicts directly to Dataset
            return Dataset.from_list(tmplist)
    
    ########################
    # BASIC DATA OUTPUT

    def get_dataset(self, subset=None):
        if subset == 'instance':
            return self.dataset.filter(lambda x: x['instance'] == 1)
        else:
            return self.dataset
        
    
    def get_df(self,subset=None):
        """
        Return full dataframe by default
        
        arguments:
        **subset**: set to 'instance' to get only rows with instance==1 
        """
        if self.df is None:
            if self.dataset:
                print("Converting dataset to pandas DataFrame. Consider using get_dataset() directly for efficiency.")
                self.df = self.dataset.to_pandas()
            else:
                return None

        if subset == 'instance':
            return self.df[self.df.instance==1]
        else:
            return self.df
        

    def df_index(self,index):
        return self.df.loc[index]
    
    
    ##################################
    # Generating and combining data

    @timeit
    def generate_biolabels_dataset(self):
        """
        Tokenize "Hit" column with the BERT tokenizer and create BIO labels for it.
        The labels are directly aligned with BERT's subword tokens.
        Handles case-insensitivity for finding the APC string within Hit.
        """
        if self.dataset is None:
            raise ValueError("No dataset available. Load data first.")
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Please provide a tokenizer when initializing APCData or set it before calling this method.")

        def process_example(example):
            hit_text = str(example['Hit']) if pd.notnull(example['Hit']) else ""
            apc_string = str(example['APC']) if pd.notnull(example['APC']) else ""

            # Tokenize the 'Hit' sentence with BERT tokenizer, getting offsets
            tokenized_hit = self.tokenizer(
                hit_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False,
                padding=False
            )

            # Initialize labels with 'O' for all tokens in the 'Hit' sentence
            labels = ["O"] * len(tokenized_hit['input_ids'])

            # If it's an instance with an APC, assign B-APC/I-APC labels
            if example['instance'] == 1 and apc_string:
                # Use lowercase for finding the APC string to handle capitalization differences
                # The `find` method works on character indices. The `offset_mapping`
                # also refers to character indices in the *original* string.
                # So we find the lowercase match, then use those character indices
                # to refer back to the original cased `hit_text` for label assignment.
                hit_text_lower = hit_text.lower()
                apc_string_lower = apc_string.lower()

                apc_start_char_in_hit = hit_text_lower.find(apc_string_lower)
                apc_end_char_in_hit = apc_start_char_in_hit + len(apc_string_lower)

                if apc_start_char_in_hit != -1: # APC string (lowercase) found in Hit (lowercase)
                    is_inside_apc = False
                    for i, (char_start, char_end) in enumerate(tokenized_hit['offset_mapping']):
                        if char_start is None or char_end is None: # Should not happen with add_special_tokens=False
                            continue

                        # Check for overlap: [token_char_start, token_char_end) vs [apc_start_char_in_hit, apc_end_char_in_hit)
                        # The `offset_mapping` points to chars in the original `hit_text`.
                        # Since `apc_start_char_in_hit` also points to the same character positions (just in the lowercased version),
                        # this check correctly aligns the tokens with the identified APC span.
                        if max(char_start, apc_start_char_in_hit) < min(char_end, apc_end_char_in_hit):
                            if not is_inside_apc:
                                labels[i] = "B-APC"
                                is_inside_apc = True
                            else:
                                labels[i] = "I-APC"
                        else:
                            if is_inside_apc:
                                is_inside_apc = False

            example['bert_tok_Hit_input_ids'] = tokenized_hit['input_ids']
            example['bert_tok_Hit_attention_mask'] = tokenized_hit['attention_mask']
            example['bert_tok_Hit_offsets'] = tokenized_hit['offset_mapping']
            example['bert_biolabels'] = labels

            return example

        self.dataset = self.dataset.map(
            process_example,
            num_proc=mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1,
            desc="Generating BERT-aligned BIO labels for 'Hit'"
        )
        print("Generated BERT-aligned BIO labels.")
                


    # Original DataFrame-based methods (keep if conversion overhead is acceptable, or refactor to Dataset operations)
    def find_duplicate_hits(self):
        """ 
        Detect duplicate sentences that might need merging
        
        Returns: list of tuples of indices for each group of duplicates
        """
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
    
    ###
    # Experimenting with direct use of datasets
    def create_dataset_from_df(self):
        """Convert processed DataFrame to HuggingFace Dataset"""
        if self.df is None:
            raise ValueError("No dataframe available. Load data first.")
            
        # Convert DataFrame to Dataset
        self.dataset = Dataset.from_pandas(self.df)
        return self.dataset


    @timeit
    def tokenize_dataset(self, num_proc=None, batch_size=1000,
                        max_length=512, # Use typical BERT max length
                        overlap=64,    # Overlap between chunks
                        cache_dir=None):
        """
        General tokenization method that works for both training and inference data.
        Handles long sequences by splitting them into overlapping chunks.
        Each chunk becomes a separate row in the output dataset.
        """

        if self.dataset is None:
            raise ValueError("No dataset available. Load data first.")
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Please provide a tokenizer when initializing APCData or set it before calling this method.")
        
        if self.tokenizer.is_fast:
            effective_num_proc = None
            print("Using a 'fast' tokenizer. Setting num_proc to None for optimal performance.")
        else:
            effective_num_proc = num_proc if num_proc is not None else (min(4, mp.cpu_count() - 1) if mp.cpu_count() > 1 else 1)
            print(f"Processing with {effective_num_proc} processes.")

        def tokenize_and_chunk_function(example, idx):
            """
            Process a single example and return its chunks.
            This function processes one example at a time to avoid length mismatch issues.
            """
            original_example_idx = idx
            original_instance = example['instance'] if 'instance' in example else 0

            context_before_text = str(example['ContextBefore']) if pd.notnull(example['ContextBefore']) else ""
            hit_text = str(example['Hit']) if pd.notnull(example['Hit']) else ""
            context_after_text = str(example['ContextAfter']) if pd.notnull(example['ContextAfter']) else ""

            full_text = context_before_text + hit_text + context_after_text

            # Skip empty texts
            if not full_text.strip():
                return {
                    'input_ids': [],
                    'attention_mask': [],
                    'token_type_ids': [],
                    'offset_mapping': [],
                    'labels': [],
                    'hit_token_ranges_in_chunk': [],
                    'hit_char_ranges_in_full_text': [],
                    'original_text_full': [],
                    'original_idx': [],
                    'original_instance': []
                }

            full_encoding = self.tokenizer(
                full_text,
                truncation=False,
                padding=False,
                return_tensors=None,
                return_offsets_mapping=True,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True
            )

            # Handle cases where tokenization yields no tokens
            if not full_encoding['input_ids']:
                return {
                    'input_ids': [],
                    'attention_mask': [],
                    'token_type_ids': [],
                    'offset_mapping': [],
                    'labels': [],
                    'hit_token_ranges_in_chunk': [],
                    'hit_char_ranges_in_full_text': [],
                    'original_text_full': [],
                    'original_idx': [],
                    'original_instance': []
                }

            # Calculate character ranges for hit text in full text
            cb_len = len(context_before_text)
            h_len = len(hit_text)
            hit_char_start_in_full = cb_len
            hit_char_end_in_full = cb_len + h_len

            # Prepare labels for the full sequence if training
            full_sequence_labels = None
            if self.training and 'bert_biolabels' in example:
                bert_biolabels_hit = example['bert_biolabels']
                full_sequence_labels = [-100] * len(full_encoding['input_ids'])
                hit_labels_idx = 0
                
                for t_idx, (c_start, c_end) in enumerate(full_encoding['offset_mapping']):
                    if c_start is None or c_end is None:
                        continue
                    # Check if token overlaps with hit text
                    if max(c_start, hit_char_start_in_full) < min(c_end, hit_char_end_in_full):
                        if hit_labels_idx < len(bert_biolabels_hit):
                            full_sequence_labels[t_idx] = self.labeltoint[bert_biolabels_hit[hit_labels_idx]]
                            hit_labels_idx += 1

            # Lists to collect all chunks for this example
            chunks_input_ids = []
            chunks_attention_mask = []
            chunks_token_type_ids = []
            chunks_offset_mapping = []
            chunks_labels = []
            chunks_hit_token_ranges = []
            chunks_hit_char_ranges = []
            chunks_original_text_full = []
            chunks_original_idx = []
            chunks_original_instance = []

            # Create chunks from this example
            token_start = 0
            while token_start < len(full_encoding['input_ids']):
                token_end = min(token_start + max_length, len(full_encoding['input_ids']))
                
                # Extract chunk
                chunk_ids = full_encoding['input_ids'][token_start:token_end]
                chunk_offsets = full_encoding['offset_mapping'][token_start:token_end]
                
                # Pad to max_length
                padding_length = max_length - len(chunk_ids)
                padded_input_ids = chunk_ids + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = [1] * len(chunk_ids) + [0] * padding_length
                token_type_ids = [0] * max_length
                padded_offset_mapping = chunk_offsets + [(0, 0)] * padding_length
                
                # Handle labels for this chunk
                if self.training and full_sequence_labels is not None:
                    chunk_labels = full_sequence_labels[token_start:token_end]
                    chunk_labels = chunk_labels + [-100] * padding_length
                else:
                    chunk_labels = [-100] * max_length
                
                # Find hit token range in this chunk (for potential future use)
                hit_start_token_idx_in_chunk = 0
                hit_end_token_idx_in_chunk = 0
                
                for j, (c_start, c_end) in enumerate(chunk_offsets):
                    if c_start is None or c_end is None:
                        continue
                    if max(c_start, hit_char_start_in_full) < min(c_end, hit_char_end_in_full):
                        if hit_start_token_idx_in_chunk == 0:
                            hit_start_token_idx_in_chunk = j
                        hit_end_token_idx_in_chunk = j + 1
                
                # Append this chunk to the lists
                chunks_input_ids.append(padded_input_ids)
                chunks_attention_mask.append(attention_mask)
                chunks_token_type_ids.append(token_type_ids)
                chunks_offset_mapping.append(padded_offset_mapping)
                chunks_labels.append(chunk_labels)
                chunks_hit_token_ranges.append((hit_start_token_idx_in_chunk, hit_end_token_idx_in_chunk))
                chunks_hit_char_ranges.append((hit_char_start_in_full, hit_char_end_in_full))
                chunks_original_text_full.append(full_text)
                chunks_original_idx.append(original_example_idx)
                chunks_original_instance.append(original_instance)
                
                # Move to next chunk with overlap
                token_start += (max_length - overlap)

            # Return all chunks for this example
            return {
                'input_ids': chunks_input_ids,
                'attention_mask': chunks_attention_mask,
                'token_type_ids': chunks_token_type_ids,
                'offset_mapping': chunks_offset_mapping,
                'labels': chunks_labels,
                'hit_token_ranges_in_chunk': chunks_hit_token_ranges,
                'hit_char_ranges_in_full_text': chunks_hit_char_ranges,
                'original_text_full': chunks_original_text_full,
                'original_idx': chunks_original_idx,
                'original_instance': chunks_original_instance
            }

        columns_to_remove = [
            'ContextBefore',
            'Hit',
            'ContextAfter',
            'APC',
            'bert_tok_Hit_input_ids',
            'bert_tok_Hit_attention_mask',
            'bert_tok_Hit_offsets'
        ]

        # Process one example at a time (batched=False) to avoid length mismatch issues
        self.tokenized_dataset = self.dataset.map(
            tokenize_and_chunk_function,
            batched=False,  # Process one example at a time
            with_indices=True,
            num_proc=effective_num_proc,
            remove_columns=columns_to_remove,
            desc="Tokenizing and chunking sequences",
            cache_file_name=f"{cache_dir}/tokenized_chunked.arrow" if cache_dir else None
        )

        # This is your existing print
        print(f"Tokenization complete. Original examples: {len(self.dataset)}, Total chunks: {total_chunks}")
        return self.tokenized_dataset




    @timeit                 
    def tokenize_and_split_native(self, test_size=0.1, val_size=0.2, 
                                random_state=None, num_proc=None, batch_size=1000,
                                max_length=128, overlap_size=64, cache_dir=None):
        """
        Tokenize and split method specifically for training data.
        Now much simpler thanks to flat dataset structure.
        """
        if not self.training: 
            raise ValueError("This method is only for training data. Use tokenize_dataset() for inference data.") 
        
        print('Calling tokenization')
        # First tokenize the dataset (creates flat structure)
        self.tokenize_dataset( 
            num_proc=num_proc, batch_size=batch_size, max_length=max_length, overlap=overlap_size, cache_dir=cache_dir 
        ) 
        
        print('Finished tokenization')
        
        if 'labels' not in self.tokenized_dataset.column_names: 
            raise ValueError("Labels not found in tokenized dataset. Ensure APCData was initialized with training=True and generate_biolabels_dataset was called.") 

        if 'original_instance' not in self.tokenized_dataset.column_names: 
            raise ValueError("'original_instance' column not found in tokenized dataset.") 
        
        if 'original_idx' not in self.tokenized_dataset.column_names:
            raise ValueError("'original_idx' column not found in tokenized dataset.") 

        # Extract unique original examples for stratified splitting
        print("Extracting unique original examples for splitting...")
        
        # Get unique original examples - much simpler now!
        unique_examples_df = (
            self.tokenized_dataset
            .select_columns(['original_idx', 'original_instance'])
            .to_pandas()
            .drop_duplicates(subset=['original_idx'])
        )
        
        print(f"Found {len(unique_examples_df)} unique original examples")
        print(f"Instance distribution: {unique_examples_df['original_instance'].value_counts().to_dict()}")
        
        # Stratified splits on unique original examples
        train_val_df, test_df = train_test_split( 
            unique_examples_df, 
            test_size=test_size,  
            stratify=unique_examples_df["original_instance"],
            random_state=random_state 
        ) 
        
        train_df, val_df = train_test_split( 
            train_val_df, 
            test_size=val_size / (1 - test_size), 
            stratify=train_val_df["original_instance"],  
            random_state=random_state 
        )         
        
        # Get sets of original_idx values for each split (O(1) lookup)
        train_original_indices_set = set(train_df['original_idx'].tolist())
        val_original_indices_set = set(val_df['original_idx'].tolist())
        test_original_indices_set = set(test_df['original_idx'].tolist())
        
        print(f"Split original examples - Train: {len(train_original_indices_set)}, Val: {len(val_original_indices_set)}, Test: {len(test_original_indices_set)}")
        
        # Filter chunks based on their original_idx - much simpler now!
        print("Creating dataset splits...")
        train_dataset = self.tokenized_dataset.filter(
            lambda example: example['original_idx'] in train_original_indices_set, 
            num_proc=num_proc,
            desc="Creating train dataset"
        ) 
        val_dataset = self.tokenized_dataset.filter(
            lambda example: example['original_idx'] in val_original_indices_set, 
            num_proc=num_proc,
            desc="Creating validation dataset"
        ) 
        test_dataset = self.tokenized_dataset.filter(
            lambda example: example['original_idx'] in test_original_indices_set, 
            num_proc=num_proc,
            desc="Creating test dataset"
        ) 
        
        print(f"Final chunk counts - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Store as DatasetDict for easy access
        self.datasets = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        return train_dataset, val_dataset, test_dataset

    def prepare_for_inference(self, max_length=64, num_proc=None, batch_size=1000):
        """
        Convenience method specifically for preparing inference data
        """
        if self.training:
            print("Warning: This appears to be training data. Consider using tokenize_and_split_native() instead.")
        
        return self.tokenize_dataset(
            max_length=max_length,
            num_proc=num_proc,
            batch_size=batch_size
        )

    
    def get_tokenized_dataset(self):
        """
        Return the tokenized dataset (for inference or before splitting)
        """
        if hasattr(self, 'tokenized_dataset'):
            return self.tokenized_dataset
        else:
            raise ValueError("No tokenized dataset available. Run tokenize_dataset() first.")
        
    def save_datasets(self, path):
        """Save processed datasets to disk"""
        if self.datasets is None:
            raise ValueError("No datasets to save. Process data first.")
        
        self.datasets.save_to_disk(path)
        print(f"Datasets saved to {path}")

    def load_datasets(self, path):
        """Load processed datasets from disk"""
        self.datasets = DatasetDict.load_from_disk(path)
        print(f"Datasets loaded from {path}")
        return self.datasets

    def get_datasets(self):
        """Return the DatasetDict"""
        return self.datasets
    
    def get_train_dataset(self):
        """Return just the training dataset"""
        return self.datasets['train'] if self.datasets else None
    
    def get_val_dataset(self):
        """Return just the validation dataset"""
        return self.datasets['validation'] if self.datasets else None
    
    def get_test_dataset(self):
        """Return just the test dataset"""
        return self.datasets['test'] if self.datasets else None
    
    
    def post_process_predictions(self, predictions, include_personal_pronouns=False):
        """
        Reconstructs text and extracts APCs from BERT model predictions, handling chunked data.
        Optionally, also captures personal pronouns that are not part of an APC.

        Args:
            predictions (np.array or torch.Tensor): The raw logits predictions from the BERT model.
                                                Shape: (num_examples_in_tokenized_dataset, sequence_length, num_labels).
            include_personal_pronouns (bool): If True, also find and report non-APC personal pronouns.

        Returns:
            list[dict]: A list of dictionaries, each representing an APC instance
                        or a non-APC personal pronoun.
                        Format for APC: {'ContextBefore': str, 'Hit': str, 'ContextAfter': str, 'instance': 1, 'APC': str}
                        Format for pronoun: {'ContextBefore': str, 'Hit': str, 'ContextAfter': str, 'instance': 0, 'APC': ''}
        """
        if not hasattr(self, 'tokenized_dataset') or self.tokenized_dataset is None:
            raise ValueError("Tokenized dataset not found. Please run tokenize_dataset() first.")
        if self.dataset is None:
            raise ValueError("Original dataset not found. Cannot retrieve ContextBefore/Hit/ContextAfter.")

        # Convert predictions to labels (integers)
        predicted_label_ids = np.argmax(predictions, axis=2).tolist()

        # Group predictions and offsets by original example ID
        grouped_predictions = {} # Key: original_idx, Value: list of (chunk_labels, chunk_offsets, hit_char_range_in_full, original_full_text)
        
        # Create a mapping of original examples for easy lookup
        original_examples_map = {i: example for i, example in enumerate(self.dataset)}

        for i, tokenized_example in enumerate(self.tokenized_dataset):
            original_idx = tokenized_example['original_idx']
            
            # Get only the active (non-padded) predictions and offsets for this chunk
            actual_seq_len = sum(tokenized_example['attention_mask'])
            chunk_predicted_labels = [self.inttolabel[l_id] for l_id in predicted_label_ids[i][:actual_seq_len]]
            chunk_offsets = tokenized_example['offset_mapping'][:actual_seq_len]

            # Store hit_char_range_in_full_text as it's consistent across chunks from the same original example
            hit_char_range = tokenized_example['hit_char_ranges_in_full_text']
            original_full_text = tokenized_example['original_text_full']

            if original_idx not in grouped_predictions:
                grouped_predictions[original_idx] = []
            
            grouped_predictions[original_idx].append({
                'labels': chunk_predicted_labels,
                'offsets': chunk_offsets,
                'hit_char_range': hit_char_range,
                'original_full_text': original_full_text
            })

        final_results = []

        # Process each original example
        for original_idx, chunks_data in grouped_predictions.items():
            original_example = original_examples_map[original_idx]
            context_before_original = str(original_example['ContextBefore']) if pd.notnull(original_example['ContextBefore']) else ""
            hit_original = str(original_example['Hit']) if pd.notnull(original_example['Hit']) else ""
            context_after_original = str(original_example['ContextAfter']) if pd.notnull(original_example['ContextAfter']) else ""

            # Consolidate labels and offsets for the full original sequence
            # Use the first chunk's full_text and hit_char_range as they are consistent
            full_text_for_original = chunks_data[0]['original_full_text']
            hit_char_start_in_full, hit_char_end_in_full = chunks_data[0]['hit_char_range']

            # Create a dictionary to store the "best" label for each token position
            # Use character start position as key to handle overlapping chunks
            consolidated_labels_map = {} # Key: token_start_char_offset, Value: label info

            # Iterate through each chunk for this original example
            for chunk_data in chunks_data:
                chunk_labels = chunk_data['labels']
                chunk_offsets = chunk_data['offsets']

                for token_idx, (char_start, char_end) in enumerate(chunk_offsets):
                    if token_idx >= len(chunk_labels):  # Safety check
                        break
                        
                    label = chunk_labels[token_idx]
                    
                    if char_start is None or char_end is None: # Special tokens/padding
                        continue

                    # If this token is an APC label (B-APC or I-APC), it takes precedence
                    if label != 'O':
                        consolidated_labels_map[char_start] = {
                            'label': label,
                            'char_end': char_end,
                            'token_text': full_text_for_original[char_start:char_end]
                        }
                    elif char_start not in consolidated_labels_map:
                        # Only add 'O' if no other label for this token has been seen yet
                        consolidated_labels_map[char_start] = {
                            'label': label,
                            'char_end': char_end,
                            'token_text': full_text_for_original[char_start:char_end]
                        }
            
            # Sort tokens by their start character offset to reconstruct original order
            sorted_tokens = sorted(consolidated_labels_map.items())

            apc_spans = [] # Store identified APCs as (start_char, end_char) within full_text_for_original
            current_apc_start_char = -1
            current_apc_end_char = -1

            # --- 1. Extract APCs from consolidated labels ---
            for char_start, token_info in sorted_tokens:
                label = token_info['label']
                char_end = token_info['char_end']

                # Only process tokens that fall within the original HIT range
                if not (hit_char_start_in_full <= char_start < hit_char_end_in_full):
                    # If an APC was open and we moved outside HIT, close it.
                    if current_apc_start_char != -1:
                        apc_string = full_text_for_original[current_apc_start_char:current_apc_end_char].strip()
                        if apc_string:
                            final_results.append({
                                'ContextBefore': context_before_original,
                                'Hit': hit_original,
                                'ContextAfter': context_after_original,
                                'instance': 1,
                                'APC': apc_string
                            })
                            apc_spans.append((current_apc_start_char, current_apc_end_char))
                        current_apc_start_char = -1
                        current_apc_end_char = -1
                    continue # Skip tokens outside HIT for APC extraction

                if label == 'B-APC':
                    if current_apc_start_char != -1: # Previous APC was open, close it
                        apc_string = full_text_for_original[current_apc_start_char:current_apc_end_char].strip()
                        if apc_string:
                            final_results.append({
                                'ContextBefore': context_before_original,
                                'Hit': hit_original,
                                'ContextAfter': context_after_original,
                                'instance': 1,
                                'APC': apc_string
                            })
                            apc_spans.append((current_apc_start_char, current_apc_end_char))
                    current_apc_start_char = char_start
                    current_apc_end_char = char_end
                elif label == 'I-APC':
                    if current_apc_start_char != -1: # Continue current APC
                        current_apc_end_char = char_end
                    # else: # I-APC without preceding B-APC (isolated I), ignore for extraction
                elif label == 'O':
                    if current_apc_start_char != -1: # End of an APC
                        apc_string = full_text_for_original[current_apc_start_char:current_apc_end_char].strip()
                        if apc_string:
                            final_results.append({
                                'ContextBefore': context_before_original,
                                'Hit': hit_original,
                                'ContextAfter': context_after_original,
                                'instance': 1,
                                'APC': apc_string
                            })
                            apc_spans.append((current_apc_start_char, current_apc_end_char))
                        current_apc_start_char = -1 # Reset for next APC
                        current_apc_end_char = -1

            # After iterating all tokens, check if an APC is still open
            if current_apc_start_char != -1:
                # Ensure the APC ends within the HIT text
                if current_apc_end_char > hit_char_end_in_full:
                    current_apc_end_char = hit_char_end_in_full # Clip if it runs over
                
                apc_string = full_text_for_original[current_apc_start_char:current_apc_end_char].strip()
                if apc_string:
                    final_results.append({
                        'ContextBefore': context_before_original,
                        'Hit': hit_original,
                        'ContextAfter': context_after_original,
                        'instance': 1,
                        'APC': apc_string
                    })
                    apc_spans.append((current_apc_start_char, current_apc_end_char))

            # --- 2. Capture Non-APC Personal Pronouns (if requested) ---
            if include_personal_pronouns:
                # Define German personal pronouns (this was referenced but not defined in the original code)
                german_personal_pronouns = {
                    'ich', 'du', 'wir', 'ihr', 'Sie',
                    'mich', 'dich', 'uns', 'euch',
                    'mir', 'dir', 'Ihnen'
                }
                
                processed_pronoun_spans = set() # To store (start_char, end_char) of pronouns already added

                for char_start, token_info in sorted_tokens:
                    token_text = token_info['token_text']
                    char_end = token_info['char_end']

                    # Only consider tokens that fall within the original HIT range
                    if not (hit_char_start_in_full <= char_start < hit_char_end_in_full):
                        continue

                    # Check if it's a personal pronoun (case-insensitive check)
                    # And ensure it's not empty and not just punctuation/whitespace
                    if token_text and token_text.lower() in german_personal_pronouns:
                        # Check if this pronoun overlaps with any extracted APC
                        is_part_of_apc = False
                        for apc_s, apc_e in apc_spans:
                            # Check for overlap: [token_char_start, token_char_end) vs [apc_s, apc_e)
                            if max(char_start, apc_s) < min(char_end, apc_e):
                                is_part_of_apc = True
                                break

                        # Check if this pronoun has already been processed and added
                        if (char_start, char_end) in processed_pronoun_spans:
                            continue

                        if not is_part_of_apc:
                            final_results.append({
                                'ContextBefore': context_before_original,
                                'Hit': hit_original,
                                'ContextAfter': context_after_original,
                                'instance': 0, # Not an APC
                                'APC': token_text # The pronoun itself
                            })
                            processed_pronoun_spans.add((char_start, char_end)) # Mark as processed

        return final_results

    
    
    ######################################
    # DEPRECATED STUFF slated for removal
    
    

    # @timeit
    # def generate_biolabels_dataset_old(self):
    #     """
    #     Tokenize text columns and create bio labels for "Hit" column directly on the dataset.
    #     """
    #     if self.dataset is None:
    #         raise ValueError("No dataset available. Load data first.")

    #     # define the function to apply to each example in the dataset
    #     def process_example(example):
    #         # tokenize text columns
    #         example['tok_ContextBefore'] = word_tokenize(str(example['ContextBefore']), language=self.language) if pd.notnull(example['ContextBefore']) else []
    #         example['tok_Hit'] = word_tokenize(str(example['Hit']), language=self.language) if pd.notnull(example['Hit']) else []
    #         example['tok_ContextAfter'] = word_tokenize(str(example['ContextAfter']), language=self.language) if pd.notnull(example['ContextAfter']) else []
    #         example['tok_APC'] = word_tokenize(str(example['APC']), language=self.language) if pd.notnull(example['APC']) else []

    #         # apply BIO labeling for rows with APCs
    #         if example['instance'] == 1:
    #             example['biolabels'] = self.generate_biolabels_single(example['tok_Hit'], example['tok_APC'])
    #         else:
    #             example['biolabels'] = ["O"] * len(example['tok_Hit'])

    #         # check for length mismatch
    #         if len(example['tok_Hit']) != len(example['biolabels']):
    #             print(f"Length mismatch: tok_Hit={len(example['tok_Hit'])}, biolabels={len(example['biolabels'])}")

    #         return example

    #     # Use .map() for efficient processing
    #     self.dataset = self.dataset.map(process_example, num_proc=mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1,
    #                                     desc="Generating BIO labels and tokenizing text columns")

        # Remove original text columns if no longer needed to save memory
        # self.dataset = self.dataset.remove_columns(self.textcols)


    # def generate_biolabels_single(self, sentence_tokens, apc_tokens):
    #     # This method remains largely the same as it operates on single token lists
    #     if not sentence_tokens or not apc_tokens:
    #         return ["O"] * len(sentence_tokens) if sentence_tokens else []

    #     labels = ["O"] * len(sentence_tokens)
    #     sentence_tokens_lower = [t.lower() for t in sentence_tokens]
    #     apc_tokens_lower = [t.lower() for t in apc_tokens]

    #     for i in range(len(sentence_tokens_lower) - len(apc_tokens_lower) + 1):
    #         if sentence_tokens_lower[i:i+len(apc_tokens_lower)] == apc_tokens_lower:
    #             labels[i] = "B-APC"
    #             for j in range(1, len(apc_tokens_lower)):
    #                 if i + j < len(labels):
    #                     labels[i + j] = "I-APC"
    #     return labels

    # Merging duplicates is still best done on a DataFrame if you need to modify rows based on indices
    # and then convert back to a Dataset. Or, try to achieve it with Dataset.group_by and .map if possible.
    # For now, keeping find_duplicate_hits and merge_apc_annotations DataFrame-based might be simpler
    # if the logic is complex, but be aware of the conversion overhead for very large datasets.
    
        # @timeit
    # def merge_apc_annotations(self):
    #     if self.df is None:
    #         print("No data to merge.")
    #         return
    #     else:
    #         self.df = self.dataset.to_pandas()


    #     id_tuples = self.find_duplicate_hits()

    #     print("Merging BIO-labels")
    #     processed=0
    #     for t in id_tuples:
    #         consolidated = self.df.loc[t[0]]['biolabels']
    #         for i in t[1:]:
    #             processed+=1
    #             tmp_bio = self.df.loc[i]['biolabels']
    #             for count in range(len(consolidated)):
    #                 if tmp_bio[count] != 'O':
    #                     consolidated[count] = tmp_bio[count]

    #         self.df.at[t[0],'biolabels'] = consolidated

    #     print(f"Processed {processed} duplicate rows")

    #     print("Removing now-redundant rows")
    #     self.df = self.df.drop_duplicates(subset='Hit',keep='first')

    #     # Convert back to Dataset after modification
    #     self.dataset = Dataset.from_pandas(self.df, preserve_index=False)
    #     print("Dataset updated after merging and removing duplicates.")
    
    
    
        # def prepare_for_inference(self,, max_length=64, num_proc=None, cache_dir=None):
    #     # Set training to False temporarily for inference prep if it's currently True
    #     original_training_status = self.training
    #     self.training = False
    #     tokenized_data = self.tokenize_dataset(tokenizer, max_length=max_length, num_proc=num_proc, cache_dir=cache_dir)
    #     self.training = original_training_status # Restore original status
    #     return tokenized_data
    # @timeit     # measure time for executing method
    # def generate_biolabels_df(self):
    #     """
    #     Tokenise text columns and create bio labels for "Hit" column
    #     """
    #     for col in self.textcols:
    #         self.df['tok_' + col] = self.df[col].apply(lambda x: word_tokenize(str(x), language=self.language) if pd.notnull(x) else [])

    #     # apply BIO labeling for rows with APCs
    #     mask = self.df["instance"] == 1
    #     self.df.loc[mask, "biolabels"] = self.df[mask].apply(
    #         lambda row: self.generate_biolabels_single(row["tok_Hit"], row["tok_APC"]), axis=1
    #     )

    #     # assign dummy labels to the rest
    #     self.df.loc[~mask, "biolabels"] = self.df.loc[~mask, "tok_Hit"].apply(
    #         lambda tokens: ["O"] * len(tokens) if isinstance(tokens, list) else []
    #     )       
        
    #     mask = self.df["instance"] == 1
    #     for idx in self.df[mask].index:
    #         tok_hit_len = len(self.df.loc[idx, "tok_Hit"])
    #         biolabel_len = len(self.df.loc[idx, "biolabels"])
    #         if tok_hit_len != biolabel_len:
    #             print(f"Length mismatch at index {idx}: tok_Hit={tok_hit_len}, biolabels={biolabel_len}")                         
    
    
    # def generate_biolabels_single(self, sentence_tokens, apc_tokens):
    #     """
    #     Given tokenized sentence and tokenized APC, return BIO labels for sentence.
    #     All instances of the same APC in the same sentence are marked, i.e. we can capture repetitions appropriately. 
    #     In line with what the base dataset annotation provides, however, this only works for one specific APC.
    #     """
    #     if not sentence_tokens or not apc_tokens:
    #         return ["O"] * len(sentence_tokens) if sentence_tokens else []
        
    #     labels = ["O"] * len(sentence_tokens)
    #     sentence_tokens_lower = [t.lower() for t in sentence_tokens]
    #     apc_tokens_lower = [t.lower() for t in apc_tokens]
        
    #     # Check if we have enough tokens left to match
    #     for i in range(len(sentence_tokens_lower) - len(apc_tokens_lower) + 1):
    #         if sentence_tokens_lower[i:i+len(apc_tokens_lower)] == apc_tokens_lower:
    #             labels[i] = "B-APC"
    #             for j in range(1, len(apc_tokens_lower)):
    #                 if i + j < len(labels):  # Safety check
    #                     labels[i + j] = "I-APC"
        
    #     return labels


    # def tokenize_and_align_labels(self,tokenizer):
    #     # Tokenize and align BIO labels, assigning -100 to all elements in context
    #     tokenized_inputs = []

    #     for _, row in self.df.iterrows():
    #         # 1. Concatenate tokens
    #         full_tokens = row["tok_ContextBefore"] + row["tok_Hit"] + row["tok_ContextAfter"]

    #         # 2. Create full BIO labels
    #         len_before = len(row["tok_ContextBefore"])
    #         len_hit = len(row["tok_Hit"])
    #         len_after = len(row["tok_ContextAfter"])

    #         hit_labels = row["biolabels"] if isinstance(row["biolabels"], list) else ["O"] * len_hit

    #         labels = (
    #             [-100] * len_before +
    #             [label for label in hit_labels] +
    #             [-100] * len_after
    #         )

    #         # 3. Tokenize with is_split_into_words=True
    #         tokenized = tokenizer(
    #             full_tokens,
    #             is_split_into_words=True,
    #             truncation=True,
    #             padding="max_length",
    #             max_length=64,  # Adjust as needed
    #             return_tensors="pt"
    #         )

    #         # 4. Align labels with wordpieces
    #         word_ids = tokenized.word_ids(batch_index=0)
    #         aligned_labels = []
    #         previous_word_idx = None
            
    #         for word_idx in word_ids:
    #             if word_idx is None:
    #                 aligned_labels.append(-100)
    #             else:
    #                 label = labels[word_idx]
    #                 if label == -100:
    #                     aligned_labels.append(-100)
    #                 else:
    #                     # Handle subword continuation
    #                     if word_idx == previous_word_idx:
    #                         # This is a continuation of the previous word
    #                         if isinstance(label, str) and label.startswith('B-'):
    #                             # Convert B- to I- for subword continuations
    #                             aligned_labels.append(self.labeltoint[label.replace('B-', 'I-')])
    #                         elif isinstance(label, str):
    #                             # Keep the same label for I- and O
    #                             aligned_labels.append(self.labeltoint[label])
    #                         else:
    #                             # If label is already an integer, use it directly
    #                             aligned_labels.append(label)
    #                     else:
    #                         # First subword of this word, keep original label
    #                         if isinstance(label, str):
    #                             aligned_labels.append(self.labeltoint[label])
    #                         else:
    #                             aligned_labels.append(label)
                
    #             previous_word_idx = word_idx  # FIXED: Update previous_word_idx

    #         tokenized["labels"] = aligned_labels
    #         tokenized_inputs.append(tokenized)

    #     self.df["tokenized"] = tokenized_inputs
    #     return tokenized_inputs

    # @timeit     # measure time for executing method
    # def train_test_split_ds(self, tokenizer, train_size=0.7, val_to_test=2/3,random_state=None):
    #     """
    #     Create train-test-split, save dataframe in object and return huggingface datasets
    #     """        
    #     self.tokenize_and_align_labels(tokenizer)

    #     # Convert tensor-based BatchEncoding to Python-native dicts
    #     self.df["tokenized_clean"] = self.df["tokenized"].apply(
    #         lambda x: {k: np.array(v).squeeze().tolist() for k, v in x.items()}
    #     )

    #     # Train/test split
    #     self.train_df, intermed = train_test_split(
    #         self.df, train_size=train_size, stratify=self.df["instance"],
    #         random_state=random_state
    #     )
        
    #     self.val_df, self.test_df = train_test_split(
    #         intermed, test_size=val_to_test, stratify=intermed["instance"],
    #         random_state=random_state
    #     )

    #     train_dataset = Dataset.from_list(self.train_df["tokenized_clean"].tolist())
    #     val_dataset = Dataset.from_list(self.val_df["tokenized_clean"].tolist())
    #     test_dataset = Dataset.from_list(self.test_df["tokenized_clean"].tolist())

    #     return train_dataset, val_dataset, test_dataset
    
    # # candidate for cleanup?
    # def prepare_dataset_for_tokenization(self):
    #     """Convert DataFrame to HuggingFace Dataset format before tokenization"""
    #     # Prepare the data in the format expected by HF datasets
    #     dataset_dict = {
    #         'tok_ContextBefore': self.df['tok_ContextBefore'].tolist(),
    #         'tok_Hit': self.df['tok_Hit'].tolist(), 
    #         'tok_ContextAfter': self.df['tok_ContextAfter'].tolist(),
    #         'biolabels': self.df['biolabels'].tolist(),
    #         'instance': self.df['instance'].tolist(),
    #     }
        
    #     return Dataset.from_dict(dataset_dict)

    # def tokenize_function(self, examples, tokenizer, max_length=64):
    #     """Optimized tokenization function for HF datasets.map()"""
    #     batch_size = len(examples['tok_Hit'])
        
    #     # Prepare batch data
    #     full_tokens_batch = []
    #     labels_batch = []
        
    #     for i in range(batch_size):
    #         # Concatenate tokens
    #         full_tokens = (examples['tok_ContextBefore'][i] + 
    #                       examples['tok_Hit'][i] + 
    #                       examples['tok_ContextAfter'][i])
            
    #         # Create labels
    #         len_before = len(examples['tok_ContextBefore'][i])
    #         len_hit = len(examples['tok_Hit'][i])
    #         len_after = len(examples['tok_ContextAfter'][i])
            
    #         hit_labels = (examples['biolabels'][i] if 
    #                      isinstance(examples['biolabels'][i], list) 
    #                      else ["O"] * len_hit)
            
    #         labels = ([-100] * len_before + 
    #                  hit_labels + 
    #                  [-100] * len_after)
            
    #         full_tokens_batch.append(full_tokens)
    #         labels_batch.append(labels)
        
    #     # Batch tokenization
    #     tokenized = tokenizer(
    #         full_tokens_batch,
    #         is_split_into_words=True,
    #         truncation=True,
    #         padding="max_length",
    #         max_length=max_length,
    #         return_tensors=None  # Return lists, not tensors
    #     )
        
    #     # Align labels for each example in batch
    #     aligned_labels_batch = []
    #     for i in range(batch_size):
    #         word_ids = tokenized.word_ids(batch_index=i)
    #         aligned_labels = self._align_labels_with_tokens(
    #             word_ids, labels_batch[i]
    #         )
    #         aligned_labels_batch.append(aligned_labels)
        
    #     tokenized["labels"] = aligned_labels_batch
    #     return tokenized

    # def _align_labels_with_tokens(self, word_ids, labels):
    #     """Helper function to align labels with wordpiece tokens"""
    #     aligned_labels = []
    #     previous_word_idx = None
        
    #     for word_idx in word_ids:
    #         if word_idx is None:
    #             aligned_labels.append(-100)
    #         else:
    #             if word_idx >= len(labels):
    #                 aligned_labels.append(-100)
    #                 continue
                    
    #             label = labels[word_idx]
    #             if label == -100:
    #                 aligned_labels.append(-100)
    #             else:
    #                 if word_idx == previous_word_idx:
    #                     # Subword continuation
    #                     if isinstance(label, str) and label.startswith('B-'):
    #                         aligned_labels.append(self.labeltoint[label.replace('B-', 'I-')])
    #                     elif isinstance(label, str):
    #                         aligned_labels.append(self.labeltoint[label])
    #                     else:
    #                         aligned_labels.append(label)
    #                 else:
    #                     # First subword of word
    #                     if isinstance(label, str):
    #                         aligned_labels.append(self.labeltoint[label])
    #                     else:
    #                         aligned_labels.append(label)
            
    #         previous_word_idx = word_idx
        
    #     return aligned_labels

    # @timeit                 # measure execution time
    # def train_test_split_ds_threaded(self, tokenizer, train_size=0.7, 
    #                                val_to_test_ratio=2/3, random_state=None, 
    #                                max_workers=None):
    #     """
    #     Alternative using ThreadPoolExecutor (good for I/O bound tasks)
    #     """
    #     if max_workers is None:
    #         max_workers = min(mp.cpu_count(), 12)
            
    #     print(f"Using {max_workers} threads for tokenization")
        
    #     # Prepare data
    #     rows_data = [(idx, row) for idx, row in self.df.iterrows()]
        
    #     # Thread-based processing
    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         tokenize_func = partial(
    #             self._tokenize_single_row_safe,
    #             tokenizer=tokenizer
    #         )
    #         results = list(executor.map(tokenize_func, rows_data))
        
    #     # Reconstruct results
    #     tokenized_dict = {idx: result for idx, result in results}
    #     self.df["tokenized_clean"] = [tokenized_dict[idx] for idx in self.df.index]

    #     # Train/test split
    #     self.train_df, intermed = train_test_split(
    #         self.df, train_size=train_size, 
    #         stratify=self.df["instance"],
    #         random_state=random_state
    #     )
        
    #     self.val_df, self.test_df = train_test_split(
    #         intermed, test_size=val_to_test_ratio, 
    #         stratify=intermed["instance"],
    #         random_state=random_state
    #     )
        
    #     train_dataset = Dataset.from_list(self.train_df["tokenized_clean"].tolist())
    #     val_dataset = Dataset.from_list(self.val_df["tokenized_clean"].tolist())
    #     test_dataset = Dataset.from_list(self.test_df["tokenized_clean"].tolist())

        
    #     return train_dataset, val_dataset, test_dataset

    # def _tokenize_single_row_safe(self, row_data, tokenizer):
    #     """Thread-safe version of single row tokenization"""
    #     idx, row = row_data
        
    #     # Create a copy of tokenizer for thread safety
    #     # (though modern transformers tokenizers are generally thread-safe)
        
    #     full_tokens = row["tok_ContextBefore"] + row["tok_Hit"] + row["tok_ContextAfter"]
        
    #     len_before = len(row["tok_ContextBefore"])
    #     len_hit = len(row["tok_Hit"])
    #     len_after = len(row["tok_ContextAfter"])
        
    #     hit_labels = row["biolabels"] if isinstance(row["biolabels"], list) else ["O"] * len_hit
        
    #     labels = ([-100] * len_before + hit_labels + [-100] * len_after)
        
    #     tokenized = tokenizer(
    #         full_tokens,
    #         is_split_into_words=True,
    #         truncation=True,
    #         padding="max_length",
    #         max_length=64,
    #         return_tensors=None  # Return lists instead of tensors
    #     )
        
    #     word_ids = tokenized.word_ids()
    #     aligned_labels = self._align_labels_with_tokens(word_ids, labels)
        
    #     result = {
    #         'input_ids': tokenized['input_ids'],
    #         'attention_mask': tokenized['attention_mask'],
    #         'labels': aligned_labels
    #     }
        
    #     return idx, result
    
    
    



    
# Load your metrics once, outside the compute_metrics function for efficiency
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
# Add any other metrics you need
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2) # Get predicted label IDs

    # Flatten and filter out -100 (ignored tokens)
    true_labels = []
    true_predictions = []
    for pred_row, label_row in zip(predictions, labels):
        for p, l in zip(pred_row, label_row):
            if l != -100:
                true_labels.append(l)
                true_predictions.append(p)

    results = {}
    if len(true_labels) == 0:
        # Handle cases with no valid labels to avoid errors
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    # Compute each metric. Remember to specify 'average' for F1, Precision, Recall
    # in multi-class settings. Accuracy doesn't typically need 'average'.
    accuracy_result = accuracy_metric.compute(predictions=true_predictions, references=true_labels)
    f1_result = f1_metric.compute(predictions=true_predictions, references=true_labels, average="micro") # Or "macro", "weighted"
    precision_result = precision_metric.compute(predictions=true_predictions, references=true_labels, average="micro")
    recall_result = recall_metric.compute(predictions=true_predictions, references=true_labels, average="micro")

    # Combine results into a single dictionary
    results.update(accuracy_result) # Adds {'accuracy': value}
    results.update(f1_result)      # Adds {'f1': value}
    results.update(precision_result) # Adds {'precision': value}
    results.update(recall_result)    # Adds {'recall': value}

    return results

    

# # to check for utility and inclusion into 
# def predict_labels(model, tokenizer, context_before, hit, context_after, label_map):
#     model.eval()

#     # Tokenize manually split input
#     tokens = word_tokenize(context_before) + word_tokenize(hit) + word_tokenize(context_after)

#     # Tokenize with BERT tokenizer
#     tokenized = tokenizer(
#         tokens,
#         is_split_into_words=True,
#         return_tensors="pt",
#         truncation=True,
#         padding="max_length",
#         max_length=64,
#     )

#     with torch.no_grad():
#         output = model(**tokenized)

#     # Get predicted label IDs
#     predictions = torch.argmax(output.logits, dim=-1).squeeze().tolist()
#     word_ids = tokenized.word_ids()

#     # Align word-level labels
#     aligned_labels = []
#     last_word_id = None
#     for i, word_id in enumerate(word_ids):
#         if word_id is None or word_id == last_word_id:
#             aligned_labels.append(None)
#         else:
#             aligned_labels.append(predictions[i])
#             last_word_id = word_id

#     # Only return tokens + predictions for the Hit section
#     hit_start = len(word_tokenize(context_before))
#     hit_end = hit_start + len(word_tokenize(hit))

#     # Map back to readable labels
#     readable_labels = [
#         (tokens[i], label_map[aligned_labels[i]]) 
#         for i in range(len(tokens)) 
#         if aligned_labels[i] is not None and hit_start <= i < hit_end
#     ]

#     return readable_labels



