
import pandas as pd
import numpy as np

import multiprocessing as mp
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer
import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize,sent_tokenize

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
    def __init__(self, training=False,language="german", csvfile=None, strinput=None):
        # setting core properties of data
        self.language=language              # marks (main) language of dataset
        self.training = training            # markes whether data set is annotated as training data
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
    # BASIC DATAFRAME OUTPUT

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
        Tokenize text columns and create bio labels for "Hit" column directly on the dataset.
        """
        if self.dataset is None:
            raise ValueError("No dataset available. Load data first.")

        # define the function to apply to each example in the dataset
        def process_example(example):
            # tokenize text columns
            example['tok_ContextBefore'] = word_tokenize(str(example['ContextBefore']), language=self.language) if pd.notnull(example['ContextBefore']) else []
            example['tok_Hit'] = word_tokenize(str(example['Hit']), language=self.language) if pd.notnull(example['Hit']) else []
            example['tok_ContextAfter'] = word_tokenize(str(example['ContextAfter']), language=self.language) if pd.notnull(example['ContextAfter']) else []
            example['tok_APC'] = word_tokenize(str(example['APC']), language=self.language) if pd.notnull(example['APC']) else []

            # apply BIO labeling for rows with APCs
            if example['instance'] == 1:
                example['biolabels'] = self.generate_biolabels_single(example['tok_Hit'], example['tok_APC'])
            else:
                example['biolabels'] = ["O"] * len(example['tok_Hit'])

            # check for length mismatch
            if len(example['tok_Hit']) != len(example['biolabels']):
                print(f"Length mismatch: tok_Hit={len(example['tok_Hit'])}, biolabels={len(example['biolabels'])}")

            return example

        # Use .map() for efficient processing
        self.dataset = self.dataset.map(process_example, num_proc=mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1,
                                        desc="Generating BIO labels and tokenizing text columns")

        # Remove original text columns if no longer needed to save memory
        # self.dataset = self.dataset.remove_columns(self.textcols)


    def generate_biolabels_single(self, sentence_tokens, apc_tokens):
        # This method remains largely the same as it operates on single token lists
        if not sentence_tokens or not apc_tokens:
            return ["O"] * len(sentence_tokens) if sentence_tokens else []

        labels = ["O"] * len(sentence_tokens)
        sentence_tokens_lower = [t.lower() for t in sentence_tokens]
        apc_tokens_lower = [t.lower() for t in apc_tokens]

        for i in range(len(sentence_tokens_lower) - len(apc_tokens_lower) + 1):
            if sentence_tokens_lower[i:i+len(apc_tokens_lower)] == apc_tokens_lower:
                labels[i] = "B-APC"
                for j in range(1, len(apc_tokens_lower)):
                    if i + j < len(labels):
                        labels[i + j] = "I-APC"
        return labels

    # Merging duplicates is still best done on a DataFrame if you need to modify rows based on indices
    # and then convert back to a Dataset. Or, try to achieve it with Dataset.group_by and .map if possible.
    # For now, keeping find_duplicate_hits and merge_apc_annotations DataFrame-based might be simpler
    # if the logic is complex, but be aware of the conversion overhead for very large datasets.

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
            self.df = self.dataset.to_pandas()


        id_tuples = self.find_duplicate_hits()

        print("Merging BIO-labels")
        processed=0
        for t in id_tuples:
            consolidated = self.df.loc[t[0]]['biolabels']
            for i in t[1:]:
                processed+=1
                tmp_bio = self.df.loc[i]['biolabels']
                for count in range(len(consolidated)):
                    if tmp_bio[count] != 'O':
                        consolidated[count] = tmp_bio[count]

            self.df.at[t[0],'biolabels'] = consolidated

        print(f"Processed {processed} duplicate rows")

        print("Removing now-redundant rows")
        self.df = self.df.drop_duplicates(subset='Hit',keep='first')

        # Convert back to Dataset after modification
        self.dataset = Dataset.from_pandas(self.df, preserve_index=False)
        print("Dataset updated after merging and removing duplicates.")

    
    
    @timeit     # measure time for executing method
    def generate_biolabels_df(self):
        """
        Tokenise text columns and create bio labels for "Hit" column
        """
        for col in self.textcols:
            self.df['tok_' + col] = self.df[col].apply(lambda x: word_tokenize(str(x), language=self.language) if pd.notnull(x) else [])

        # apply BIO labeling for rows with APCs
        mask = self.df["instance"] == 1
        self.df.loc[mask, "biolabels"] = self.df[mask].apply(
            lambda row: self.generate_biolabels_single(row["tok_Hit"], row["tok_APC"]), axis=1
        )

        # assign dummy labels to the rest
        self.df.loc[~mask, "biolabels"] = self.df.loc[~mask, "tok_Hit"].apply(
            lambda tokens: ["O"] * len(tokens) if isinstance(tokens, list) else []
        )       
        
        mask = self.df["instance"] == 1
        for idx in self.df[mask].index:
            tok_hit_len = len(self.df.loc[idx, "tok_Hit"])
            biolabel_len = len(self.df.loc[idx, "biolabels"])
            if tok_hit_len != biolabel_len:
                print(f"Length mismatch at index {idx}: tok_Hit={tok_hit_len}, biolabels={biolabel_len}")                         
    
    
    def generate_biolabels_single(self, sentence_tokens, apc_tokens):
        """
        Given tokenized sentence and tokenized APC, return BIO labels for sentence.
        All instances of the same APC in the same sentence are marked, i.e. we can capture repetitions appropriately. 
        In line with what the base dataset annotation provides, however, this only works for one specific APC.
        """
        if not sentence_tokens or not apc_tokens:
            return ["O"] * len(sentence_tokens) if sentence_tokens else []
        
        labels = ["O"] * len(sentence_tokens)
        sentence_tokens_lower = [t.lower() for t in sentence_tokens]
        apc_tokens_lower = [t.lower() for t in apc_tokens]
        
        # Check if we have enough tokens left to match
        for i in range(len(sentence_tokens_lower) - len(apc_tokens_lower) + 1):
            if sentence_tokens_lower[i:i+len(apc_tokens_lower)] == apc_tokens_lower:
                labels[i] = "B-APC"
                for j in range(1, len(apc_tokens_lower)):
                    if i + j < len(labels):  # Safety check
                        labels[i + j] = "I-APC"
        
        return labels


    def tokenize_and_align_labels(self,tokenizer):
        # Tokenize and align BIO labels, assigning -100 to all elements in context
        tokenized_inputs = []

        for _, row in self.df.iterrows():
            # 1. Concatenate tokens
            full_tokens = row["tok_ContextBefore"] + row["tok_Hit"] + row["tok_ContextAfter"]

            # 2. Create full BIO labels
            len_before = len(row["tok_ContextBefore"])
            len_hit = len(row["tok_Hit"])
            len_after = len(row["tok_ContextAfter"])

            hit_labels = row["biolabels"] if isinstance(row["biolabels"], list) else ["O"] * len_hit

            labels = (
                [-100] * len_before +
                [label for label in hit_labels] +
                [-100] * len_after
            )

            # 3. Tokenize with is_split_into_words=True
            tokenized = tokenizer(
                full_tokens,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=64,  # Adjust as needed
                return_tensors="pt"
            )

            # 4. Align labels with wordpieces
            word_ids = tokenized.word_ids(batch_index=0)
            aligned_labels = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)
                else:
                    label = labels[word_idx]
                    if label == -100:
                        aligned_labels.append(-100)
                    else:
                        # Handle subword continuation
                        if word_idx == previous_word_idx:
                            # This is a continuation of the previous word
                            if isinstance(label, str) and label.startswith('B-'):
                                # Convert B- to I- for subword continuations
                                aligned_labels.append(self.labeltoint[label.replace('B-', 'I-')])
                            elif isinstance(label, str):
                                # Keep the same label for I- and O
                                aligned_labels.append(self.labeltoint[label])
                            else:
                                # If label is already an integer, use it directly
                                aligned_labels.append(label)
                        else:
                            # First subword of this word, keep original label
                            if isinstance(label, str):
                                aligned_labels.append(self.labeltoint[label])
                            else:
                                aligned_labels.append(label)
                
                previous_word_idx = word_idx  # FIXED: Update previous_word_idx

            tokenized["labels"] = aligned_labels
            tokenized_inputs.append(tokenized)

        self.df["tokenized"] = tokenized_inputs
        return tokenized_inputs

    @timeit     # measure time for executing method
    def train_test_split_ds(self, tokenizer, train_size=0.7, val_to_test=2/3,random_state=None):
        """
        Create train-test-split, save dataframe in object and return huggingface datasets
        """        
        self.tokenize_and_align_labels(tokenizer)

        # Convert tensor-based BatchEncoding to Python-native dicts
        self.df["tokenized_clean"] = self.df["tokenized"].apply(
            lambda x: {k: np.array(v).squeeze().tolist() for k, v in x.items()}
        )

        # Train/test split
        self.train_df, intermed = train_test_split(
            self.df, train_size=train_size, stratify=self.df["instance"],
            random_state=random_state
        )
        
        self.val_df, self.test_df = train_test_split(
            intermed, test_size=val_to_test, stratify=intermed["instance"],
            random_state=random_state
        )

        train_dataset = Dataset.from_list(self.train_df["tokenized_clean"].tolist())
        val_dataset = Dataset.from_list(self.val_df["tokenized_clean"].tolist())
        test_dataset = Dataset.from_list(self.test_df["tokenized_clean"].tolist())

        return train_dataset, val_dataset, test_dataset
    
    # candidate for cleanup?
    def prepare_dataset_for_tokenization(self):
        """Convert DataFrame to HuggingFace Dataset format before tokenization"""
        # Prepare the data in the format expected by HF datasets
        dataset_dict = {
            'tok_ContextBefore': self.df['tok_ContextBefore'].tolist(),
            'tok_Hit': self.df['tok_Hit'].tolist(), 
            'tok_ContextAfter': self.df['tok_ContextAfter'].tolist(),
            'biolabels': self.df['biolabels'].tolist(),
            'instance': self.df['instance'].tolist(),
        }
        
        return Dataset.from_dict(dataset_dict)

    def tokenize_function(self, examples, tokenizer, max_length=64):
        """Optimized tokenization function for HF datasets.map()"""
        batch_size = len(examples['tok_Hit'])
        
        # Prepare batch data
        full_tokens_batch = []
        labels_batch = []
        
        for i in range(batch_size):
            # Concatenate tokens
            full_tokens = (examples['tok_ContextBefore'][i] + 
                          examples['tok_Hit'][i] + 
                          examples['tok_ContextAfter'][i])
            
            # Create labels
            len_before = len(examples['tok_ContextBefore'][i])
            len_hit = len(examples['tok_Hit'][i])
            len_after = len(examples['tok_ContextAfter'][i])
            
            hit_labels = (examples['biolabels'][i] if 
                         isinstance(examples['biolabels'][i], list) 
                         else ["O"] * len_hit)
            
            labels = ([-100] * len_before + 
                     hit_labels + 
                     [-100] * len_after)
            
            full_tokens_batch.append(full_tokens)
            labels_batch.append(labels)
        
        # Batch tokenization
        tokenized = tokenizer(
            full_tokens_batch,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None  # Return lists, not tensors
        )
        
        # Align labels for each example in batch
        aligned_labels_batch = []
        for i in range(batch_size):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned_labels = self._align_labels_with_tokens(
                word_ids, labels_batch[i]
            )
            aligned_labels_batch.append(aligned_labels)
        
        tokenized["labels"] = aligned_labels_batch
        return tokenized

    def _align_labels_with_tokens(self, word_ids, labels):
        """Helper function to align labels with wordpiece tokens"""
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            else:
                if word_idx >= len(labels):
                    aligned_labels.append(-100)
                    continue
                    
                label = labels[word_idx]
                if label == -100:
                    aligned_labels.append(-100)
                else:
                    if word_idx == previous_word_idx:
                        # Subword continuation
                        if isinstance(label, str) and label.startswith('B-'):
                            aligned_labels.append(self.labeltoint[label.replace('B-', 'I-')])
                        elif isinstance(label, str):
                            aligned_labels.append(self.labeltoint[label])
                        else:
                            aligned_labels.append(label)
                    else:
                        # First subword of word
                        if isinstance(label, str):
                            aligned_labels.append(self.labeltoint[label])
                        else:
                            aligned_labels.append(label)
            
            previous_word_idx = word_idx
        
        return aligned_labels

    @timeit                 # measure execution time
    def train_test_split_ds_threaded(self, tokenizer, train_size=0.7, 
                                   val_to_test_ratio=2/3, random_state=None, 
                                   max_workers=None):
        """
        Alternative using ThreadPoolExecutor (good for I/O bound tasks)
        """
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 12)
            
        print(f"Using {max_workers} threads for tokenization")
        
        # Prepare data
        rows_data = [(idx, row) for idx, row in self.df.iterrows()]
        
        # Thread-based processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tokenize_func = partial(
                self._tokenize_single_row_safe,
                tokenizer=tokenizer
            )
            results = list(executor.map(tokenize_func, rows_data))
        
        # Reconstruct results
        tokenized_dict = {idx: result for idx, result in results}
        self.df["tokenized_clean"] = [tokenized_dict[idx] for idx in self.df.index]

        # Train/test split
        self.train_df, intermed = train_test_split(
            self.df, train_size=train_size, 
            stratify=self.df["instance"],
            random_state=random_state
        )
        
        self.val_df, self.test_df = train_test_split(
            intermed, test_size=val_to_test_ratio, 
            stratify=intermed["instance"],
            random_state=random_state
        )
        
        train_dataset = Dataset.from_list(self.train_df["tokenized_clean"].tolist())
        val_dataset = Dataset.from_list(self.val_df["tokenized_clean"].tolist())
        test_dataset = Dataset.from_list(self.test_df["tokenized_clean"].tolist())

        
        return train_dataset, val_dataset, test_dataset

    def _tokenize_single_row_safe(self, row_data, tokenizer):
        """Thread-safe version of single row tokenization"""
        idx, row = row_data
        
        # Create a copy of tokenizer for thread safety
        # (though modern transformers tokenizers are generally thread-safe)
        
        full_tokens = row["tok_ContextBefore"] + row["tok_Hit"] + row["tok_ContextAfter"]
        
        len_before = len(row["tok_ContextBefore"])
        len_hit = len(row["tok_Hit"])
        len_after = len(row["tok_ContextAfter"])
        
        hit_labels = row["biolabels"] if isinstance(row["biolabels"], list) else ["O"] * len_hit
        
        labels = ([-100] * len_before + hit_labels + [-100] * len_after)
        
        tokenized = tokenizer(
            full_tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors=None  # Return lists instead of tensors
        )
        
        word_ids = tokenized.word_ids()
        aligned_labels = self._align_labels_with_tokens(word_ids, labels)
        
        result = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': aligned_labels
        }
        
        return idx, result
    
    
    ###
    # Experimenting with direct use of datasets
    def create_dataset_from_df(self):
        """Convert processed DataFrame to HuggingFace Dataset"""
        if self.df is None:
            raise ValueError("No dataframe available. Load data first.")
            
        # Convert DataFrame to Dataset
        self.dataset = Dataset.from_pandas(self.df)
        return self.dataset

    # Add these methods to your APCData class

    @timeit
    def tokenize_dataset(self, tokenizer, num_proc=None, batch_size=1000, 
                        max_length=64, cache_dir=None):
        """
        General tokenization method that works for both training and inference data
        """
        # If the tokenizer is "fast" (Rust-based), it handles its own parallelism internally.
        # Using num_proc > 1 with a fast tokenizer can actually *disable* its internal parallelism
        # and introduce unnecessary multiprocessing overhead.
        if tokenizer.is_fast:
            effective_num_proc = None # Let the tokenizer handle parallelism
            print("Using a 'fast' tokenizer. Setting num_proc to None for optimal performance.")
        else:
            effective_num_proc = num_proc if num_proc is not None else (min(4, mp.cpu_count() - 1) if mp.cpu_count() > 1 else 1)
            print(f"Processing with {effective_num_proc} processes.")
        
        # Create initial dataset if not exists
        if self.dataset is None:
            self.create_dataset_from_df()
        
        # Define tokenization function that handles both training and inference
        def tokenize_function(examples):
            batch_size = len(examples['tok_Hit'])
            
            # Prepare batch data
            full_tokens_batch = []
            labels_batch = []
            
            for i in range(batch_size):
                # Concatenate tokens
                full_tokens = (examples['tok_ContextBefore'][i] + 
                            examples['tok_Hit'][i] + 
                            examples['tok_ContextAfter'][i])
                full_tokens_batch.append(full_tokens)
                
                # Handle labels based on whether this is training data
                if self.training and 'biolabels' in examples:
                    # Create labels for training data
                    len_before = len(examples['tok_ContextBefore'][i])
                    len_hit = len(examples['tok_Hit'][i])
                    len_after = len(examples['tok_ContextAfter'][i])
                    
                    hit_labels = (examples['biolabels'][i] if 
                                isinstance(examples['biolabels'][i], list) 
                                else ["O"] * len_hit)
                    
                    labels = ([-100] * len_before + 
                            hit_labels + 
                            [-100] * len_after)
                    labels_batch.append(labels)
                else:
                    # For inference data, we don't need real labels
                    labels_batch.append(None)
            
            # Batch tokenization
            tokenized = tokenizer(
                full_tokens_batch,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors=None
            )
            
            # Align labels only for training data
            if self.training and any(labels is not None for labels in labels_batch):
                aligned_labels_batch = []
                for i in range(batch_size):
                    if labels_batch[i] is not None:
                        word_ids = tokenized.word_ids(batch_index=i)
                        aligned_labels = self._align_labels_with_tokens(
                            word_ids, labels_batch[i]
                        )
                        aligned_labels_batch.append(aligned_labels)
                    else:
                        # Dummy labels for consistency
                        aligned_labels_batch.append([-100] * len(tokenized['input_ids'][i]))
                
                tokenized["labels"] = aligned_labels_batch
            
            return tokenized
        
        # Determine which columns to remove based on data type
        columns_to_remove = [col for col in self.dataset.column_names 
                            if col not in (['instance'] if self.training else [])]
        
        # Apply tokenization
        tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=columns_to_remove,
            desc="Tokenizing",
            cache_file_name=f"{cache_dir}/tokenized.arrow" if cache_dir else None
        )
        
        self.tokenized_dataset = tokenized_dataset
        return tokenized_dataset

    @timeit                 
    def tokenize_and_split_native(self, tokenizer, train_size=0.7, val_to_test_ratio=2/3, 
                                random_state=None, num_proc=None, batch_size=1000,
                                max_length=64, cache_dir=None):
        """
        Tokenize and split method specifically for training data
        """
        if not self.training:
            raise ValueError("This method is only for training data. Use tokenize_dataset() for inference data.")
        
        # First tokenize the dataset
        tokenized_dataset = self.tokenize_dataset(
            tokenizer, num_proc, batch_size, max_length, cache_dir
        )
        
        # Convert to pandas temporarily for stratified splitting
        temp_df = tokenized_dataset.to_pandas()
        
        # Stratified splits
        train_df, intermed = train_test_split(
            temp_df, train_size=train_size, 
            stratify=temp_df["instance"],
            random_state=random_state
        )
        
        val_df, test_df = train_test_split(
            intermed, test_size=val_to_test_ratio,
            stratify=intermed["instance"], 
            random_state=random_state
        )
        
        # Convert back to datasets (remove instance column)
        train_dataset = Dataset.from_pandas(
            train_df.drop(columns=['instance', '__index_level_0__'], errors='ignore')
        )
        val_dataset = Dataset.from_pandas(
            val_df.drop(columns=['instance', '__index_level_0__'], errors='ignore')
        )
        test_dataset = Dataset.from_pandas(
            test_df.drop(columns=['instance', '__index_level_0__'], errors='ignore')
        )
        
        # Store as DatasetDict for easy access
        self.datasets = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        return self.datasets

    def prepare_for_inference(self, tokenizer, max_length=64, num_proc=None, batch_size=1000):
        """
        Convenience method specifically for preparing inference data
        """
        if self.training:
            print("Warning: This appears to be training data. Consider using tokenize_and_split_native() instead.")
        
        return self.tokenize_dataset(
            tokenizer=tokenizer,
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

    

# to check for utility and inclusion into 
def predict_labels(model, tokenizer, context_before, hit, context_after, label_map):
    model.eval()

    # Tokenize manually split input
    tokens = word_tokenize(context_before) + word_tokenize(hit) + word_tokenize(context_after)

    # Tokenize with BERT tokenizer
    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64,
    )

    with torch.no_grad():
        output = model(**tokenized)

    # Get predicted label IDs
    predictions = torch.argmax(output.logits, dim=-1).squeeze().tolist()
    word_ids = tokenized.word_ids()

    # Align word-level labels
    aligned_labels = []
    last_word_id = None
    for i, word_id in enumerate(word_ids):
        if word_id is None or word_id == last_word_id:
            aligned_labels.append(None)
        else:
            aligned_labels.append(predictions[i])
            last_word_id = word_id

    # Only return tokens + predictions for the Hit section
    hit_start = len(word_tokenize(context_before))
    hit_end = hit_start + len(word_tokenize(hit))

    # Map back to readable labels
    readable_labels = [
        (tokens[i], label_map[aligned_labels[i]]) 
        for i in range(len(tokens)) 
        if aligned_labels[i] is not None and hit_start <= i < hit_end
    ]

    return readable_labels



