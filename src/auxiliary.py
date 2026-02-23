import os
import yaml
import pandas as pd
import numpy as np


import multiprocessing as mp
from functools import partial
import time

import torch
from tqdm import tqdm
from multiprocessing import Pool
import functools

from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer



from datasets import load_dataset, Dataset, DatasetDict
import evaluate
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Time taken: {elapsed:.6f} seconds')
        return result
    return wrapper



# class for data input and transformer training
class APCData:
    def __init__(self, model_config:str="german-bilou", config_file:str="config.yaml"):
        """        
        Initializes the APCData class with configurations and prepares the dataset for fine-tuning.

        Parameters:
            model_config (str): Specifies the model configuration section in the YAML file. Default is "german-bilou".
            config_file (str): Path to the YAML configuration file. Default is "config.yaml".

        Raises:
            ValueError: If an undefined label scheme or missing keys are detected.
            
        Example Usage:
            >>> data = APCData(model_config="english-bio", config_file="path/to/config.yaml")
        """
        
        # extract config data
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Ensure the general and model-specific sections exist
        if "training" not in self.config or model_config not in self.config:
            raise KeyError(f"Missing 'training' or '{model_config}' section in the configuration file.")
        
        # Merge general settings with model-specific settings
        self.config = {**self.config["training"], **self.config[model_config]}
        
        # validate and set desired label scheme (bilou as new default)
        if self.config["label_scheme"] in ['bilou', 'bio']:
            self.label_list = ["B-APC", "I-APC", "L-APC", "U-APC", "O"] if self.config["label_scheme"] == 'bilou' else ["B-APC", "I-APC", "O"]  # BILOU labels by default

            self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
            self.id2label = {idx: label for idx, label in enumerate(self.label_list)}  
        else:
            raise ValueError("Undefined value for label_scheme, allowed values: bilou (default) or bio.")
        
        # import tokenizer and model
        # number of labels for classification determined by length of `label_list`
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["modelname"]) 
        self.model = AutoModelForTokenClassification.from_pretrained(self.config["modelname"], num_labels=len(self.label_list))

        self.training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            evaluation_strategy=self.config["evaluation_strategy"],
            save_strategy=self.config["save_strategy"],
            learning_rate=self.config["learning_rate"],
            per_device_train_batch_size=self.config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["per_device_eval_batch_size"],
            num_train_epochs=self.config["num_train_epochs"],
            weight_decay=self.config["weight_decay"],
            logging_dir="./logs",
            logging_steps=10,
            save_total_limit=self.config["save_total_limit"],
            push_to_hub=self.config["push_to_hub"], 
            hub_model_id=self.config["hub_model_id"], 
            revision=self.config["hf_revision"],
            hub_strategy=self.config["hub_strategy"]
        )
        
        # instantiate storage
        self.dataset = None                 # Primary data storage
        self.df = None                      # Pandas data storage
        self.tokenized_dataset = None       # Stores the chunked, tokenized data
        
        # column headers required in input csv
        # keep rather rigid structure for now
        self.textcols = ['ContextBefore', 'Hit', 'ContextAfter','APC']      

        # load training data from csv
        self._load_csv(self.config["datacsv"])
        
        # Load your metrics once, outside the compute_metrics method for efficiency
        self.accuracy_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")
        # Add any other metrics you need
        self.precision_metric = evaluate.load("precision")
        self.recall_metric = evaluate.load("recall")


    # INPUT METHOD
    
    def _load_csv(self, path):
        print(f'Loading data from {path}...')
        try:
            df = pd.read_csv(path)
            # validate presence of required columns
            if not all(col in df.columns for col in self.textcols):
                raise ValueError(f"CSV file is missing one or more required columns: {self.textcols}")
            # Add a unique 'original_idx' based on the DataFrame's default integer index to ensure each original row has a persistent ID
            df['original_idx'] = df.index
            self.df = df
            self.dataset = Dataset.from_pandas(self.df, preserve_index=False) # preserve_index=False if 'original_idx' is already a column
            print('CSV loaded and original_idx created.')
        except FileNotFoundError:
            print(f"Error: CSV file not found at {path}")
            self.dataset = None
            self.df = None
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.dataset = None
            self.df = None

    ########################
    # BASIC DATA OUTPUT

    def get_dataset(self, instances:bool=False):
        """Return dataset from object

        Args:
            subset (bool, optional): Return the subsect of data with APC instances (i.e. "instance" set to 1). Defaults to False.

        Returns:
            Dataset: the full or subsetted dataset
        """
        if instances:
            return self.dataset.filter(lambda x: x['instance'] == 1)
        else:
            return self.dataset
        
    
    def get_df(self,instances:bool=False):
        """Return dataframe from object
        
        Args:
            subset (bool, optional): Return the subsect of data with APC instances (i.e. "instance" set to 1). Defaults to False.

        Returns:
            Dataset: the full or subsetted dataset
        """
        if instances:
            return self.df[self.df.instance==1]
        else:
            return self.df
        

    def get_df_at_index(self,index:int):
        """Return data line at specific index

        Args:
            index (int): The row index of the DataFrame to retrieve.

        Returns:
            pd.Series: A Series object containing the data for the specified index.
        """
        return self.df.loc[index]
    
    
    ##################################
    # data preparation

    def prepare_training_data(self, test_size:float=None, val_size:float=None, random_state:int=None, num_proc:int=None, max_length:int=None, overlap_size:int=None):
        """Main entry point: orchestrates all preprocessing steps in efficient order"""
        
        # Use default values from config if none are provided
        if test_size is None:
            try:
                test_size = self.config["test_size"]
            except KeyError:
                raise ValueError("Missing 'test_size' key in the configuration file.")
        
        if val_size is None:
            try:
                val_size = self.config["val_size"]
            except KeyError:
                raise ValueError("Missing 'val_size' key in the configuration file.")
        
        if random_state is None:
            try:
                random_state = self.config["random_state"]
            except KeyError:
                raise ValueError("Missing 'random_state' key in the configuration file.")
        
        if num_proc is None:
            try:
                num_proc = self.config["num_proc"]
            except KeyError:
                raise ValueError("Missing 'num_proc' key in the configuration file.")
        
        if max_length is None:
            try:
                max_length = self.config["max_length"]
            except KeyError:
                raise ValueError("Missing 'max_length' key in the configuration file.")
        
        if overlap_size is None:
            try:
                overlap_size = self.config["overlap_size"]
            except KeyError:
                raise ValueError("Missing 'overlap_size' key in the configuration file.")
        
        # Beginning processing
        print("Step 1: Merging duplicate Hits...")
        self._merge_duplicate_hits_raw()
        
        print("Step 2: Splitting, labeling and creating chunks...")
        train, val, test = self.tokenize_and_split(
            test_size=test_size, val_size=val_size, random_state=random_state,
            num_proc=num_proc, max_length=max_length, overlap_size=overlap_size
        )
        return train, val, test

    def _merge_duplicate_hits_raw(self):
        """Merge duplicate Hits before tokenization by combining APC values"""
        if self.df is None:
            raise ValueError("No dataframe available. Load data first.") 
        
        # Group by Hit and merge APC values
        agg_dict = {col: 'first' for col in self.df.columns if col != 'APC'}
        agg_dict['APC'] = lambda apcs: ' | '.join(apcs.dropna().unique())  
        agg_dict['instance'] = 'max'  # If any row has instance=1, keep it
        
        self.df = self.df.groupby('Hit', as_index=False).agg(agg_dict)
        self.dataset = Dataset.from_pandas(self.df)
        
        print(f"Merged to {len(self.df)} unique Hits")


    
    ###
    # Experimenting with direct use of datasets
    def create_dataset_from_df(self):
        """Convert processed DataFrame to HuggingFace Dataset"""
        if self.df is None:
            raise ValueError("No dataframe available. Load data first.")
            
        # Convert DataFrame to Dataset
        self.dataset = Dataset.from_pandas(self.df)
        return self.dataset


    def tokenize_dataset(self, dataset, num_proc=None,
                        max_length=512, overlap=64, cache_dir=None):
        """
        Robust tokenization method using a two-step process to avoid PyArrow issues.
        Step 1: Generate all chunks as Python objects
        Step 2: Create new dataset from chunks
        """
        
        if dataset is None:
            raise ValueError("No dataset available. Load data first.")
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set.")
        
        
        print('Generating all chunks...')
        
        # Step 1: Process all examples and collect chunks in memory
        all_chunks = []
        
        # Use multiprocessing if available for chunk generation
        if num_proc and num_proc > 1:
            
            # Create a partial function with fixed parameters
            process_func = functools.partial(
                self._create_chunks_for_example_static,
                tokenizer=self.tokenizer,
                label2id=self.label2id,
                label_scheme=self.config["label_scheme"],
                max_length=max_length,
                overlap=overlap
            )
            
            print(f"Using {num_proc} processes for chunk generation...")
            with Pool(num_proc) as pool:
                chunk_lists = pool.map(process_func, dataset)
                
            # Flatten the list of lists
            for chunk_list in chunk_lists:
                all_chunks.extend(chunk_list)
        else:
            # Single-threaded processing
            print("Processing examples sequentially...")
            for i, example in enumerate(tqdm(dataset, desc="Creating chunks")):
                chunks = self._create_chunks_for_example_static(example, self.tokenizer, self.label2id, self.config["label_scheme"], max_length, overlap)
                all_chunks.extend(chunks)
                
                if i % 100 == 0 and i > 0:
                    print(f"Processed {i} examples, generated {len(all_chunks)} chunks so far")
        
        print(f'Generated {len(all_chunks)} total chunks from {len(dataset)} examples')
        
        if len(all_chunks) == 0:
            raise ValueError("No chunks were generated. Check your input data.")
        
        # Step 2: Create dataset from chunks
        print('Creating dataset from chunks...')
        
        # Convert to the format expected by Dataset.from_list()
        # Ensure all values are native Python types, not numpy arrays
        clean_chunks = []
        for chunk in all_chunks:
            clean_chunk = {}
            for key, value in chunk.items():
                if isinstance(value, np.ndarray):
                    clean_chunk[key] = value.tolist()
                elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    clean_chunk[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
                else:
                    clean_chunk[key] = value
            clean_chunks.append(clean_chunk)
        
        tokenized_dataset = Dataset.from_list(clean_chunks)
        
        print(f"Created tokenized dataset with {len(tokenized_dataset)} chunks")
        print(f"Dataset columns: {tokenized_dataset.column_names}")
        
        # Save to cache if requested
        if cache_dir:
            import os
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "tokenized_chunked.arrow")
            tokenized_dataset.save_to_disk(cache_path)
            print(f"Saved tokenized dataset to {cache_path}")
        
        return tokenized_dataset

    @staticmethod
    def _create_chunks_for_example_static(example, tokenizer, label2id, label_scheme, max_length, overlap):
        """
        Static method for creating chunks - can be used with multiprocessing.
        Returns a list of chunk dictionaries with scalar values.
        """
        import pandas as pd
        
        original_example_idx = example['original_idx']
        original_instance = example.get('instance', 0)
        
        context_before_text = str(example['ContextBefore']) if pd.notnull(example['ContextBefore']) else ""
        hit_text = str(example['Hit']) if pd.notnull(example['Hit']) else ""
        context_after_text = str(example['ContextAfter']) if pd.notnull(example['ContextAfter']) else ""
        
        full_text = context_before_text + hit_text + context_after_text
        
        # Skip empty texts
        if len(full_text.strip()) == 0:
            return []
        
        try:
            full_encoding = tokenizer(
                full_text,
                truncation=False,
                padding=False,
                return_tensors=None,
                return_offsets_mapping=True,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True
            )
        except Exception as e:
            print(f"Tokenization failed for example {original_example_idx}: {e}")
            return []
        
        if len(full_encoding['input_ids']) == 0:
            return []
        
        # Calculate character ranges
        cb_len = len(context_before_text)
        h_len = len(hit_text)
        hit_char_start_in_full = cb_len
        hit_char_end_in_full = cb_len + h_len
        
        # Prepare labels
        # Initialize all to -100 (ignore)
        full_sequence_labels = [-100] * len(full_encoding['input_ids'])

        # First pass: label all Hit tokens as 'O' (they're in the annotated region)
        for t_idx, (c_start, c_end) in enumerate(full_encoding['offset_mapping']):
            if c_start is None or c_end is None:
                continue
            # Check if token overlaps with Hit
            if max(c_start, hit_char_start_in_full) < min(c_end, hit_char_end_in_full):
                full_sequence_labels[t_idx] = label2id["O"]  # Default label for Hit

        # Second pass: overwrite with APC labels where applicable
        if example['instance'] == 1:
            apc_strings = str(example['APC']).split(' | ')
            hit_text_lower = hit_text.lower()
            
            for apc_string in apc_strings:
                apc_string = apc_string.strip().lower()
                if not apc_string:
                    continue
                
                # Find ALL occurrences, not just the first
                start_pos = 0
                while True:
                    apc_start = hit_text_lower.find(apc_string, start_pos)
                    if apc_start == -1:
                        break
                    
                    apc_end = apc_start + len(apc_string)
                    apc_char_start_in_full = hit_char_start_in_full + apc_start
                    apc_char_end_in_full = hit_char_start_in_full + apc_end
                    
                    # Label all tokens overlapping this APC occurrence
                    is_inside_apc = False
                    for t_idx, (c_start, c_end) in enumerate(full_encoding['offset_mapping']):
                        if c_start is None or c_end is None:
                            continue
                        # Check overlap
                        if max(c_start, apc_char_start_in_full) < min(c_end, apc_char_end_in_full):
                            if not is_inside_apc:
                                full_sequence_labels[t_idx] = label2id["B-APC"]
                                is_inside_apc = True
                            else:
                                full_sequence_labels[t_idx] = label2id["I-APC"] if label_scheme == 'bio' else label2id["I-APC"]
                        elif is_inside_apc and label_scheme == 'bilou':
                            # Convert previous I-APC to L-APC
                            if t_idx > 0 and full_sequence_labels[t_idx-1] in [label2id.get("I-APC"), label2id.get("B-APC")]:
                                full_sequence_labels[t_idx-1] = label2id["L-APC"]
                            is_inside_apc = False
                    
                    start_pos = apc_start + 1  # Continue searching after this occurrence

        # Create chunks
        chunks = []
        token_start = 0
        
        while token_start < len(full_encoding['input_ids']):
            token_end = min(token_start + max_length, len(full_encoding['input_ids']))
            
            # Extract chunk
            chunk_ids = full_encoding['input_ids'][token_start:token_end]
            chunk_offsets = full_encoding['offset_mapping'][token_start:token_end]
            
            # Pad to max_length
            padding_length = max_length - len(chunk_ids)
            padded_input_ids = chunk_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(chunk_ids) + [0] * padding_length
            token_type_ids = [0] * max_length
            padded_offset_mapping = chunk_offsets + [(0, 0)] * padding_length
            
            # Handle labels for this chunk
            if full_sequence_labels is not None:
                chunk_labels = full_sequence_labels[token_start:token_end]
                chunk_labels = chunk_labels + [-100] * padding_length
            else:
                chunk_labels = [-100] * max_length
            
            # Find hit token range in this chunk
            hit_start_token_idx_in_chunk = 0
            hit_end_token_idx_in_chunk = 0
            
            for j, (c_start, c_end) in enumerate(chunk_offsets):
                if c_start is None or c_end is None:
                    continue
                if max(c_start, hit_char_start_in_full) < min(c_end, hit_char_end_in_full):
                    if hit_start_token_idx_in_chunk == 0:
                        hit_start_token_idx_in_chunk = j
                    hit_end_token_idx_in_chunk = j + 1
            
            # Create chunk dictionary - CRITICAL for debugging: all values must be native Python types
            chunk_dict = {
                'input_ids': padded_input_ids,                  # Python list
                'attention_mask': attention_mask,               # Python list
                'token_type_ids': token_type_ids,               # Python list
                'offset_mapping': padded_offset_mapping,        # Python list of tuples
                'labels': chunk_labels,                         # Python list
                'hit_token_ranges_in_chunk': (hit_start_token_idx_in_chunk, hit_end_token_idx_in_chunk),  # Python tuple
                'hit_char_ranges_in_full_text': (hit_char_start_in_full, hit_char_end_in_full),  # Python tuple
                'original_text_full': full_text,                # Python string
                'original_idx': int(original_example_idx),      # Python int (not numpy)
                'original_instance': int(original_instance)     # Python int (not numpy)
            }
            
            chunks.append(chunk_dict)
            
            # Move to next chunk with overlap
            token_start += (max_length - overlap)
        
        return chunks



    @timeit                 
    def tokenize_and_split(self, test_size:float=None, val_size:float=None, 
                                random_state:int=None, num_proc:int=None,
                                max_length=None, overlap_size=None, cache_dir=None):
        """
        Tokenize and split method specifically for training data.
        Now much simpler thanks to flat dataset structure.
        """
        
                # Use default values from config if none are provided
        if test_size is None:
            try:
                test_size = self.config["test_size"]
            except KeyError:
                raise ValueError("Missing 'test_size' key in the configuration file.")
        
        if val_size is None:
            try:
                val_size = self.config["val_size"]
            except KeyError:
                raise ValueError("Missing 'val_size' key in the configuration file.")
        
        if random_state is None:
            try:
                random_state = self.config["random_state"]
            except KeyError:
                raise ValueError("Missing 'random_state' key in the configuration file.")
        
        if num_proc is None:
            try:
                num_proc = self.config["num_proc"]
            except KeyError:
                raise ValueError("Missing 'num_proc' key in the configuration file.")
        
        if max_length is None:
            try:
                max_length = self.config["max_length"]
            except KeyError:
                raise ValueError("Missing 'max_length' key in the configuration file.")
        
        if overlap_size is None:
            try:
                overlap_size = self.config["overlap_size"]
            except KeyError:
                raise ValueError("Missing 'overlap_size' key in the configuration file.")
        

        # Extract unique original examples for stratified splitting
        print("Converting dataset to DataFrame for stratified splitting...")        
        print(f"Found {len(self.df.index)} unique original examples")
        print(f"Instance distribution: {self.df['instance'].value_counts().to_dict()}")
        
        # Stratified splits on unique original examples
        train_val_df, test_df = train_test_split( 
            self.df, 
            test_size=test_size,  
            stratify=self.df["instance"],
            random_state=random_state 
        ) 
        
        train_df, val_df = train_test_split( 
            train_val_df, 
            test_size=val_size / (1 - test_size), 
            stratify=train_val_df["instance"],  
            random_state=random_state 
        )         
        
        print(f"Split original examples - Train: {len(train_df.index)}, Val: {len(val_df.index)}, Test: {len(test_df.index)}")

        print("Converting the splits back to datasets and store in DatasetDict")
        
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Combine into a DatasetDict
        self.dataset = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        # tokenize
        print(f'Running tokenization and {self.config["label_type"]} labeling')
        
        tokenized_splits = DatasetDict()
        for split_name, dataset_obj in self.dataset.items():
            print(f"Tokenizing and labeling {split_name} split...")
            # Call tokenize_dataset() on the individual dataset_obj
            tokenized_splits[split_name] = self.tokenize_dataset(dataset_obj,num_proc=num_proc,
                                                                 max_length=max_length, 
                                                                 overlap=overlap_size, 
                                                                 cache_dir=cache_dir 
                                                                 ) 


        
        # save tokenised dataset in object
        self.tokenized_dataset = tokenized_splits
        # save untokenised dataframes for later reference
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        print('Finished tokenization')
        print(f"Final chunk counts - Train: {len(self.tokenized_dataset['train'])}, Val: {len(self.tokenized_dataset['validation'])}, Test: {len(self.tokenized_dataset['test'])}")
        
        return self.tokenized_dataset['train'], self.tokenized_dataset['validation'], self.tokenized_dataset['test']

    #############################################
    # MODEL TRAINING
    def setup_trainer(self):
        """
        Set up the trainer

        Arguments are defined in the config file (config.yaml by default)
        """
 
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenised_dataset['train'],
            eval_dataset=self.tokenised_dataset['validation'],
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[LogToHubCallback()]
        )
        
    @timeit
    def run_trainer(self):
        """Run fine-tuning
        """
        print("Running training on dataset")
        print("Training arguments:")
        print(self.training_arguments)
        
        self.trainer.train()
    
    
    ############################
    # MODEL EVALUATION
    def evaluate_model(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        print("Running evaluation of test dataset")
        eval_trainer = Trainer(
            model=self.model,
            args=self.training_args, # You might create new args for evaluation or re-use parts
            eval_dataset=self.tokenized_dataset['test'],
            compute_metrics=self.compute_metrics
        )
        
        results = eval_trainer.evaluate()
        self.log_evaluation_results(results)
        
        return results


    def log_evaluation_results(self, results):
        """
        Logs the evaluation results to a file.

        Args:
            results (dict): Dictionary containing evaluation metrics.
        """
        # Define the path for the log file
        log_file_path = os.path.join("evaluation_logs", "evaluation_log.txt")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # Append the results to the log file
        with open(log_file_path, 'a') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Timestamp: {timestamp}\n")
            for metric_name, value in results.items():
                f.write(f"{metric_name}: {value}\n")
            
            # Log trainer settings
            f.write("\nTrainer Arguments:\n")
            for arg_name, arg_value in self.training_args.to_dict().items():
                f.write(f"{arg_name}: {arg_value}\n")
            
            f.write("\n")

        print(f"Evaluation results logged to {log_file_path}")
    
    def compute_metrics(self, eval_preds):
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
        accuracy_result = self.accuracy_metric.compute(predictions=true_predictions, references=true_labels)
        f1_result = self.f1_metric.compute(predictions=true_predictions, references=true_labels, average="micro") # Or "macro", "weighted"
        precision_result = self.precision_metric.compute(predictions=true_predictions, references=true_labels, average="micro")
        recall_result = self.recall_metric.compute(predictions=true_predictions, references=true_labels, average="micro")

        # Combine results into a single dictionary
        results.update(accuracy_result) # Adds {'accuracy': value}
        results.update(f1_result)      # Adds {'f1': value}
        results.update(precision_result) # Adds {'precision': value}
        results.update(recall_result)    # Adds {'recall': value}

        return results
    
    
    def get_tokenized_dataset(self):
        """
        Return the tokenized dataset
        """
        if hasattr(self, 'tokenized_dataset'):
            return self.tokenized_dataset
        else:
            raise ValueError("No tokenized dataset available. Run tokenize_dataset() first.")
        
    def save_datasets(self, path):
        """Save processed datasets to disk"""
        if self.tokenized_dataset is None:
            raise ValueError("No datasets to save. Process data first.")
        
        self.tokenized_dataset.save_to_disk(path)
        print(f"Datasets saved to {path}")

    def load_datasets(self, path):
        """Load processed datasets from disk"""
        self.tokenized_dataset = DatasetDict.load_from_disk(path)
        print(f"Datasets loaded from {path}")
        return self.tokenized_dataset

    def get_datasets(self):
        """Return the DatasetDict"""
        return self.tokenized_dataset
    
    def get_train_dataset(self):
        """Return just the training dataset"""
        return self.tokenized_dataset['train'] if self.tokenized_dataset else None
    
    def get_val_dataset(self):
        """Return just the validation dataset"""
        return self.tokenized_dataset['validation'] if self.tokenized_dataset else None
    
    def get_test_dataset(self):
        """Return just the test dataset"""
        return self.tokenized_dataset['test'] if self.tokenized_dataset else None
    
    

    

