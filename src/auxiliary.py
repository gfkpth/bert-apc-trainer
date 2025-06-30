
#%%
import pandas as pd
import numpy as np

import multiprocessing as mp
from functools import partial

from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer
import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# %%

# class for data input and pre-processing
class DataPrep:
    def __init__(self, language="german"):
        self.df = None
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
        self.language=language
        self.textcols = ['ContextBefore', 'Hit', 'ContextAfter','APC']

    # loading the csv
    def load_csv(self, path):
        self.df = pd.read_csv(path,index_col='ID', dtype={'instance': int})
    
    # allowing data output
    def get_df(self,subset=None):
        """
        Return full dataframe by default
        
        arguments:
        **subset**: set to 'instance' to get only rows with instance==1 
        """
        if subset == 'instance':
            return self.df[self.df.instance==1]
        else:
            return self.df

    
    def find_duplicate_hits(self):
        """ 
        Detect duplicate sentences that might need merging
        
        Returns: list of tuples of indices for each group of duplicates
        """
        # Step 1: Filter by instance == 1
        step1 = self.df[self.df.instance == 1]
        
        # Step 2: Find actual duplicates in Hit column
        step2 = step1[step1.duplicated(subset=['Hit'], keep=False)]
        
        # Debug: Show what's being grouped
        groups = step2.sort_values('Hit').groupby('Hit').groups
        # for hit_value, indices in groups.items():
        #     print(f"Hit value '{hit_value}' appears at indices: {tuple(indices)}")
        #     # Verify they actually have the same Hit value
        #     print(f"Actual Hit values: {step2.loc[indices, 'Hit'].unique()}")
        #     print("---")
        print(f'Found {len(groups)} groups of duplicate rows')
        
        # Group and return
        return [tuple(indices) for indices in step2.sort_values('Hit').groupby('Hit').groups.values()]



    def merge_apc_annotations(self):
        """
        Merge BIO annotations and rows with multiple instances of APCs in "Hit"
        """
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
            
            # write consolidated biolabel to first instance
            self.df.at[t[0],'biolabels'] = consolidated
        
        print(f"Processed {processed} duplicate rows")
        
        # cleanup
        print("Removing now-redundant rows")
        self.df = self.df.drop_duplicates(subset='Hit',keep='first')
            
    
    def df_index(self,index):
        return self.df.loc[index]
                
    
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

    
    def train_test_split_ds(self, tokenizer, test_size=0.2):
        """
        Create train-test-split, save dataframe in object and return huggingface datasets
        """        
        self.tokenize_and_align_labels(tokenizer)

        # Convert tensor-based BatchEncoding to Python-native dicts
        self.df["tokenized_clean"] = self.df["tokenized"].apply(
            lambda x: {k: np.array(v).squeeze().tolist() for k, v in x.items()}
        )

        # Train/test split
        self.train_df, self.val_df = train_test_split(
            self.df, test_size=test_size, stratify=self.df["instance"]
        )

        train_dataset = Dataset.from_list(self.train_df["tokenized_clean"].tolist())
        val_dataset = Dataset.from_list(self.val_df["tokenized_clean"].tolist())

        return train_dataset, val_dataset
    
    def tokenize_single_row(row_data, tokenizer, labeltoint):
        """Process a single row - designed for multiprocessing"""
        idx, row = row_data
        
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
            max_length=64,
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
                    if word_idx == previous_word_idx:
                        if isinstance(label, str) and label.startswith('B-'):
                            aligned_labels.append(labeltoint[label.replace('B-', 'I-')])
                        elif isinstance(label, str):
                            aligned_labels.append(labeltoint[label])
                        else:
                            aligned_labels.append(label)
                    else:
                        if isinstance(label, str):
                            aligned_labels.append(labeltoint[label])
                        else:
                            aligned_labels.append(label)
            
            previous_word_idx = word_idx

        # Convert to clean format immediately
        result = {
            'input_ids': tokenized['input_ids'].squeeze().tolist(),
            'attention_mask': tokenized['attention_mask'].squeeze().tolist(),
            'labels': aligned_labels
        }
        
        return idx, result

    def tokenize_and_align_labels_mp(self, tokenizer, n_cores=None):
        """Multiprocessing version of tokenize_and_align_labels"""
        if n_cores is None:
            n_cores = mp.cpu_count() - 1  # Leave one core free
        
        print(f"Using {n_cores} cores for tokenization")
        
        # Prepare data for multiprocessing
        row_data = [(idx, row) for idx, row in self.df.iterrows()]
        
        # Create partial function with fixed arguments
        tokenize_func = partial(
            self.tokenize_single_row, 
            tokenizer=tokenizer, 
            labeltoint=self.labeltoint
        )
        
        # Process in parallel
        with mp.Pool(n_cores) as pool:
            results = pool.map(tokenize_func, row_data)
        
        # Reconstruct the results in original order
        tokenized_dict = {idx: result for idx, result in results}
        tokenized_inputs = [tokenized_dict[idx] for idx in self.df.index]
        
        return tokenized_inputs

    def train_test_split_ds_mp(self, tokenizer, test_size=0.2, n_cores=None):
        """Multiprocessing version of split_to_dataset"""
        # Use multiprocessing tokenization
        tokenized_inputs = self.tokenize_and_align_labels_mp(tokenizer, n_cores)
        
        # Create temporary dataframe with clean tokenized data
        import pandas as pd
        temp_df = self.df.copy()
        temp_df["tokenized_clean"] = tokenized_inputs
        
        # Train/test split
        train_df, val_df = train_test_split(
            temp_df, test_size=test_size, stratify=temp_df["instance"], random_state=42
        )
        
        train_dataset = Dataset.from_list(train_df["tokenized_clean"].tolist())
        val_dataset = Dataset.from_list(val_df["tokenized_clean"].tolist())
        
        return train_dataset, val_dataset




def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }