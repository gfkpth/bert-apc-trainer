
#%%
import pandas as pd

from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer
import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize

# %%

# class for data input and pre-processing
class DataPrep:
    def __init__(self, tokenizer_name="bert-base-german-cased",language="german"):
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
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

    def load_csv(self, path):
        self.df = pd.read_csv(path,index_col='ID', dtype={'instance': int})
        
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


    def find_duplicates(self):
        # Detect duplicate sentences that might need merging
        return self.df[(self.df.instance == 1) & self.df.duplicated(subset=['Hit'], keep=False)]


    def merge_apc_annotations(self):
        # Merge BIO annotations and rows for hits with multiple instances of APCs
        pass
    
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
        

    def generate_biolabels_single(self, sentence_tokens, apc_tokens):
        # Convert APC string spans to token-level BIO labels
        """
        Given tokenized sentence and tokenized APC, return BIO labels for sentence.
        All instances of the same APC in the same sentence are marked, i.e. we can capture repetitions appropriately. 
        In line with what the base dataset annotation provides, however, this only works for one specific APC.
        """
        labels = ["O"] * len(sentence_tokens)
        sentence_tokens = [t.lower() for t in sentence_tokens]
        apc_tokens = [t.lower() for t in apc_tokens]

        for i in range(len(sentence_tokens)):
            # Try to match full APC starting at this token
            if sentence_tokens[i:i+len(apc_tokens)] == apc_tokens:
                labels[i] = "B-APC"
                for j in range(1, len(apc_tokens)):
                    labels[i + j] = "I-APC"

        return labels

    def tokenize_and_align_labels(self):
        # Tokenize and align BIO labels with -100 where needed
        self.df['tokenise'] = self.tokenizer()
        pass
    
    def to_dataset(self):
        # for transferring to huggingface dataset
        pass
    
    
    
    #%% trialing
    dat = DataPrep()
    dat.load_csv('../data/copyright/DWDS_APC_main_redux.csv')
    
    # %%
    dat.generate_biolabels_df()
    
    
    # %%
    dat.get_df(subset='instance').tail()