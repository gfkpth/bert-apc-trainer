
# Libraries
import streamlit as st
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, AutoModelForTokenClassification, AutoTokenizer
import pandas as pd

# custom imports
import auxiliary as aux

model_path = '../models/BERTfixedtrain_3epochs'


# cache model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tunedmodel = AutoModelForTokenClassification.from_pretrained(model_path)
    return tokenizer, tunedmodel

@st.cache_data
def get_data():
    df = pd.DataFrame(
    )
    return df

@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")

tokenizer, tunedmodel = load_model()


# Sidebar
with st.sidebar:
    inputchoice = st.radio('How would you like to submit your input',options=['Free text', 'Upload text file'], captions=['Enter your text in the text field', 'Select a text file for analysis'] )
    outputchoice = st.radio('Output options', options=['Only APCs', 'APCs + pronouns'], captions=['List only sentences identified as containing APCs', 'List sentences with detected APCs and personal pronouns'])
    st.button('Find APCs')
# Main window

st.title('Welcome to the APC detector')

# Provide a short description
st.write("""'What is an APC' you might ask? The term is used here as shorthand for *adnominal pronoun construction* as in English 'you academics' or 'we/us geniuses'. 
         The current version of the detector model supports only German APCs like *wir Linguisten*. An extension to English might follow at a later point.""")

if inputchoice == 'Free text':
    st.text_area('Enter the text you want to check for APCs')
elif inputchoice == 'Upload text file':
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        