
# Libraries
import streamlit as st
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification, AutoTokenizer
import pandas as pd
from io import StringIO

# custom imports
import auxiliary as aux

#
# local helper functions
#

# cache model and tokenizer
@st.cache_resource
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return tokenizer, model

# create and cache trainer
# create and cache trainer - KEY FIX: Include model_path as a parameter for cache key
@st.cache_resource
def setup_trainer(model_path):
    tokenizer, model = load_model(model_path)
    inference_args = TrainingArguments(
        output_dir="./inference_results", # Required, but won't save much for predict
        per_device_eval_batch_size=16,
        do_train=False,
        do_eval=False,
        do_predict=True,
        # Consider fp16=True if your model supports it for faster inference on GPU
        # no_cuda_empty_cache=True, # Often helpful
    )

    trainer = Trainer(
        model=model,
        args=inference_args,
        processing_class=tokenizer
        # train_dataset=None, # No training
        # eval_dataset=None,  # No evaluation
        # compute_metrics=None # No metrics computation during predict, if not needed
    )
    
    return trainer, tokenizer

# run inference and cache output dataframe
def get_data(string, _trainer, _tokenizer, language='german', inclprons=True, num_proc=4):
    return aux.string_in_apc_df_out(string, _trainer, _tokenizer, language=language, inclprons=inclprons, num_proc=num_proc)

# convert dataframe to csv and cache
@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")

def reset():
    st.session_state.preinference = True
    st.session_state.df_full = pd.DataFrame()
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = "german"
        
#
# Setting up tools
# 

# Initialize session state
if 'preinference' not in st.session_state:
    reset()
    

# setting up tokenizer, model and inferencer
models = [
    {"label": "german", 
     "language": "german",
     "display": "German (BERTfull_GER-APC)", 
     "path": "../models/BERTfull_GER-APC"},
    {"label": "english",
     "language": "english", 
     "display": "English (BERT-EN-3epochs)", 
     "path": "../models/BERT-EN-3epochs"}
]

# Load the selected model based on dropdown
# selected_model_path = model_paths[st.session_state.model_choice]
# tokenizer, model = load_model(selected_model_path)
# inferencer = setup_trainer(model, tokenizer)


#
# Setting up page
#

st.set_page_config(page_title="APC detector", layout="wide")

#
# Content and functionality
#

# Main window - Define input_string FIRST
st.title('Welcome to the APC detector')

st.markdown("""'What is an APC' you might ask? The term is used here as shorthand for **adnominal pronoun construction** as in English *you academics* or *we/us geniuses*. 
         The current version of the detector supports detection of APCs in English and in German (*wir Linguisten*).""")

# Get input choice from sidebar first
#st.session_state.model_choice = st.sidebar.selectbox("Choose model language", options=["German", "English"]).lower()
#st.session_state.model_choice = st.sidebar.selectbox("Choose model language", options=["german", "english"], index=0 if st.session_state.get('model_choice', 'german') == 'german' else 1)
# Just use the key parameter and let Streamlit handle state
selected_model = st.sidebar.selectbox(
    "Choose model language",
    options=models,
    format_func=lambda x: x["display"],  # Show the display name
    key="model_selector"
)

st.session_state.model_choice = selected_model["label"]  # Use the short label

inputchoice = st.sidebar.radio('How would you like to submit your input',
                       options=['Free text', 'Upload text file'], 
                       captions=['Enter your text in the text field', 'Select a text file for analysis'])

# Initialize input_string
input_string = None

# Handle input based on choice
if inputchoice == 'Free text':
    input_string = st.text_area('Enter the text you want to check for APCs',
                                height=300)
elif inputchoice == 'Upload text file':
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # To read file as string:
        input_string = stringio.read()
        st.text_area('Preview of uploaded text:', value=input_string, height=200, disabled=True)

# Sidebar controls
with st.sidebar:
    outputchoice = st.radio('Output options', 
                            options=['Only APCs', 'APCs + pronouns'], 
                            captions=['List only sentences identified as containing APCs', 'List sentences with detected APCs and personal pronouns'])
    
    # Save choice of output
    output_includes_pronouns = outputchoice == 'APCs + pronouns'
        
    if st.button('Find APCs',
              help='Extract APCs from the provided text',
              type='primary',
              icon=":material/search:"):
        
        # Validate input
        if input_string is None or input_string.strip() == "":
            st.error("Please provide some text to analyze")
        else:
            try:
                with st.spinner('Processing input...'):
                    st.session_state.preinference = False
                    
                    # KEY FIX: Load model and trainer based on current selection
                    inferencer, tokenizer = setup_trainer(selected_model["path"])
                    
                    # Reduce num_proc for Streamlit environment
                    returndf = get_data(input_string, inferencer, tokenizer, language=selected_model["language"], inclprons=output_includes_pronouns, num_proc=1)
                    if returndf.empty:
                        st.toast('No relevant constructions found')
                    else:
                        st.session_state.df_full = pd.concat([st.session_state.df_full, returndf], ignore_index=True)
                    st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                st.session_state.preinference = True  # Reset state on error
    
    if st.button('Reset',
              help='Start fresh. ATTENTION: This removes any previous results, so make sure to save the csv if you want to keep them.',
              type='secondary',
              icon=":material/delete:"):
        reset()
        st.rerun()  # Refresh the app to show reset state

# Display results
if not st.session_state.preinference and not st.session_state.df_full.empty:
    st.subheader("Results")
    st.dataframe(st.session_state.df_full)
    
    st.download_button('Download results as csv',
                    convert_for_download(st.session_state.df_full),
                    file_name='APC-results.csv',
                    mime='text/csv',
                    key='download_csv',
                    help='Save the results displayed above as a csv file',
                    icon=":material/download:"
                    )
elif not st.session_state.preinference and st.session_state.df_full.empty:
    st.info("No APCs found in the provided text.")