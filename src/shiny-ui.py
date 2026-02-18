# Libraries
from shiny import App, render, ui, reactive
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification, AutoTokenizer
import pandas as pd
from io import StringIO

# custom imports
import auxiliary as aux

# NOTE: Shiny doesn't have built-in caching like Streamlit's @st.cache_resource
# You'll need to implement caching manually using functools.lru_cache or similar
# Or load models once at startup (shown below)

#
# Model and Trainer Setup (loaded once at app startup)
#

models = [
    {"label": "german", 
     "language": "german",
     "display": "German (BERTfull_GER-APC)", 
     "path": "../models/BERTfull_GER-APC"},
    {"label": "english",
     "language": "english", 
     "display": "English (BERT-EN-v1)", 
     "path": "../models/BERT-EN-3epochs_v1"},
    {"label": "english",
     "language": "english", 
     "display": "English (BERT-EN-manualfixes)", 
     "path": "../models/BERT-EN-manualfixes_v1"}
]

# Cache for loaded models (simple dict-based cache)
_model_cache = {}

def load_model(model_path):
    if model_path not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        _model_cache[model_path] = (tokenizer, model)
    return _model_cache[model_path]

def setup_trainer(model_path):
    tokenizer, model = load_model(model_path)
    inference_args = TrainingArguments(
        output_dir="./inference_results",
        per_device_eval_batch_size=16,
        do_train=False,
        do_eval=False,
        do_predict=True,
    )

    trainer = Trainer(
        model=model,
        args=inference_args,
        processing_class=tokenizer
    )
    
    return trainer, tokenizer

def get_data(string, trainer, tokenizer, language='german', inclprons=True, num_proc=4):
    return aux.string_in_apc_df_out(string, trainer, tokenizer, language=language, inclprons=inclprons, num_proc=num_proc)

#
# UI Definition
#

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select(
            "model_selector",
            "Choose model language",
            choices={m["path"]: m["display"] for m in models},
            selected=models[0]["path"]
        ),
        ui.input_radio_buttons(
            "input_choice",
            "How would you like to submit your input",
            choices={
                "free_text": "Free text",
                "upload": "Upload text file"
            },
            selected="free_text"
        ),
        ui.input_radio_buttons(
            "output_choice",
            "Output options",
            choices={
                "apc_only": "Only APCs",
                "apc_pronouns": "APCs + pronouns"
            },
            selected="apc_only"
        ),
        ui.input_action_button(
            "find_apcs",
            "Find APCs",
            class_="btn-primary",
            icon=ui.tags.i(class_="fa fa-search")
        ),
        ui.input_action_button(
            "reset",
            "Reset",
            class_="btn-secondary",
            icon=ui.tags.i(class_="fa fa-trash")
        ),
        width=300
    ),
    ui.panel_title("Welcome to the APC detector"),
    ui.markdown(
        """'What is an APC' you might ask? The term is used here as shorthand for **adnominal pronoun construction** 
        as in English *you academics* or *we/us geniuses*. The current version of the detector supports detection of 
        APCs in English and in German (*wir Linguisten*)."""
    ),
    ui.panel_conditional(
        "input.input_choice === 'free_text'",
        ui.input_text_area(
            "text_input",
            "Enter the text you want to check for APCs",
            height="300px",
            width="100%"
        )
    ),
    ui.panel_conditional(
        "input.input_choice === 'upload'",
        ui.input_file(
            "file_upload",
            "Choose a file",
            accept=[".txt"],
            multiple=False
        ),
        ui.output_text_verbatim("file_preview")
    ),
    ui.output_ui("results_section"),
    # NOTE: Shiny's page layout doesn't have a direct equivalent to Streamlit's layout="wide"
    # You can use custom CSS to achieve wider layouts if needed
    title="APC Detector"
)

#
# Server Logic
#

def server(input, output, session):
    # Reactive values to store state
    results_df = reactive.Value(pd.DataFrame())
    processing_status = reactive.Value("")
    
    # Get the current model info based on selection
    @reactive.Calc
    def current_model():
        selected_path = input.model_selector()
        return next(m for m in models if m["path"] == selected_path)
    
    # Get input text based on input method
    @reactive.Calc
    def input_text():
        if input.input_choice() == "free_text":
            return input.text_input()
        elif input.input_choice() == "upload":
            file_info = input.file_upload()
            if file_info is not None and len(file_info) > 0:
                file_path = file_info[0]["datapath"]
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        return ""
    
    # Preview uploaded file
    @output
    @render.text
    def file_preview():
        text = input_text()
        if input.input_choice() == "upload" and text:
            # Limit preview length
            preview = text[:500] + "..." if len(text) > 500 else text
            return preview
        return ""
    
    # Handle Find APCs button
    @reactive.Effect
    @reactive.event(input.find_apcs)
    def _():
        text = input_text()
        
        if not text or text.strip() == "":
            # NOTE: Shiny doesn't have direct equivalents to st.error/st.toast
            # You'll need to implement notifications using ui.notification_show or custom UI
            ui.notification_show("Please provide some text to analyze", type="error")
            return
        
        try:
            processing_status.set("Processing...")
            
            model_info = current_model()
            inferencer, tokenizer = setup_trainer(model_info["path"])
            
            output_includes_pronouns = input.output_choice() == "apc_pronouns"
            
            returndf = get_data(
                text, 
                inferencer, 
                tokenizer, 
                language=model_info["language"], 
                inclprons=output_includes_pronouns, 
                num_proc=1
            )
            
            if returndf.empty:
                ui.notification_show("No relevant constructions found", type="warning")
                results_df.set(pd.DataFrame())
            else:
                # Concatenate with existing results
                current_df = results_df.get()
                new_df = pd.concat([current_df, returndf], ignore_index=True)
                results_df.set(new_df)
                ui.notification_show("Analysis complete!", type="message")
            
            processing_status.set("")
            
        except Exception as e:
            ui.notification_show(f"Error during processing: {str(e)}", type="error")
            processing_status.set("")
    
    # Handle Reset button
    @reactive.Effect
    @reactive.event(input.reset)
    def _():
        results_df.set(pd.DataFrame())
        processing_status.set("")
        # NOTE: Shiny doesn't have st.rerun() equivalent
        # UI will automatically update based on reactive values
    
    # Render results section
    @output
    @render.ui
    def results_section():
        df = results_df.get()
        
        if df.empty:
            if processing_status.get():
                return ui.div(
                    ui.p(processing_status.get()),
                    class_="alert alert-info"
                )
            return ui.div()
        
        return ui.div(
            ui.h3("Results"),
            ui.output_data_frame("results_table"),
            ui.download_button(
                "download_csv",
                "Download results as csv",
                icon=ui.tags.i(class_="fa fa-download")
            )
        )
    
    # Render results table
    @output
    @render.data_frame
    def results_table():
        return results_df.get()
    
    # Handle download
    @session.download(filename="APC-results.csv")
    def download_csv():
        df = results_df.get()
        # NOTE: Shiny's download handler expects a file path or a generator
        # This returns the CSV as a string
        return df.to_csv(index=False)

#
# Create App
#

app = App(app_ui, server)

# To run: shiny run app.py