# %%
import os 
from dotenv import load_dotenv
from huggingface_hub import login, whoami


import auxiliary as aux


# setting up environment
load_dotenv()
login(token = os.getenv('HF_TOKEN'))

try:
    user = whoami()
    print(f"Authenticated as: {user['name']}")
except Exception as e:
    print(f"Not authenticated: {e}")

# %% training preparation

def apc_pipeline(commit_message:str, model_config:str, config_file:str = "config.yaml",model_revision:str = None):
    apc_train = aux.APCData(model_config, config_file)
    apc_train.prepare_training_data()

    # save datasets for reuse
    apc_train.save_datasets()
    
    # training
    apc_train.setup_trainer()
    apc_train.run_trainer()
    
    # evaluate and push model
    results = apc_train.evaluate_model()
    apc_train.push_model_to_hub(commit_message, model_revision ,eval_results=results)


def update_model_card(local_model_path:str, model_config:str, config_file:str = "config.yaml"):
        
    apc_reload = aux.APCData.from_pretrained(local_model_path,model_config,config_file)
    apc_reload.load_datasets()

    results= apc_reload.evaluate_model()

    apc_reload.create_model_card(results)


def push_local_model(local_model_path:str, model_config:str, config_file:str = "config.yaml", commit_message:str=None, model_revision:str=None):
        
    apc_reload = aux.APCData.from_pretrained(local_model_path,model_config,config_file)
    apc_reload.load_datasets()

    results = apc_reload.evaluate_model()
    apc_reload.push_model_to_hub(commit_message,revision=model_revision ,eval_results=results)



# %% train and push German bilou model

apc_pipeline("baseline model", model_config="german-bilou")

# %% train and push English bilou model


apc_pipeline("baseline model", model_config="english-bilou")

# %%

push_local_model("../models/bert-apc-detector-ger",model_config="german-bilou",commit_message="updating labels")

push_local_model("../models/bert-apc-detector-en",model_config="english-bilou",commit_message="updating labels")
