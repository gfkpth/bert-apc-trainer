import auxiliary as aux

apc_train = aux.APCData("german-bilou","config.yaml")

apc_train.prepare_training_data()
apc_train.setup_trainer()

# %%
apc_train.run_trainer()

# %%
apc_train.evaluate_model()