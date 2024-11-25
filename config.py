from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20, 
        "lr": 10**-4, # NOTE: usually give a high lr and reduce during training
        "seq_len": 350,
        "d_model": 512, # NOTE default via paper
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel" # save losses while training
    }


# path for saving weights:
def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path('.') / model_folder / model_filename)
