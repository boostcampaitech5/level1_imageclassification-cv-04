import json


def load_config(config_path):
    with open(config_path,'r') as f:
        config_data = json.load(f)
    return config_data