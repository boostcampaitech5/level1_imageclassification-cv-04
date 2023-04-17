from .plot import *
from .sampler import *
from .transform import *

import json
def read_json(file_dir):
    with open(file_dir,'r') as f:
        json_data = json.load(f)
    return json_data