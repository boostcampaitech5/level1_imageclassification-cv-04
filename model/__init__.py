import torch
import importlib
def get_module(module_name):
    print('Load model start')
    target_lib = 'model.'+module_name
    print('Target_lib: ',target_lib)
    model_lib = importlib.import_module(target_lib)
    target_model = "".join(map(str.title,module_name.split('_')))
    print('Target_model: ',target_model)
    