import torch
import importlib
def get_model(module_name,args):
    print('Load model start')
    target_lib = 'model.'+module_name
    print('Target_lib: ',target_lib)
    model_lib = importlib.import_module(target_lib)
    target_model = "".join(map(str.title,module_name.split('_')))
    print('Target_model: ',target_model)
    return getattr(model_lib, target_model)(**args)
    