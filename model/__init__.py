import torch
import importlib
def get_model(name,args):
    print('Load model start')
    target_lib = 'model.'+name
    print('Target_lib: ',target_lib)
    model_lib = importlib.import_module(target_lib)
    target_model = "".join(map(str.title,name.split('_')))
    print('Target_model: ',target_model)
    return getattr(model_lib, target_model)(**args)
    