import torch.optim as opt


def set_optimizer(model_param,optim_name, optim_args):
    target = getattr(opt, optim_name)
    optim = target(model_param, **optim_args)