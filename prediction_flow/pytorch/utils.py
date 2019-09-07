import torch.nn as nn
import torch.nn.init as init


def init_weights(model):
    if isinstance(model, nn.Linear):
        if model.weight is not None:
            init.kaiming_uniform_(model.weight.data)
        if model.bias is not None:
            init.normal_(model.bias.data)
    elif isinstance(model, nn.BatchNorm1d):
        if model.weight is not None:
            init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm2d):
        if model.weight is not None:
            init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm3d):
        if model.weight is not None:
            init.normal_(model.weight.data, mean=1, std=0.02)
        if model.bias is not None:
            init.constant_(model.bias.data, 0)
    else:
        pass
