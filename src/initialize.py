"""Initialize weights and biases for Neural Networks."""

import torch.nn as nn


def rnn_param_init(module: nn.Module) -> None:
    """Use orthogonal init for recurrent layers, xavier uniform for input
    layers Bias is 0 except for forget gate."""
    for name, param in module.named_parameters():
        if "weight_hh" in name:
            nn.init.orthogonal_(param.data)
        elif "weight_ih" in name:
            nn.init.xavier_uniform_(param.data)
        elif "bias" in name:
            nn.init.zeros_(param.data)


def xavier_param_init(module: nn.Module) -> None:
    """for initializing weights with xavier initializer."""

    for name, param in module.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0.0)

        if "weight" in name:
            nn.init.xavier_uniform_(param)
