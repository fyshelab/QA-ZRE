"""This module defines the required functions to setup the optimizers for each
experiment type with the T5 models."""

from typing import Dict, List, Optional

import torch
from absl import flags
from torch.optim.optimizer import Optimizer
from transformers import Adafactor

FLAGS = flags.FLAGS
flags.DEFINE_float(
    "learning_rate", 0.005, "The learning rate used in the optimizer", lower_bound=0.0
)

OPTIMIZER_ARGS_TYPE = Dict[str, torch.nn.Module]


def construct_optimizer(
    model: torch.nn.Module, second_model: Optional[torch.nn.Module] = None
) -> List[Optimizer]:
    """Define the adafactor optimizers over the parameters."""

    # Configurations suggested by the T5 paper.
    # https://discuss.huggingface.co/t/t5-finetuning-tips/684/35
    # to know more about Adafactor: https://arxiv.org/abs/1804.04235
    # Adafactor has small memory footprint compared to adam in transformers.

    optimizers = []
    optimizer = Adafactor(
        model.parameters(),
        lr=FLAGS.learning_rate,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    optimizers.append(optimizer)

    if second_model is not None:
        # define a separate optimizer over the second model.
        optimizer = Adafactor(
            second_model.parameters(),
            lr=FLAGS.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
        optimizers.append(optimizer)

    return optimizers


def single_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> List[Optimizer]:
    """Define the optimizer that fine-tunes all the weights in a single T5
    model.

    This will be used to define a single optimizer to pre-train or fine-
    tune a single T5 model on QA or Relation Extraction corpora.
    """
    return construct_optimizer(model=opt_args["t5_model"])


def dual_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> List[Optimizer]:
    """Define the optimizers that fine-tune one question generator and one
    response generator on the RE corpora."""
    return construct_optimizer(
        model=opt_args["q_model"], second_model=opt_args["r_model"]
    )


# store the functions that setup the optimizer for each experiment type.
optimizer_definer = {
    "qa": single_opt,
    "qa_zre": dual_opt,
}
