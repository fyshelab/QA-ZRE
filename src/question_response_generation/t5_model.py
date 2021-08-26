"""Implementation of the T5 Model for Response Generation.

Some parts are from huggingface Library.
https://github.com/huggingface/transformers/tree/master/src/transformers/models/t5
https://arxiv.org/pdf/1910.10683.pdf
https://arxiv.org/abs/1804.04235
"""

import gc
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Optional

import numpy
import torch
from transformers import Adafactor, T5ForConditionalGeneration, T5Tokenizer


@dataclass
class HyperParameters:
    """General Model configuration."""

    model_path: Optional[str] = None
    batch_size: int = 64
    source_max_length: int = 512
    decoder_max_length: int = 128
    config_file: str = "config.ini"
    dim_embedding: int = 100
    dim_model: int = 128
    dropout: float = 0.5
    gpu: bool = False
    l2_norm_weight: float = 0.01
    learning_rate: float = 0.0005
    max_epochs: int = 16
    mode: str = "train"
    train: Optional[str] = None
    prediction_file: Optional[str] = None
    seed: int = 8
    test: Optional[str] = None
    dev: Optional[str] = None
    checkpoint: Optional[str] = "_3_model"

    # Related to beam search decoding.
    beam_decoding: Optional[bool] = False
    num_beams: Optional[int] = 5
    no_repeat_ngram_size: Optional[int] = 2
    early_stopping: Optional[bool] = True


def set_random_seed(seed: int) -> Any:
    """Set the random seed, which initializes the random number generator.

    Ensures that runs are reproducible and eliminates differences due to
    randomness.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save(model: torch.nn.Module, path: str) -> None:
    """Save the model to task at the specified path."""
    torch.save(model.state_dict(), path)


MODEL_NAME = "t5-base"
#MODEL_NAME = "t5-base"
#MODEL_NAME = "iarfmoose/t5-base-question-generator"
#MODEL_NAME = "mrm8488/t5-base-finetuned-question-generation-ap"

class T5QA(object):
    """Wrapper class around the T5 Model."""

    def __init__(self, cfg: HyperParameters):
        self.config = cfg

        set_random_seed(cfg.seed)

        # Check the gpu actually exists.
        cfg.gpu = cfg.gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if cfg.gpu else "cpu")

        if cfg.mode == "train":
            tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

            # Construct model
            model = torch.nn.DataParallel(
                T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
            )
            model.to(self.device)

            # Configurations suggested by the paper.
            self.optimizer = Adafactor(
                model.parameters(),
                lr=cfg.learning_rate,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
            if not os.path.exists(cfg.model_path):
                os.makedirs(cfg.model_path)
            self.model_path = os.path.join(cfg.model_path, "model")

        elif cfg.mode in ["test", "inference"]:
            tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
            model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
            model.to(self.device)

            self.model_path = os.path.join(cfg.model_path, "model")
            loaded_weights = torch.load(
                self.model_path + cfg.checkpoint,
                map_location=lambda storage, loc: storage,
            )
            new_weights = {}
            for name, param in loaded_weights.items():
                new_weights[self.remove_prefix(name, "module.")] = param

            model.load_state_dict(new_weights)

        self.model = model
        self.tokenizer = tokenizer

    def remove_prefix(self, text, prefix):
        if text.startswith(prefix):
            return text[len(prefix) :]
        return text

    def save(self, checkpoint_name: str):
        """Save the encoder model to the specified path name."""
        path = self.model_path + "_" + checkpoint_name
        save(self.model, path + "_model")

    def predict(self, batch):
        # Free memory in GPU, very important!
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        # disable dropout
        self.model.eval()

        input_ids = batch["input_ids"]
        input_mask = batch["attention_mask"]
        if self.config.gpu:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)

        if not self.config.beam_decoding:
            predictions = self.model.generate(
                input_ids=input_ids,
                attention_mask=input_mask,
            )

            # all special tokens including will be removed
            predictions_str = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
        else:
            predictions = self.model.generate(
                input_ids=input_ids,
                attention_mask=input_mask,
                num_beams=self.config.num_beams,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                early_stopping=self.config.early_stopping,
                max_length=self.config.decoder_max_length,
                num_return_sequences=1,
            )
            # all special tokens including will be removed
            predictions_str = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )

        input_str = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        for index in range(len(predictions_str)):
            pred_str = predictions_str[index]
            pred_str = pred_str if pred_str != "" else "<EMPTY>"
            output_batch = {
                "predictions_str": pred_str,
                "input_str": input_str[index],
            }
            yield output_batch

    def train(self, batch):
        # Free memory in GPU, very important!
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        # Turn on training mode which enables dropout.
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"]
        input_mask = batch["attention_mask"]
        target_mask = batch["target_attention_mask"]
        labels = batch["labels"]
        if self.config.gpu:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            target_mask = target_mask.to(self.device)
            labels = labels.to(self.device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=input_mask,
            decoder_attention_mask=target_mask,
            labels=labels,
        )

        # mean loss from multiple GPUs
        loss = output.loss.mean()
        loss_value = loss.item()

        # is loss nan? don't backpropagate!
        if math.isnan(loss):
            return {"loss_value": loss_value}

        # BackProp
        loss.backward()

        # Optimize
        self.optimizer.step()

        return {"loss_value": loss_value}
