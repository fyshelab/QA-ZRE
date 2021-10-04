"""Implementation of the T5 Model for Response and Question Generation.

(DataParallel Mode)
"""

import math
import os

import torch
from transformers import Adafactor, T5ForConditionalGeneration, T5Tokenizer

from src.re_qa_model import (HyperParameters, clear_cache, load_module,
                             set_random_seed)


def save(model: torch.nn.Module, path: str) -> None:
    """Save the model to task at the specified path."""
    torch.save(model.state_dict(), path)


MODEL_NAME = "t5-base"


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

            loaded_weights = torch.load(
                self.model_path + cfg.checkpoint,
                map_location=lambda storage, loc: storage,
            )
            model.load_state_dict(loaded_weights)

        elif cfg.mode in ["test", "inference"]:
            tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
            model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
            model.to(self.device)

            self.model_path = os.path.join(cfg.model_path, "model")
            load_module(model, self.model_path, cfg.checkpoint)

        self.model = model
        self.tokenizer = tokenizer

    def save(self, checkpoint_name: str):
        """Save the encoder model to the specified path name."""
        path = self.model_path + "_" + checkpoint_name
        save(self.model, path + "_model")

    def predict(self, batch):
        clear_cache()
        # disable dropout
        self.model.eval()

        input_ids = batch["input_ids"]
        input_mask = batch["attention_mask"]
        if self.config.gpu:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)

        predictions = self.model.generate(
            input_ids=input_ids,
            attention_mask=input_mask,
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
        clear_cache()
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
