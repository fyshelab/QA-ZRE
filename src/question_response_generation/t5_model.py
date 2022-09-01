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


# MODEL_NAME = "t5-base"


class T5QA(object):
    """Wrapper class around the T5 Model."""

    def __init__(self, cfg: HyperParameters):
        self.config = cfg

        MODEL_NAME = self.config.model_name

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
            load_module(model, self.model_path, cfg.checkpoint)

        self.model = model
        self.tokenizer = tokenizer

    def save(self, checkpoint_name: str):
        """Save the encoder model to the specified path name."""
        path = self.model_path + "_" + checkpoint_name
        save(self.model, path + "_model")

    def relation_extraction_predict(self, batch):
        clear_cache()
        # disable dropout
        self.model.eval()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        if self.config.gpu:
            loss_fct = loss_fct.to(self.device)

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
            decoder_input_ids=self.model._shift_right(labels),
            labels=None,
        )

        log_p = -loss_fct(
            output.logits.view(-1, output.logits.size(-1)),
            labels.view(-1),
        )

        # b: batch size * num_unseen_relations
        # sz: sequence size
        # v: vocab size
        b, sz, v = output.logits.size()
        log_p = log_p.view(b, sz)
        good_log_p = log_p.masked_fill_(labels == -100, 0.0)
        answer_log_p = torch.sum(good_log_p, dim=1).squeeze().cpu().detach().numpy()

        for index in range(b):
            relation_log_p = answer_log_p[index]
            output_batch = {
                "relation_log_p": relation_log_p,
            }
            yield output_batch

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
