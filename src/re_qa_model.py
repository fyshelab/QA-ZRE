"""Implementation of the T5 Models for Response Generation and Question
Generation Used for relation extraction."""

import gc
import os
import random
from dataclasses import dataclass
from typing import Any, Optional

import numpy
import torch
from transformers import Adafactor, T5ForConditionalGeneration, T5Tokenizer


def white_space_fix(text):
    return " ".join(text.split())


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
    answer_checkpoint: Optional[str] = "_3_model"
    question_checkpoint: Optional[str] = "_3_model"
    partition_checkpoint: Optional[str] = "_3_model"
    checkpoint: Optional[str] = "_3_model"

    # Related to beam search decoding.
    beam_decoding: Optional[bool] = False
    num_beams: Optional[int] = 5
    num_beam_groups: Optional[int] = 4
    beam_diversity_penalty: Optional[float] = 0.5
    no_repeat_ngram_size: Optional[int] = 2
    early_stopping: Optional[bool] = True
    num_search_samples: Optional[int] = 8
    question_training_steps: Optional[int] = 5
    answer_training_steps: Optional[int] = 1
    training_steps: Optional[int] = 1
    update_switch_steps: Optional[int] = 10


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.stack(list(tuple_of_tensors), dim=0)


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


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever


def torch_save(model: torch.nn.Module, path: str) -> None:
    """Save the model to task at the specified path."""
    torch.save(model.state_dict(), path)


def save(model, model_path: str, checkpoint_name: str):
    """Save the model to the specified path name."""
    torch_save(model, model_path + "_" + checkpoint_name)


def load_module(model, model_path, checkpoint_name):
    # Load the model from the checkpoint.
    loaded_weights = torch.load(
        model_path + checkpoint_name,
        map_location=lambda storage, loc: storage,
    )
    new_weights = {}
    for key, val in loaded_weights.items():
        new_weights[remove_prefix(key, "module.")] = val
    model.load_state_dict(new_weights)


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def prob_of_sampled_predictions(loss_fct, sample_outputs):
    """Helper function to compute the predictions and the probability of a
    sampled sequence from the output of the generate function in the
    transformers library."""
    # Skip the first pad token generated by the T5 model.
    sampled_predictions = sample_outputs.sequences[:, 1:]
    sampled_scores = sample_outputs.scores
    sampled_scores = tuple_of_tensors_to_tensor(sampled_scores)
    l, n, v = sampled_scores.size()
    log_p = -loss_fct(
        sampled_scores.view(-1, v),
        torch.reshape(torch.transpose(sampled_predictions, 0, 1), (l * n,)),
    ).view(l, n)
    pad_mask = torch.transpose(sampled_predictions, 0, 1) == 0
    good_log_p = log_p.masked_fill_(pad_mask, 0.0)
    log_p = torch.sum(good_log_p, dim=0).squeeze()
    # sampled_p = torch.exp(log_p)
    return sampled_predictions, log_p


MODEL_NAME = "t5-base"
Q_MODEL_NAME = "mrm8488/t5-base-finetuned-question-generation-ap"


class REQA(torch.nn.Module):
    """Wrapper class around the T5 Models."""

    def __init__(self, cfg: HyperParameters):
        super(REQA, self).__init__()
        self.config = cfg

        set_random_seed(cfg.seed)

        self.model_path = os.path.join(cfg.model_path, "model")

        # Answer model
        answer_tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

        # Construct the answer model
        answer_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

        # Question Model
        question_tokenizer = T5Tokenizer.from_pretrained(Q_MODEL_NAME)

        # Construct the Question model
        question_model = T5ForConditionalGeneration.from_pretrained(Q_MODEL_NAME)

        if cfg.mode == "train":
            # Configurations suggested by the T5 paper.
            self.answer_optimizer = Adafactor(
                answer_model.parameters(),
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

            # Configurations suggested by the T5 paper.
            self.question_optimizer = Adafactor(
                question_model.parameters(),
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

            load_module(answer_model, self.model_path, cfg.answer_checkpoint)
            load_module(question_model, self.model_path, cfg.question_checkpoint)

        elif cfg.mode in ["test", "inference"]:
            load_module(answer_model, self.model_path, cfg.answer_checkpoint)
            load_module(question_model, self.model_path, cfg.question_checkpoint)

        self.answer_model = answer_model
        self.answer_tokenizer = answer_tokenizer
        self.question_model = question_model
        self.question_tokenizer = question_tokenizer

    def question_greedy_predict(self, batch, current_device):
        """Greedily generate the questions and prepare inputs for the answer
        module."""
        question_input_ids = batch["entity_relation_passage_input_ids"]
        question_input_mask = batch["entity_relation_passage_attention_mask"]
        if self.config.gpu:
            question_input_ids = question_input_ids.to(current_device)
            question_input_mask = question_input_mask.to(current_device)

        with torch.no_grad():
            question_predictions = self.question_model.generate(
                input_ids=question_input_ids,
                attention_mask=question_input_mask,
            )

        question_predictions_str = self.question_tokenizer.batch_decode(
            question_predictions, skip_special_tokens=True
        )

        question_predictions_str = [
            remove_prefix(pred, "question: ") for pred in question_predictions_str
        ]
        # print(question_predictions_str)
        # print(batch["contexts"])
        new_articles = []
        for i in range(len(batch["passages"])):
            new_article = (
                "question: "
                + question_predictions_str[i]
                + " context: "
                + batch["passages"][i]
                + " </s>"
            )
            new_articles.append(new_article)

        answer_inputs = self.answer_tokenizer(
            new_articles,
            truncation=True,
            padding="max_length",
            max_length=self.config.source_max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        answer_input_ids = answer_inputs.input_ids
        answer_input_mask = answer_inputs.attention_mask
        if self.config.gpu:
            answer_input_ids = answer_input_ids.to(current_device)
            answer_input_mask = answer_input_mask.to(current_device)

        return (
            answer_input_ids,
            answer_input_mask,
            question_input_ids,
            question_input_mask,
        )

    def predict_step(self, batch, current_device):
        # Free memory in GPU, very important!
        clear_cache()
        # disable dropout
        self.answer_model.eval()
        self.question_model.eval()

        # TODO For prediction, maybe we can use beam search and then take the top 1 prediction.
        answer_input_ids, answer_input_mask, _, _ = self.question_greedy_predict(
            batch, current_device
        )

        second_entity_predictions = self.answer_model.generate(
            input_ids=answer_input_ids,
            attention_mask=answer_input_mask,
        )
        second_entity_predictions_str = self.answer_tokenizer.batch_decode(
            second_entity_predictions, skip_special_tokens=True
        )
        # print(second_entity_predictions_str)
        for index in range(len(second_entity_predictions_str)):
            pred_str = second_entity_predictions_str[index]
            output_batch = {
                "predictions_str": pred_str,
            }
            yield output_batch

    def pgg_answer_training(self, batch, current_device):
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        if self.config.gpu:
            loss_fct = loss_fct.to(current_device)

        (
            answer_input_ids,
            answer_input_mask,
            question_input_ids,
            question_input_mask,
        ) = self.question_greedy_predict(batch, current_device)
        target_mask = batch["second_entity_attention_mask"]
        labels = batch["second_entity_labels"]
        if self.config.gpu:
            target_mask = target_mask.to(current_device)
            labels = labels.to(current_device)

        # Answer Computation
        self.answer_model.train()
        output = self.answer_model(
            input_ids=answer_input_ids,
            attention_mask=answer_input_mask,
            decoder_attention_mask=target_mask,
            decoder_input_ids=self.answer_model._shift_right(labels),
            labels=None,
        )

        log_p = -loss_fct(
            output.logits.view(-1, output.logits.size(-1)),
            labels.view(-1),
        )

        b, sz, v = output.logits.size()
        log_p = log_p.view(b, sz)
        good_log_p = log_p.masked_fill_(labels == -100, 0.0)
        answer_log_p = torch.sum(good_log_p, dim=1).squeeze()

        loss = -torch.mean(answer_log_p, dim=0)
        loss_value = loss.item()
        return loss, loss_value

    def mml_question_training(self, batch, current_device, sample_p=0.95):
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        if self.config.gpu:
            loss_fct = loss_fct.to(current_device)

        # Loss from the entity relation examples!
        question_input_ids = batch["entity_relation_passage_input_ids"]
        question_input_mask = batch["entity_relation_passage_attention_mask"]
        if self.config.gpu:
            question_input_ids = question_input_ids.to(current_device)
            question_input_mask = question_input_mask.to(current_device)

        b_sz, _ = question_input_ids.size()

        with torch.no_grad():
            self.question_model.eval()
            # Use top-p sampling to collect samples.
            sampled_question_outputs = self.question_model.generate(
                input_ids=question_input_ids,
                do_sample=True,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                early_stopping=self.config.early_stopping,
                max_length=self.config.decoder_max_length,
                num_return_sequences=self.config.num_search_samples,
                top_p=sample_p,
                output_scores=True,
                return_dict_in_generate=True,
                attention_mask=question_input_mask,
            )

            sampled_questions, _ = prob_of_sampled_predictions(
                loss_fct, sampled_question_outputs
            )

        sampled_question_predictions_str = self.question_tokenizer.batch_decode(
            sampled_questions, skip_special_tokens=True
        )

        sampled_question_predictions_str = [
            remove_prefix(pred, "question: ")
            for pred in sampled_question_predictions_str
        ]
        # print(batch["contexts"])
        # print(beam_question_predictions_str)
        sampled_question_predictions_str_reshaped = [
            sampled_question_predictions_str[
                i
                * (self.config.num_search_samples) : (i + 1)
                * (self.config.num_search_samples)
            ]
            for i in range(b_sz)
        ]

        new_articles = []
        for i in range(b_sz):
            for j in range(self.config.num_search_samples):
                new_article = (
                    "question: "
                    + sampled_question_predictions_str_reshaped[i][j]
                    + " context: "
                    + batch["passages"][i]
                    + " </s>"
                )
                new_articles.append(new_article)

        answer_inputs = self.answer_tokenizer(
            new_articles,
            truncation=True,
            padding="max_length",
            max_length=self.config.source_max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )

        answer_input_ids = answer_inputs.input_ids
        answer_input_mask = answer_inputs.attention_mask
        if self.config.gpu:
            answer_input_ids = answer_input_ids.to(current_device)
            answer_input_mask = answer_input_mask.to(current_device)

        target_mask = batch["second_entity_attention_mask"]
        labels = batch["second_entity_labels"]
        if self.config.gpu:
            target_mask = target_mask.to(current_device)
            labels = labels.to(current_device)

        b_sz, seq_len = labels.size()
        labels = labels.repeat(1, self.config.num_search_samples).view(-1, seq_len)
        target_mask = target_mask.repeat(1, self.config.num_search_samples).view(
            -1, seq_len
        )

        with torch.no_grad():
            output = self.answer_model(
                input_ids=answer_input_ids,
                attention_mask=answer_input_mask,
                decoder_attention_mask=target_mask,
                decoder_input_ids=self.answer_model._shift_right(labels),
                labels=None,
            )

            log_p = -loss_fct(
                output.logits.view(-1, output.logits.size(-1)),
                labels.view(-1),
            )

            b, sz, v = output.logits.size()
            log_p = log_p.view(b, sz)
            good_log_p = log_p.masked_fill_(labels == -100, 0.0)
            answer_log_p = torch.sum(good_log_p, dim=1).squeeze()
            answer_log_p = answer_log_p.view(self.config.num_search_samples, b_sz)

        # Now re-run the question generator and compute the loss for the sampled predictions.
        # This will compute the gradients in the question module.
        b_sz, seq_len = question_input_ids.size()
        question_input_ids = question_input_ids.repeat(
            1, self.config.num_search_samples
        ).view(-1, seq_len)
        question_input_mask = question_input_mask.repeat(
            1, self.config.num_search_samples
        ).view(-1, seq_len)
        output_questions = []
        for i in range(b_sz):
            for j in range(self.config.num_search_samples):
                output_question = (
                    white_space_fix(sampled_question_predictions_str_reshaped[i][j])
                    + " </s>"
                )
                output_questions.append(output_question)

        question_encodings = self.question_tokenizer(
            output_questions,
            truncation=True,
            padding="max_length",
            max_length=self.config.decoder_max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        question_labels = question_encodings.pop("input_ids")
        question_target_mask = question_encodings.pop("attention_mask")

        # because HuggingFace automatically shifts the labels, the labels correspond exactly to `target_ids`.
        # We have to make sure that the PAD token is ignored

        question_labels = [
            [
                -100 if token == self.question_tokenizer.pad_token_id else token
                for token in labels
            ]
            for labels in question_labels.tolist()
        ]
        question_labels = torch.tensor(question_labels)
        if self.config.gpu:
            question_target_mask = question_target_mask.to(current_device)
            question_labels = question_labels.to(current_device)

        self.question_model.train()
        question_output = self.question_model(
            input_ids=question_input_ids,
            attention_mask=question_input_mask,
            decoder_attention_mask=question_target_mask,
            decoder_input_ids=self.question_model._shift_right(question_labels),
            labels=None,
        )

        log_question_p = -loss_fct(
            question_output.logits.view(-1, question_output.logits.size(-1)),
            question_labels.view(-1),
        )

        b, sz, v = question_output.logits.size()
        log_question_p = log_question_p.view(b, sz)
        good_log_question_p = log_question_p.masked_fill_(question_labels == -100, 0.0)
        question_log_p = torch.sum(good_log_question_p, dim=1).squeeze()
        question_log_p = question_log_p.view(self.config.num_search_samples, b_sz)

        # MML
        re_loss = -torch.mean(
            torch.log(
                torch.sum(
                    torch.mul(
                        torch.transpose(torch.exp(answer_log_p), 0, 1),
                        torch.transpose(torch.exp(question_log_p), 0, 1),
                    ),
                    dim=1,
                )
            ),
            dim=0,
        )

        loss = re_loss
        loss_value = loss.item()

        return loss, loss_value

    def iterative_train(self, batch, current_device, phase="answer", sample_p=0.9):
        # Free memory in GPU, very important!
        clear_cache()
        # Turn on training mode which enables dropout.
        if phase == "answer":
            self.question_optimizer.zero_grad()
            self.answer_optimizer.zero_grad()
            self.question_model.eval()
            return self.pgg_answer_training(batch, current_device)

        elif phase == "question":
            self.question_optimizer.zero_grad()
            self.answer_optimizer.zero_grad()
            self.answer_model.eval()
            return self.mml_question_training(batch, current_device, sample_p=sample_p)
