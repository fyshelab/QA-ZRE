"""This module implements the t5 models for the response and question
generation for relation extraction."""

import gc
import os
import random
from abc import abstractmethod
from typing import Dict, Iterator, List, Tuple

import numpy
import torch
from absl import flags
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.optimizers import optimizer_definer
from src.zero_extraction_utils import remove_prefix

FLAGS = flags.FLAGS

# for all possible t5_exp_type, see 'optimizer_definer'
flags.DEFINE_string("t5_exp_type", "qa", "The type of experiment with the T5 model.")

flags.DEFINE_integer("seed", 42, "the seed number")
flags.DEFINE_integer("no_repeat_ngram_size", 2, "related to beam search decoding.")
flags.DEFINE_integer(
    "num_search_samples", 8, "number of samples to search for the questions."
)
flags.DEFINE_bool("gpu", False, "Whether to put the model on gpu or not?")

# https://huggingface.co/t5-small
flags.DEFINE_string(
    "t5_pretrained_model", "t5-small", "initial pre-trained model to use as T5."
)

flags.DEFINE_string("mode", "train", "the mode of run? train or test")
flags.DEFINE_string(
    "model_path", "/tmp/", "main directory to save or load the model from"
)
flags.DEFINE_string("checkpoint", None, "checkpoint name to load from.")
flags.DEFINE_float("dropout_rate", 0.1, "dropout_rate used in T5 base.")


def clear_cache() -> None:
    """Clean unused GPU Cache!"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def set_random_seed(seed: int) -> None:
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


class MyBaseT5(torch.nn.Module):
    """Base class for different t5 experiments."""

    def __init__(self) -> None:
        super(MyBaseT5, self).__init__()

        set_random_seed(FLAGS.seed)

        # check the gpu actually exists and setup device.
        self.gpu_check = FLAGS.gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu_check else "cpu")

        # will contain a dictionary with model name as the key
        # and the actual model as the value.
        self.model_pool: Dict[str, torch.nn.Module] = {}

    def setup_models(self) -> None:
        """Setup optimizer in training or load from the checkpoint for
        testing."""
        # put model on gpu or cpu.
        for model in self.model_pool.values():
            model.to(self.device)

        if FLAGS.mode == "train":
            # create optimizer only for training.
            # based on the experiment type, setup the optimizer.
            optimizers = optimizer_definer[FLAGS.t5_exp_type](self.model_pool)
            if FLAGS.t5_exp_type == "qa":
                # only use the single optimizer defined.
                self.optimizer = optimizers[0]
            elif FLAGS.t5_exp_type == "qa_zre":
                # create two optimizers only for training.
                self.question_optimizer = optimizers[0]
                self.answer_optimizer = optimizers[1]
                # need to load from the pre-trained models.
                self.load_from_checkpoint()

        elif FLAGS.mode in ["test", "inference", "eval"]:
            # load from the given checkpoint.
            self.load_from_checkpoint()
        elif FLAGS.mode in ["no_finetune_test"]:
            # just rely on the pre-trained T5 for prediction and no loading from the checkpoint.
            pass
        else:
            raise Exception("Wrong mode {}!".format(FLAGS.mode))

    def load_from_checkpoint(self) -> None:
        """Loads the weights from the given checkpoint."""
        m_path = FLAGS.model_path
        ckp_name = FLAGS.checkpoint
        try:
            for m_name, model in self.model_pool.items():
                if FLAGS.t5_exp_type == "qa_zre" and m_name == "init_question_model":
                    # search model is initialized from the same pretrained question model.
                    m_name = "question_model"
                model_ckp = os.path.join(m_path, f"{m_name}_{ckp_name}")
                model.load_state_dict(
                    torch.load(
                        model_ckp,
                        map_location=lambda storage, loc: storage,
                    )
                )
        except Exception as e:
            raise Exception("Could not load the checkpoint due to error:{}".format(e))

    def save(self, checkpoint_name: str) -> None:
        """Save the modules to the model_path for the specified checkpoint
        name."""
        m_path = FLAGS.model_path
        if not os.path.exists(m_path):
            os.makedirs(m_path)
        for m_name, model in self.model_pool.items():
            if FLAGS.t5_exp_type == "qa_zre" and m_name == "init_question_model":
                # skip saving the pre-trained search module.
                continue
            torch.save(
                model.state_dict(), os.path.join(m_path, f"{m_name}_{checkpoint_name}")
            )

    def train_mode_on(self) -> None:
        """Before every forward-backward iteration over batch, clear gpu cache,
        turn on train mode."""

        clear_cache()

        # turn on training mode which enables dropout.
        for model in self.model_pool.values():
            model.train()

    def predict_mode_on(self) -> None:
        """For each iteration of prediction over batch, clear gpu cache, turn
        on eval mode."""

        clear_cache()

        # turn on eval mode which disables dropout.
        for model in self.model_pool.values():
            model.eval()

    def move_to_gpu(
        self, batch: torch.utils.data.Dataset, keys: List[str]
    ) -> Dict[str, torch.Tensor]:
        """If gpu flag is set, move the batch tensors specified by keys into
        the gpu and return a dictionary to access the gpu tensors."""
        return {key: batch[key].to(self.device) for key in keys}

    @abstractmethod
    def train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """The abstract train function."""
        pass

    @abstractmethod
    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The abstract predict function."""
        pass


class QAT5(MyBaseT5):
    """Wrapper class around the MyBaseT5 Model to experiment with a single T5
    model as a question or response module and pre-train it on QA corpora or
    fine-tune it on the RE corpora."""

    def __init__(self) -> None:
        super(QAT5, self).__init__()

        # construct tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(FLAGS.t5_pretrained_model)

        # construct the underlying t5 model
        self.model_pool["t5_model"] = T5ForConditionalGeneration.from_pretrained(
            FLAGS.t5_pretrained_model
        )

        self.setup_models()

    def train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """The main train loop for generating the output sequence in the
        decoder T5."""

        self.train_mode_on()
        self.optimizer.zero_grad()

        loaded_batch = self.move_to_gpu(
            batch,
            keys=["input_ids", "attention_mask", "target_attention_mask", "labels"],
        )

        # we have to make sure that the PAD token is ignored.
        # huggingface ignores a pad token if the token is -100!
        labels = loaded_batch["labels"]
        labels.masked_fill_(labels == self.tokenizer.pad_token_id, -100)

        t5_model = self.model_pool["t5_model"]

        output = t5_model(
            input_ids=loaded_batch["input_ids"],
            attention_mask=loaded_batch["attention_mask"],
            decoder_attention_mask=loaded_batch["target_attention_mask"],
            labels=labels,
        )

        loss = output.loss
        loss_value = loss.item()

        # backProp
        loss.backward()

        # optimize
        self.optimizer.step()

        return {"loss_value": loss_value}

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The main prediction loop for generating the output sequence."""
        self.predict_mode_on()

        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask"])

        # this uses greedy decoding.
        t5_model = self.model_pool["t5_model"]
        predictions = t5_model.generate(
            input_ids=loaded_batch["input_ids"],
            attention_mask=loaded_batch["attention_mask"],
        )

        # all special tokens will be removed.
        predictions_str = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        input_str = self.tokenizer.batch_decode(
            loaded_batch["input_ids"], skip_special_tokens=True
        )
        for index, pred_str in enumerate(predictions_str):
            output_row = {
                "predictions_str": pred_str,
                "input_str": input_str[index],
            }
            yield output_row


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.stack(list(tuple_of_tensors), dim=0)


def prepare_response_module_input(
    labels=None,
    target_mask=None,
    num_samples=1,
):
    """Repeat the labels and the target_mask num_samples times in dimension
    1."""
    b_sz, dec_seq_len = labels.size()
    labels = labels.repeat(1, num_samples).view(-1, dec_seq_len)
    target_mask = target_mask.repeat(1, num_samples).view(-1, dec_seq_len)

    return target_mask, labels


def prob_of_sampled_predictions(loss_fct, sample_outputs):
    """Helper function to compute the predictions and the probability of a
    sampled sequence from the output of the generate function in the
    transformers library."""

    # Skip the first pad token generated by the T5 model.
    sampled_predictions = sample_outputs.sequences[:, 1:]
    sampled_scores = sample_outputs.scores
    sampled_scores = tuple_of_tensors_to_tensor(sampled_scores)

    # v: vocab size
    # n: batch_size * num_samples
    # l: sequence len
    l, n, v = sampled_scores.size()
    log_p = -loss_fct(
        sampled_scores.view(-1, v),
        torch.reshape(torch.transpose(sampled_predictions, 0, 1), (l * n,)),
    ).view(l, n)
    pad_mask = torch.transpose(sampled_predictions, 0, 1) == 0
    good_log_p = log_p.masked_fill_(pad_mask, 0.0)
    log_p = torch.sum(good_log_p, dim=0).squeeze()
    return sampled_predictions, log_p


class QA_ZRET5(MyBaseT5):
    """Wrapper class around the MyBaseT5 Model to experiment with three T5
    models:
        1 - one T5 model as question generator.
        2 - one T5 model as response generator.
        3 - one T5 model as question search module.

    The (1) and (2) will be trained on the RE corpora for tail enity generation task using the OffMML-G objective.
    """

    def __init__(self, source_max_len, decoder_max_len) -> None:
        super(QA_ZRET5, self).__init__()

        self.source_max_length = source_max_len
        self.decoder_max_length = decoder_max_len

        # define tokenizers:
        # answer model tokenizer
        self.answer_tokenizer = T5Tokenizer.from_pretrained(FLAGS.t5_pretrained_model)

        # question model tokenizer
        self.question_tokenizer = T5Tokenizer.from_pretrained(FLAGS.t5_pretrained_model)

        # construct the underlying t5 models
        self.model_pool["answer_model"] = T5ForConditionalGeneration.from_pretrained(
            FLAGS.t5_pretrained_model
        )

        self.model_pool["question_model"] = T5ForConditionalGeneration.from_pretrained(
            FLAGS.t5_pretrained_model
        )

        if FLAGS.mode == "train":
            # the search model is only for training.
            # question search module tokenizer
            self.init_question_tokenizer = T5Tokenizer.from_pretrained(
                FLAGS.t5_pretrained_model
            )
            self.model_pool[
                "init_question_model"
            ] = T5ForConditionalGeneration.from_pretrained(FLAGS.t5_pretrained_model)

        self.setup_models()

    def question_beam_predict(self, batch, num_ret_seqs=1):
        """Use beam search to generate the questions and prepare inputs for the
        answer module."""

        loaded_batch = self.move_to_gpu(
            batch,
            keys=[
                "entity_relation_passage_input_ids",
                "entity_relation_passage_attention_mask",
            ],
        )
        q_input_ids = loaded_batch["entity_relation_passage_input_ids"]
        q_attn_mask = loaded_batch["entity_relation_passage_attention_mask"]
        with torch.no_grad():
            question_model = self.model_pool["question_model"]
            question_output = question_model.generate(
                input_ids=q_input_ids,
                attention_mask=q_attn_mask,
                no_repeat_ngram_size=FLAGS.no_repeat_ngram_size,
                early_stopping=True,
                max_length=self.decoder_max_length,
                num_return_sequences=num_ret_seqs,
                num_beams=FLAGS.num_search_samples,
                length_penalty=1.0,  # no penalty
                output_scores=True,
                return_dict_in_generate=True,
            )
            question_predictions = question_output.sequences
            question_log_ps = question_output.sequences_scores

        question_predictions_str = self.question_tokenizer.batch_decode(
            question_predictions, skip_special_tokens=True
        )

        question_predictions_str = [
            remove_prefix(pred, "question: ") for pred in question_predictions_str
        ]

        new_articles = []
        for i in range(len(question_predictions_str)):
            new_article = (
                "relation: "
                + batch["entity_relations"][i // num_ret_seqs]
                + " question: "
                + question_predictions_str[i]
                + " context: "
                + batch["passages"][i // num_ret_seqs]
                + " </s>"
            )
            new_articles.append(new_article)

        answer_inputs = self.answer_tokenizer(
            new_articles,
            truncation=True,
            padding="max_length",
            max_length=self.source_max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return (
            answer_inputs.input_ids.to(self.device),
            answer_inputs.attention_mask.to(self.device),
            q_input_ids,
            q_attn_mask,
            question_predictions_str,
            question_log_ps,
        )

    def tail_entity_gen(self, batch):
        """Code to generate the question from the question module and then
        generate the tail entity from the response module."""
        self.predict_mode_on()
        (
            answer_input_ids,
            answer_input_mask,
            _,
            _,
            question_predictions_str,
            question_log_ps,
        ) = self.question_beam_predict(batch)

        answer_model = self.model_pool["answer_model"]
        # greedy decoding with the answer model.
        tail_entity_predictions = answer_model.generate(
            input_ids=answer_input_ids,
            attention_mask=answer_input_mask,
        )
        tail_entity_predictions_str = self.answer_tokenizer.batch_decode(
            tail_entity_predictions, skip_special_tokens=True
        )

        for index, tail_entity in enumerate(tail_entity_predictions_str):
            output_row = {
                "predictions_str": tail_entity,
                "question_predictions": question_predictions_str[index],
            }
            yield output_row

    def relation_scorer(self, batch):
        """Relation classifier using tail entity generation for scoring every
        possible relation."""
        self.predict_mode_on()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        loss_fct = loss_fct.to(self.device)
        (
            answer_input_ids,
            answer_input_mask,
            question_input_ids,
            question_input_mask,
            question_predictions_str,
            question_log_ps,
        ) = self.question_beam_predict(
            batch,
            num_ret_seqs=FLAGS.num_search_samples,
        )

        loaded_batch = self.move_to_gpu(
            batch,
            keys=[
                "second_entity_attention_mask",
                "second_entity_labels",
            ],
        )
        target_mask, labels = prepare_response_module_input(
            labels=loaded_batch["second_entity_labels"],
            target_mask=loaded_batch["second_entity_attention_mask"],
            num_samples=FLAGS.num_search_samples,
        )
        labels.masked_fill_(labels == self.answer_tokenizer.pad_token_id, -100)

        # Answer Computation
        with torch.no_grad():
            answer_model = self.model_pool["answer_model"]
            output = answer_model(
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

            # b: batch size
            # sz: sequence size
            # v: vocab size
            b, sz, v = output.logits.size()
            log_p = log_p.view(b, sz)
            good_log_p = log_p.masked_fill_(labels == -100, 0.0)
            answer_log_p = torch.sum(good_log_p, dim=1).squeeze().cpu().numpy()
            question_log_ps = question_log_ps.cpu().numpy()
            for index in range(b):
                relation_log_p = answer_log_p[index] + question_log_ps[index]
                output_row = {
                    "relation_log_p": relation_log_p,
                    "question_log_p": question_log_ps[index],
                    "answer_log_p": answer_log_p[index],
                    "generated_question": question_predictions_str[index],
                }
                yield output_row
