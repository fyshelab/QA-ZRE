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

FLAGS = flags.FLAGS

# for all possible t5_exp_type, see 'optimizer_definer'
flags.DEFINE_string("t5_exp_type", "qa", "The type of experiment with the T5 model.")

flags.DEFINE_integer("seed", 42, "the seed number")
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
            self.optimizers = optimizer_definer[FLAGS.t5_exp_type](self.model_pool)
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

        # only use the single optimizer defined.
        self.optimizer = self.optimizers[0]

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
