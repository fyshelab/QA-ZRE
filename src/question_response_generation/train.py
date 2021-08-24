import argparse
import csv
import io
import json
import math
import os
import time
from configparser import ConfigParser
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from src.question_response_generation.question_utils import \
    create_question_dataset
from src.question_response_generation.response_utils import \
    create_response_dataset
from src.question_response_generation.t5_model import T5QA, HyperParameters


def read_squad_refs(path):
    path = Path(path)
    with open(path, "rb") as f:
        squad_dict = json.load(f)

    all_refs = []
    for group in squad_dict["data"]:
        for passage in group["paragraphs"]:
            for qa in passage["qas"]:
                temp = []
                for answer in qa["answers"]:
                    temp.append(answer["text"])
                if temp:
                    all_refs.append(temp)

    return all_refs


def run_train_epoch(
    model,
    train_dataloader,
) -> Generator:
    """Train the model and return the loss for each step.
    Randomly pick a batch from the train_dataset.
    """
    step = 0
    for batch in train_dataloader:
        loss_values = model.train(batch)
        step += 1
        yield step, loss_values["loss_value"]


def run_predict(model, dev_dataloader, prediction_file: str) -> None:
    """Read the 'dev_dataset' and predict results with the model, and save the
    results in the prediction_file."""
    writerparams = {"quotechar": '"', "quoting": csv.QUOTE_ALL}
    with io.open(prediction_file, mode="w", encoding="utf-8") as out_fp:
        writer = csv.writer(out_fp, **writerparams)
        header_written = False
        for batch in dev_dataloader:
            for ret_row in model.predict(batch):
                if not header_written:
                    headers = ret_row.keys()
                    writer.writerow(headers)
                    header_written = True
                writer.writerow(list(ret_row.values()))


def save_config(config: HyperParameters, path: str) -> None:
    """Saving config dataclass."""

    config_dict = vars(config)
    parser = ConfigParser()
    parser.add_section("train-parameters")
    for key, value in config_dict.items():
        parser.set("train-parameters", str(key), str(value))
    # save to a file
    with io.open(
        os.path.join(path, "config.ini"), mode="w", encoding="utf-8"
    ) as configfile:
        parser.write(configfile)


def run_model(
    model,
    config,
    train_dataloader=None,
    dev_dataloader=None,
    test_dataloader=None,
    save_always: Optional[bool] = False,
) -> None:
    """Run the model on input data (for training or testing)"""

    model_path = config.model_path
    max_epochs = config.max_epochs
    mode = config.mode
    if mode == "train":
        print("\nINFO: ML training\n")
        first_start = time.time()
        epoch = 0
        while epoch < max_epochs:
            print("\nEpoch:{0}\n".format(epoch))
            start = time.time()
            total_loss = []
            for step, loss in run_train_epoch(model, train_dataloader):
                if math.isnan(loss):
                    print("nan loss")

                if loss:
                    total_loss.append(loss)

                if total_loss:
                    mean_loss = np.mean(total_loss)

                else:
                    mean_loss = float("-inf")

                print(
                    "\rBatch:{0} | Loss:{1} | Mean Loss:{2}\n".format(
                        step, loss, mean_loss
                    )
                )
                if step > 0 and save_always and (step % 10000 == 0):
                    model.save(str(epoch) + "_step_" + str(step))

            if save_always:
                model.save(str(epoch))

            msg = "\nEpoch training time:{} seconds\n".format(time.time() - start)
            print(msg)
            epoch += 1

        save_config(config, model_path)
        msg = "\nTotal training time:{} seconds\n".format(time.time() - first_start)
        print(msg)

    elif mode == "test":
        print("Predicting...")
        start = time.time()
        run_predict(model, test_dataloader, config.prediction_file)
        msg = "\nTotal prediction time:{} seconds\n".format(time.time() - start)
        print(msg)


def run_all(args):
    """Run the T5 on multiple qa datasets."""
    config = HyperParameters(
        model_path=args.model_path,
        batch_size=args.batch_size,
        source_max_length=512,
        decoder_max_length=128,
        gpu=args.gpu,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        mode="train",
        prediction_file=args.prediction_file,
    )

    model = T5QA(config)

    if args.question_training:
        (
            train_loader,
            val_loader,
            test_loader,
            train_dataset,
            dev_dataset,
            test_dataset,
            train_sampler,
        ) = create_question_dataset(
            tokenizer=model.tokenizer,
            batch_size=config.batch_size,
            source_max_length=config.source_max_length,
            decoder_max_length=config.decoder_max_length,
        )
    else:
        (
            train_loader,
            val_loader,
            test_loader,
            train_dataset,
            dev_dataset,
            test_dataset,
            train_sampler,
        ) = create_response_dataset(
            tokenizer=model.tokenizer,
            batch_size=config.batch_size,
            source_max_length=config.source_max_length,
            decoder_max_length=config.decoder_max_length,
        )

    run_model(
        model,
        config=config,
        train_dataloader=train_loader,
        dev_dataloader=val_loader,
        test_dataloader=None,
        save_always=True,
    )


def run_main(args):
    """Decides what to do in the code."""
    if args.mode in ["all_train"]:
        run_all(args)


def argument_parser():
    """Augments arguments for protein-gene model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="all_train",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path for saving or loading models.",
    )

    # Train specific
    parser.add_argument("--train", type=str, help="file for train data.")

    parser.add_argument("--dev", type=str, help="file for validation data.")

    # Test specific
    parser.add_argument("--test", type=str, help="file for test data.")

    parser.add_argument(
        "--prediction_file", type=str, help="file for saving predictions"
    )

    parser.add_argument("--input_file_name", type=str, help="input file name")

    # Hyper-Parameters
    parser.add_argument("--dim_model", type=int, default=100, help="dim of model units")

    parser.add_argument(
        "--dropout", type=float, default=0.1, help="the probability of zeroing a link"
    )

    parser.add_argument("--dim_embedding", type=int, default=100, help="embedding size")

    parser.add_argument("--learning_rate", type=float, default=0.0005)

    parser.add_argument(
        "--max_gradient_norm",
        type=float,
        default=10.0,
        help="max norm allowed for gradients",
    )

    parser.add_argument("--batch_size", type=int, default=8, help="static batch size")

    parser.add_argument(
        "--num_train_steps", type=int, default=50000, help="number of train steps"
    )

    parser.add_argument(
        "--max_epochs", type=int, default=25, help="max number of training iterations"
    )

    parser.add_argument("--gpu_device", type=int, default=0, help="gpu device to use")

    parser.add_argument("--seed", type=int, default=len("dream"), help="random seed")

    parser.add_argument(
        "--config_file", type=str, default="config.ini", help="config.ini file"
    )

    parser.add_argument(
        "--question_training",
        type=bool,
        default=False,
        help="for question generation or not? True or False",
    )

    # GPU or CPU
    parser.add_argument(
        "--gpu", type=bool, default=True, help="on gpu or not? True or False"
    )

    parser.add_argument(
        "--dream_path", type=str, help="path for reading dream scape data!"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="checkpoint of the trained model."
    )
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    run_main(args)
