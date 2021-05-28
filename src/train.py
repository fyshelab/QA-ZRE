import argparse
import csv
import io
import math
import os
import time
from configparser import ConfigParser
from typing import Generator, Optional

import datasets
import numpy as np
import pandas as pd
from datasets import load_dataset

from src.albert_model import HyperParameters, Model


def run_train_epoch(
    model,
    batch_size: int,
    num_steps: int,
    train_dataset,
) -> Generator:
    """Train the model and return the loss for 'num_steps' given the
    'batch_size' and the train_dataset.

    Randomly pick a batch from the train_dataset.
    """
    shuff_dataset = train_dataset.shuffle(seed=42)
    for step in range(num_steps):
        batch_start = step * batch_size
        batch_data = shuff_dataset.select(
            range(batch_start, batch_start + batch_size, 1)
        )
        batch_data = batch_data.shuffle(seed=step)
        loss_values = model.train(batch_data)
        yield step, loss_values["loss_value"]


def run_predict(model, batch_size: int, dev_dataset, prediction_file: str) -> None:
    """Read the 'dev_dataset' and predict results with the model, and save the
    results in the prediction_file."""

    num_steps = math.ceil(len(dev_dataset) / batch_size)
    writerparams = {"quotechar": '"', "quoting": csv.QUOTE_ALL}
    with io.open(prediction_file, mode="w", encoding="utf-8") as out_fp:
        writer = csv.writer(out_fp, **writerparams)
        header_written = False
        for step in range(num_steps):
            batch_start = step * batch_size
            batch_data = dev_dataset.select(
                range(batch_start, batch_start + batch_size, 1)
            )
            for ret_row in model.predict(batch_data):
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
    evaluator,
    train_dataset=None,
    dev_dataset=None,
    test_dataset=None,
    save_always: Optional[bool] = False,
) -> None:
    """Run the model on input data (for training or testing)"""

    model_path = config.model_path
    max_epochs = config.max_epochs
    batch_size = config.batch_size
    mode = config.mode
    if mode == "train":
        print("\nINFO: ML training\n")
        # Used to save dev set predictions.
        prediction_file = os.path.join(model_path, "temp.predicted")
        best_val_cost = float("inf")
        first_start = time.time()
        epoch = 0
        while epoch < max_epochs:
            print("\nEpoch:{0}\n".format(epoch))
            start = time.time()
            total_loss = []
            for step, loss in run_train_epoch(
                model, config.batch_size, config.num_train_steps, train_dataset
            ):
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
            print("\nValidation:\n")
            run_predict(model, config.batch_size, dev_dataset, prediction_file)
            val_cost = evaluator(prediction_file)

            print("\nValidation cost:{0}\n".format(val_cost))
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                model.save("best")
            if save_always:
                model.save(str(epoch))
            msg = "\nEpoch training time:{} seconds\n".format(time.time() - start)
            print(msg)
            epoch += 1

        save_config(config, model_path)
        msg = "\nTotal training time:{} seconds\n".format(time.time() - first_start)
        print(msg)
        # Remove the temp output file
        os.remove(prediction_file)

    elif mode == "test":
        print("Predicting...")
        start = time.time()
        run_predict(model, batch_size, test_dataset, config.prediction_file)
        msg = "\nTotal prediction time:{} seconds\n".format(time.time() - start)
        print(msg)


def list_parameters(model):
    parameters = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            parameters[name] = param

    return parameters


# load rouge for validation
rouge = datasets.load_metric("rouge")


def compute_rouge(prediction_file):
    df = pd.read_csv(prediction_file)
    rouge_output = rouge.compute(
        predictions=df["predictions_str"].tolist(),
        references=df["target_str"].tolist(),
        rouge_types=["rouge2"],
    )["rouge2"].mid

    output = {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }
    return 1.0 - output["rouge2_fmeasure"]


def create_squad_dataset(tokenizer, batch_size, source_max_length, decoder_max_length):
    """Function to create the squad dataset."""

    def process_squad_row(row):
        """Helper function to have access to the tokenizer."""

        try:
            answer = row["answers"]["text"][0]
            answer = " ".join(answer.split())
        except:
            answer = "<NoAnswer>"

        question = row["question"]
        question = " ".join(question.split())

        article = row["context"]
        article = " ".join(article.split())

        entries = {
            "cls": tokenizer.bos_token,
            "sep": tokenizer.eos_token,
            "passage": article,
            "question": question,
            "answer": answer,
        }
        return {
            "inputs": " {passage} {sep} {question} ".format(**entries),
            "outputs": " {answer} ".format(**entries),
        }

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(
            batch["inputs"],
            padding="max_length",
            truncation="only_first",
            max_length=source_max_length,
            add_special_tokens=True,
        )
        outputs = tokenizer(
            batch["outputs"],
            padding="max_length",
            truncation="only_first",
            max_length=decoder_max_length,
            add_special_tokens=True,
        )

        batch["input_ids"] = inputs.input_ids
        batch["input_token_type_ids"] = []
        for input_id in inputs.input_ids:
            sep_index = input_id.index(
                tokenizer._convert_token_to_id(tokenizer.eos_token)
            )
            article_type = [0] * (sep_index + 1)
            question_type = [1] * (source_max_length - (sep_index + 1))

            token_type_mask = article_type + question_type
            batch["input_token_type_ids"].append(token_type_mask)

        batch["input_mask"] = inputs.attention_mask
        batch["target_ids"] = outputs.input_ids
        batch["target_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        # because BERT automatically shifts the labels, the labels correspond exactly to `target_ids`.
        # We have to make sure that the PAD token is ignored

        labels = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]
        batch["labels"] = labels
        return batch

    train_dataset = load_dataset("squad_v2", split="train")
    train_dataset = train_dataset.map(
        process_squad_row,
        remove_columns=["id", "title", "question", "answers", "context"],
    ).filter(lambda row: "<NoAnswer>" not in row["outputs"])

    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["inputs", "outputs"],
    )

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "input_mask", "target_ids", "target_mask", "labels"],
    )

    val_dataset = load_dataset("squad_v2", split="validation")
    val_dataset = val_dataset.map(
        process_squad_row,
        remove_columns=["id", "title", "question", "answers", "context"],
    ).filter(lambda row: "<NoAnswer>" not in row["outputs"])

    val_dataset = val_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["inputs", "outputs"],
    )

    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "input_mask", "target_ids", "target_mask", "labels"],
    )
    print(len(train_dataset))
    return train_dataset, val_dataset


def run_squad(args):
    """Train albert model on squad dataset."""
    if args.mode == "squad_train":
        mode = "train"
    elif args.mode == "squad_test":
        mode = "test"
    config = HyperParameters(
        model_path=args.model_path,
        batch_size=args.batch_size,
        source_max_length=512,
        decoder_max_length=128,
        gpu=args.gpu,
        gpu_device=args.gpu_device,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        mode=mode,
        num_train_steps=args.num_train_steps,
        prediction_file=args.prediction_file,
    )
    albert2albert = Model(config)
    train_dataset, val_dataset = create_squad_dataset(
        tokenizer=albert2albert.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
    )
    run_model(
        albert2albert,
        config=config,
        evaluator=compute_rouge,
        train_dataset=val_dataset,
        dev_dataset=val_dataset,
        test_dataset=val_dataset,
    )


def run_main(args):
    """Decides what to do in the code."""
    if args.mode in ["squad_train", "squad_test"]:
        run_squad(args)


def argument_parser():
    """augments arguments for protein-gene model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="squad_train | squad_test",
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

    # GPU or CPU
    parser.add_argument(
        "--gpu", type=bool, default=True, help="on gpu or not? True or False"
    )

    parser.add_argument(
        "--dream_path", type=str, help="path for reading dream scape data!"
    )
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    run_main(args)
