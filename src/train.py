import argparse
import csv
import io
import json
import math
import os
import random
import re
import string
import time
from collections import Counter
from configparser import ConfigParser
from pathlib import Path
from typing import Generator, Optional

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.nq_utils import create_narrative_dataset
from src.t5_model import T5QA, HyperParameters


def white_space_fix(text):
    return " ".join(text.split())


def read_dream_data(file):
    df = pd.read_csv(file)
    articles_df = df["text"].tolist()
    questions_df = df["question"].tolist()
    answers_df = df["correctAnswers"].tolist()
    num_articles = len(articles_df)

    contexts = []
    answers = []
    for i in range(num_articles):
        question = white_space_fix(questions_df[i])
        article = white_space_fix(articles_df[i])
        answer = answers_df[i]
        contexts.append("question: " + question + " context: " + article + " </s>")
        # context = question + " \n " + article
        # ctx = context.lower()
        # ctx = re.sub("'(.*)'", r"\1", ctx)
        # contexts.append(ctx)
        answers.append(answer)

    return contexts, answers


def create_dream_dataset(
    file_name, tokenizer, batch_size, source_max_length, decoder_max_length
):
    """Function to create the squad dataset."""
    val_contexts, val_answers = read_dream_data(file_name)

    val_encodings = tokenizer(
        val_contexts,
        truncation=True,
        padding="max_length",
        max_length=source_max_length,
        add_special_tokens=False,
    )

    class SquadDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings.input_ids)

    val_dataset = SquadDataset(val_encodings)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader


def read_squad(path):
    path = Path(path)
    with open(path, "rb") as f:
        squad_dict = json.load(f)

    contexts = []
    answers = []
    for group in squad_dict["data"]:
        for passage in group["paragraphs"]:
            context = passage["context"]
            for qa in passage["qas"]:
                question = qa["question"]
                if qa["answers"]:
                    answ = random.choice(qa["answers"])
                    contexts.append(
                        "question: " + question + " context: " + context + " </s>"
                    )
                    answers.append(answ["text"] + " </s>")

    return contexts, answers


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


def create_race_dataset(tokenizer, batch_size, source_max_length, decoder_max_length):
    """Function to create the race dataset."""

    def process_race_row(row):
        """Helper function."""
        option_code = row["answer"]
        if option_code == "A":
            option_idx = 0
        elif option_code == "B":
            option_idx = 1
        elif option_code == "C":
            option_idx = 2
        elif option_code == "D":
            option_idx = 3

        answer = row["options"][option_idx]
        answer = " ".join(answer.split())

        question = row["question"]
        question = " ".join(question.split())

        article = row["article"]
        article = " ".join(article.split())

        return {
            "article": "question: " + question + " context: " + article + " </s>",
            "answer": answer + " </s>",
        }

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(
            batch["article"],
            truncation=True,
            padding="max_length",
            max_length=source_max_length,
            add_special_tokens=False,
        )
        outputs = tokenizer(
            batch["answer"],
            truncation=True,
            padding="max_length",
            max_length=decoder_max_length,
            add_special_tokens=False,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        batch["target_ids"] = outputs.input_ids
        batch["target_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        # because BERT automatically shifts the labels, the labels correspond exactly to `target_ids`.
        # We have to make sure that the PAD token is ignored

        labels = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]
        batch["labels"] = labels

        return batch

    train_dataset = load_dataset("race", "all", split="train")
    train_dataset = train_dataset.map(
        process_race_row,
        remove_columns=["options", "example_id"],
    )
    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["answer", "article"],
    )
    train_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "target_ids",
            "target_attention_mask",
            "labels",
        ],
    )

    dev_dataset = load_dataset("race", "all", split="validation")
    dev_dataset = dev_dataset.map(
        process_race_row,
        remove_columns=["options", "example_id"],
    )
    dev_dataset = dev_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["answer", "article"],
    )
    dev_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "target_ids",
            "target_attention_mask",
            "labels",
        ],
    )
    test_dataset = load_dataset("race", "all", split="test")
    test_dataset = test_dataset.map(
        process_race_row,
        remove_columns=["options", "example_id"],
    )
    test_dataset = test_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["answer", "article"],
    )
    test_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "target_ids",
            "target_attention_mask",
            "labels",
        ],
    )

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset  # , val_loader, test_loader


def run_train_epoch(
    model,
    train_dataloader,
) -> Generator:
    """Train the model and return the loss for 'num_steps' given the
    'batch_size' and the train_dataset.

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
        # Used to save dev set predictions.
        # prediction_file = os.path.join(model_path, "temp.predicted")
        # best_val_cost = float("inf")
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
                if save_always and step > 0 and (step % 1000 == 0):
                    model.save(str(epoch) + "_step_" + str(step))
            # print("\nValidation:\n")
            # run_predict(model, dev_dataloader, prediction_file)
            # val_cost = evaluator(prediction_file)

            # print("\nValidation cost:{0}\n".format(val_cost))
            # if val_cost < best_val_cost:
            #    best_val_cost = val_cost
            #    model.save("best")
            if save_always:
                model.save(str(epoch))
            msg = "\nEpoch training time:{} seconds\n".format(time.time() - start)
            print(msg)
            epoch += 1

        save_config(config, model_path)
        msg = "\nTotal training time:{} seconds\n".format(time.time() - first_start)
        print(msg)
        # Remove the temp output file
        # os.remove(prediction_file)

    elif mode == "test":
        print("Predicting...")
        start = time.time()
        run_predict(model, test_dataloader, config.prediction_file)
        msg = "\nTotal prediction time:{} seconds\n".format(time.time() - start)
        print(msg)


def list_parameters(model):
    parameters = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            parameters[name] = param

    return parameters


# load rouge for validation

# rouge = datasets.load_metric("rouge")


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
        total += 1
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths
        )
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}


def compute_rouge(prediction_file):
    df = pd.read_csv(prediction_file).astype(str)
    predictions = df["predictions_str"].tolist()
    references = df["target_str"].tolist()
    rouge_output = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rougeL"],
    )["rougeL"].mid

    output = {
        "rougeL_precision": round(rouge_output.precision, 4),
        "rougeL_recall": round(rouge_output.recall, 4),
        "rougeL_fmeasure": round(rouge_output.fmeasure, 4),
    }
    return output["rougeL_fmeasure"]


def create_squad_dataset(tokenizer, batch_size, source_max_length, decoder_max_length):
    """Function to create the squad dataset."""
    train_contexts, train_answers = read_squad("./squad/train-v2.0.json")
    val_contexts, val_answers = read_squad("./squad/dev-v2.0.json")

    val_encodings = tokenizer(
        val_contexts,
        truncation=True,
        padding="max_length",
        max_length=source_max_length,
        add_special_tokens=False,
    )
    val_answer_encodings = tokenizer(
        val_answers,
        truncation=True,
        padding="max_length",
        max_length=decoder_max_length,
        add_special_tokens=False,
    )

    train_encodings = tokenizer(
        train_contexts,
        truncation=True,
        padding="max_length",
        max_length=source_max_length,
        add_special_tokens=False,
    )
    train_answer_encodings = tokenizer(
        train_answers,
        truncation=True,
        padding="max_length",
        max_length=decoder_max_length,
        add_special_tokens=False,
    )

    train_encodings["target_ids"] = train_answer_encodings.input_ids
    train_encodings["target_attention_mask"] = train_answer_encodings.attention_mask

    train_encodings["labels"] = train_answer_encodings.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored

    train_labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in train_encodings["labels"]
    ]
    train_encodings["labels"] = train_labels

    val_encodings["target_ids"] = val_answer_encodings.input_ids
    val_encodings["target_attention_mask"] = val_answer_encodings.attention_mask

    val_encodings["labels"] = val_answer_encodings.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored

    val_labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in val_encodings["labels"]
    ]
    val_encodings["labels"] = val_labels

    class SquadDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings.input_ids)

    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_loader, val_loader


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
        decoder_max_length=32,
        gpu=args.gpu,
        gpu_device=args.gpu_device,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        mode=mode,
        num_train_steps=args.num_train_steps,
        prediction_file=args.prediction_file,
    )
    model = T5QA(config)

    train_loader, val_loader, test_loader = create_squad_dataset(
        tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
    )
    run_model(
        model,
        config=config,
        evaluator=compute_rouge,
        train_dataloader=train_loader,
        dev_dataloader=val_loader,
        test_dataloader=test_loader,
        save_always=True,
    )


def run_race(args):
    """Train albert model on race dataset."""
    if args.mode == "race_train":
        mode = "train"
    elif args.mode == "race_test":
        mode = "test"
    config = HyperParameters(
        model_path=args.model_path,
        batch_size=args.batch_size,
        source_max_length=1024,
        decoder_max_length=128,
        gpu=args.gpu,
        gpu_device=args.gpu_device,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        mode=mode,
        num_train_steps=args.num_train_steps,
        prediction_file=args.prediction_file,
    )
    model = T5QA(config)

    train_loader, val_loader, test_loader = create_race_dataset(
        tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
    )
    run_model(
        model,
        config=config,
        evaluator=compute_rouge,
        train_dataloader=train_loader,
        dev_dataloader=val_loader,
        test_dataloader=test_loader,
    )


def run_narrative(args):
    """Train albert model on narrative dataset."""
    if args.mode == "narrative_train":
        mode = "train"
    elif args.mode == "narrative_test":
        mode = "test"
    config = HyperParameters(
        model_path=args.model_path,
        batch_size=args.batch_size,
        source_max_length=512,
        decoder_max_length=128,
        gpu=args.gpu,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        mode=mode,
        prediction_file=args.prediction_file,
        checkpoint=args.checkpoint,
    )
    model = T5QA(config)

    train_loader, val_loader, test_loader, _, _, _ = create_narrative_dataset(
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
        test_dataloader=test_loader,
        save_always=True,
    )


def run_all(args):
    """Run the T5 on squad, race and narrative qa."""
    config = HyperParameters(
        model_path=args.model_path,
        batch_size=args.batch_size,
        source_max_length=512,
        decoder_max_length=128,
        gpu=args.gpu,
        gpu_device=args.gpu_device,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        mode="train",
        num_train_steps=args.num_train_steps,
        prediction_file=args.prediction_file,
    )
    model = T5QA(config)

    nar_train_dataset = create_narrative_dataset(
        tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
    )

    race_train_dataset = create_race_dataset(
        tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
    )

    sq_train_dataset = create_squad_dataset(
        tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
    )

    concat_dataset = torch.utils.data.ConcatDataset(
        [nar_train_dataset, race_train_dataset, sq_train_dataset]
    )
    train_loader = torch.utils.data.DataLoader(
        concat_dataset, batch_size=config.batch_size, shuffle=True
    )
    run_model(
        model,
        config=config,
        evaluator=compute_rouge,
        train_dataloader=train_loader,
        dev_dataloader=None,
        test_dataloader=None,
        save_always=True,
    )


def run_dream(args):
    """Test the model on dream dataset."""
    if args.mode == "dream_test":
        mode = "test"
    config = HyperParameters(
        model_path=args.model_path,
        batch_size=args.batch_size,
        source_max_length=256,
        decoder_max_length=128,
        gpu=args.gpu,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        mode=mode,
        prediction_file=args.prediction_file,
        checkpoint=args.checkpoint,
    )
    model = T5QA(config)

    val_loader = create_dream_dataset(
        args.input_file_name,
        tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
    )
    run_model(
        model,
        config=config,
        train_dataloader=None,
        dev_dataloader=None,
        test_dataloader=val_loader,
    )


def run_main(args):
    """Decides what to do in the code."""
    if args.mode in ["squad_train", "squad_test"]:
        run_squad(args)
    if args.mode in ["race_train", "race_test"]:
        run_race(args)
    if args.mode in ["narrative_train", "narrative_test"]:
        run_narrative(args)
    if args.mode in ["dream_test"]:
        run_dream(args)
    if args.mode in ["all_train"]:
        run_all(args)


def argument_parser():
    """augments arguments for protein-gene model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="dream_test | squad_train | squad_test | race_train | race_test | narrative_train | narrative_test",
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
