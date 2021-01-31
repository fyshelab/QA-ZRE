import argparse
import io
import json
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from datasets import Dataset, load_dataset
from transformers import AlbertTokenizer, Trainer, TrainingArguments

from src.albert_model import load_albert_encoder_decoder, load_pretrained_race

# set the tokenizer
tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v2")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
# for race
# source_max_length = 512
# decoder_max_length = 128
# batch_size = 2
# for dreamscape
source_max_length = 256
decoder_max_length = 64
batch_size = 4


def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["inputs"],
        padding="max_length",
        truncation="only_first",
        max_length=source_max_length,
    )
    outputs = tokenizer(
        batch["outputs"],
        padding="max_length",
        truncation="only_first",
        max_length=decoder_max_length,
    )

    batch["input_ids"] = inputs.input_ids
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


def glue_passage_question(bos_token, eos_token, passage, question=None):
    entries = {
        "cls": bos_token,
        "sep": eos_token,
        "passage": passage,
        "question": question,
    }
    if question is not None:
        return "{passage} {sep} {question} ".format(**entries)
    return "{passage} ".format(**entries)


def list_parameters(model):
    parameters = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            parameters[name] = param

    return parameters


def process_race_row(row):
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

    input_str = glue_passage_question(
        tokenizer.bos_token, tokenizer.eos_token, article, question
    )
    output_str = glue_passage_question(tokenizer.bos_token, tokenizer.eos_token, answer)

    return {"inputs": input_str, "outputs": output_str}


# load rouge for validation
rouge = datasets.load_metric("rouge")


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


def create_race_dataset():
    train_dataset = load_dataset("race", "all", split="train")
    train_dataset = train_dataset.map(
        process_race_row,
        remove_columns=["article", "options", "question", "answer", "example_id"],
    )
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

    dev_dataset = load_dataset("race", "all", split="validation")
    dev_dataset = dev_dataset.map(
        process_race_row,
        remove_columns=["article", "options", "question", "answer", "example_id"],
    )
    dev_dataset = dev_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["inputs", "outputs"],
    )
    dev_dataset.set_format(
        type="torch",
        columns=["input_ids", "input_mask", "target_ids", "target_mask", "labels"],
    )

    test_dataset = load_dataset("race", "all", split="test")
    test_dataset = test_dataset.map(
        process_race_row,
        remove_columns=["article", "options", "question", "answer", "example_id"],
    )
    test_dataset = test_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["inputs", "outputs"],
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "input_mask", "target_ids", "target_mask", "labels"],
    )
    return train_dataset, dev_dataset, test_dataset


def build_dream_dataset(args):
    data_rows = []
    with io.open(args.dream_path, encoding="utf-8", mode="r") as fd:
        dream_data = json.load(fd)
        for passage_dict in dream_data:
            text = passage_dict["text"]
            text = " ".join(text.split())
            for q_a_dict in passage_dict["questions_answers"]:
                q = q_a_dict["question"]
                a = q_a_dict["correctAnswers"][0]
                if "__" in q:
                    continue
                q = " ".join(q.split())
                a = " ".join(a.split())
                input_str = glue_passage_question(
                    tokenizer.bos_token, tokenizer.eos_token, text, q
                )
                output_str = glue_passage_question(
                    tokenizer.bos_token, tokenizer.eos_token, a
                )
                data_rows.append({"inputs": input_str, "outputs": output_str})
    df = pd.DataFrame(data_rows)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["inputs", "outputs"],
    )
    dataset.set_format(
        type="torch",
        columns=["input_ids", "input_mask", "target_ids", "target_mask", "labels"],
    )
    return dataset


def main_train(args):
    dataset = build_dream_dataset(args)
    albert2albert = load_pretrained_race(
        path=args.model_path,
        mask_token_id=tokenizer.mask_token_id,
        source_max_length=source_max_length,
        decoder_max_length=decoder_max_length,
    )

    albert2albert = albert2albert.to("cuda:0")
    # instantiate trainer
    training_args = TrainingArguments(
        output_dir="./main_trained_models/",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        do_train=True,
        do_eval=False,
        seed=10,
        gradient_accumulation_steps=1,
        # max_steps=500,
        logging_steps=10,  # set to 1000 for full training
        save_steps=10,  # set to 500 for full training
        eval_steps=10,  # set to 8000 for full training
        warmup_steps=10,  # set to 2000 for full training
        overwrite_output_dir=True,
        save_total_limit=10,
        num_train_epochs=10,
    )

    trainer = Trainer(
        model=albert2albert,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    trainer.train()


def main_predict(args):
    dataset = build_dream_dataset(args)
    albert2albert = load_pretrained_race(
        path=args.model_path,
        mask_token_id=tokenizer.mask_token_id,
        source_max_length=source_max_length,
        decoder_max_length=decoder_max_length,
    )

    albert2albert = albert2albert.to("cuda:0")

    def generate_batch(batch):
        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        input_ids = input_ids.to("cuda:0")
        input_mask = input_mask.to("cuda:0")
        predictions = albert2albert.greedy_decode(
            input_ids=input_ids, input_mask=input_mask
        )

        # all special tokens including will be removed
        predictions_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        input_str = tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        target_str = tokenizer.batch_decode(
            batch["target_ids"], skip_special_tokens=True
        )
        output_batch = {
            "predictions_str": predictions_str,
            "input_str": input_str,
            "target_str": target_str,
        }
        return output_batch

    # instantiate trainer for prediction
    eval_predictions = dataset.map(generate_batch, batched=True, batch_size=batch_size)

    eval_predictions.set_format(
        type="pandas", columns=["input_str", "predictions_str", "target_str"]
    )
    eval_predictions["predictions_str"].to_csv(
        args.prediction_file, sep="\t", encoding="utf-8"
    )
    eval_predictions["input_str"].to_csv(
        args.prediction_file + ".input.csv", sep="\t", encoding="utf-8"
    )
    eval_predictions["target_str"].to_csv(
        args.prediction_file + ".target.csv", sep="\t", encoding="utf-8"
    )


def race_pretrain():
    """main model to train."""
    train_dataset, dev_dataset, test_dataset = create_race_dataset()

    albert2albert = load_albert_encoder_decoder(
        mask_token_id=tokenizer.mask_token_id,
        source_max_length=source_max_length,
        decoder_max_length=decoder_max_length,
    )

    albert2albert = albert2albert.to("cuda:0")

    # instantiate trainer
    training_args = TrainingArguments(
        output_dir="./trained_models/",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        do_train=true,
        do_eval=true,
        seed=10,
        gradient_accumulation_steps=1,
        # max_steps=500,
        logging_steps=100,  # set to 1000 for full training
        save_steps=100,  # set to 500 for full training
        eval_steps=100,  # set to 8000 for full training
        warmup_steps=1000,  # set to 2000 for full training
        overwrite_output_dir=true,
        save_total_limit=3,
        num_train_epochs=3,
    )

    trainer = trainer(
        model=albert2albert,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )
    trainer.train()


def race_predict(args):
    """Main model to train."""
    train_dataset, dev_dataset, test_dataset = create_race_dataset()

    albert2albert = load_pretrained_race(
        path=args.model_path,
        mask_token_id=tokenizer.mask_token_id,
        source_max_length=source_max_length,
        decoder_max_length=decoder_max_length,
    )

    albert2albert = albert2albert.to("cuda:0")

    test_dataset = train_dataset.select(range(16))

    def generate_batch(batch):
        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        input_ids = input_ids.to("cuda:0")
        input_mask = input_mask.to("cuda:0")
        predictions = albert2albert.greedy_decode(
            input_ids=input_ids, input_mask=input_mask
        )

        # all special tokens including will be removed
        predictions_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        input_str = tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        target_str = tokenizer.batch_decode(
            batch["target_ids"], skip_special_tokens=True
        )
        output_batch = {
            "predictions_str": predictions_str,
            "input_str": input_str,
            "target_str": target_str,
        }
        return output_batch

    # instantiate trainer for prediction
    eval_predictions = test_dataset.map(
        generate_batch, batched=True, batch_size=batch_size
    )

    eval_predictions.set_format(
        type="pandas", columns=["input_str", "predictions_str", "target_str"]
    )
    eval_predictions["predictions_str"].to_csv(
        args.prediction_file, sep="\t", encoding="utf-8"
    )
    eval_predictions["input_str"].to_csv(
        args.prediction_file + ".input.csv", sep="\t", encoding="utf-8"
    )
    eval_predictions["target_str"].to_csv(
        args.prediction_file + ".target.csv", sep="\t", encoding="utf-8"
    )
    # test_predictions = predictor.predict(test_dataset)


def run_main(args):
    """Decides what to do in the code."""
    if args.mode == "race_pretrain":
        race_pretrain()
    if args.mode == "race_predict":
        race_predict(args)
    if args.mode == "main_train":
        main_train(args)
    if args.mode == "main_predict":
        main_predict(args)


def argument_parser():
    """augments arguments for protein-gene model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="race_pretrain | race_predict | main_train | main_predict",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path for saving or loading models.",
    )

    parser.add_argument(
        "--prediction_file", type=str, help="file for saving predictions"
    )

    parser.add_argument(
        "--dream_path", type=str, help="path for reading dream scape data!"
    )
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    run_main(args)
