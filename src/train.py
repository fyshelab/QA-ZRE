from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import torch
import torch.optim as optim
from datasets import load_dataset
from transformers import AlbertTokenizer, Trainer, TrainingArguments

from src.albert_model import load_albert_encoder_decoder

# set the tokenizer
tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v2")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
max_length = 256
batch_size = 1


def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["inputs"], padding="max_length", truncation=True, max_length=max_length
    )
    outputs = tokenizer(
        batch["outputs"], padding="max_length", truncation=True, max_length=max_length
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
        return "{passage} {sep} {question}".format(**entries)
    return "{passage}".format(**entries)


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


train_dataset, dev_dataset, test_dataset = create_race_dataset()

albert2albert = load_albert_encoder_decoder(mask_token_id=tokenizer.mask_token_id)

albert2albert = albert2albert.to("cuda:0")

# instantiate trainer
training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    do_train=True,
    do_eval=True,
    logging_steps=2,  # set to 1000 for full training
    save_steps=16,  # set to 500 for full training
    eval_steps=4,  # set to 8000 for full training
    warmup_steps=1,  # set to 2000 for full training
    max_steps=16,  # delete for full training
    overwrite_output_dir=True,
    save_total_limit=3,
)

trainer = Trainer(
    model=albert2albert,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dev_dataset,
    eval_dataset=dev_dataset,
)
trainer.train()
