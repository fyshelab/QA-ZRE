import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def white_space_fix(text):
    return " ".join(text.split())


def q_read_narrative_dataset():
    """Read the narrative qa dataset for question generation"""

    def process_narrative_row(row):
        """Helper functions for NarrativeQA Dataset."""
        answer = random.choice(row["answers"])["text"]

        question = row["question"]["text"]

        article = row["document"]["summary"]["text"]

        context = "answer: " + answer + " context: " + article + " </s>"

        return {
            "article": white_space_fix(context),
            "answer": white_space_fix(question + " </s>"),
        }

    train_dataset = load_dataset("narrativeqa", split="train")
    dev_dataset = load_dataset("narrativeqa", split="validation")
    test_dataset = load_dataset("narrativeqa", split="test")

    train_dataset = train_dataset.map(
        process_narrative_row,
        remove_columns=["document", "answers", "question"],
    )

    dev_dataset = dev_dataset.map(
        process_narrative_row,
        remove_columns=["document", "answers", "question"],
    )

    test_dataset = test_dataset.map(
        process_narrative_row,
        remove_columns=["document", "answers", "question"],
    )
    return train_dataset, dev_dataset, test_dataset


def q_read_squad_dataset():
    def process_squad_row(row):
        context = row["context"]
        question = row["question"]
        if row["answers"]["text"]:
            answ = random.choice(row["answers"]["text"])
            return {
                "article": white_space_fix(
                    "answer: " + answ + " context: " + context + " </s>"
                ),
                "answer": white_space_fix(question + " </s>"),
            }

    train_dataset = load_dataset("squad_v2", split="train")
    train_dataset = train_dataset.map(
        process_squad_row,
        remove_columns=["id", "title", "context", "question", "answers"],
    )
    dev_dataset = load_dataset("squad_v2", split="validation")
    dev_dataset = dev_dataset.map(
        process_squad_row,
        remove_columns=["id", "title", "context", "question", "answers"],
    )
    return train_dataset, dev_dataset, dev_dataset


def q_read_drop_dataset():
    def process_drop_row(row):
        context = row["passage"]
        question = row["question"]
        if row["answers_spans"]["spans"]:
            answ = random.choice(row["answers_spans"]["spans"])
            return {
                "article": white_space_fix(
                    "answer: " + answ + " context: " + context + " </s>"
                ),
                "answer": white_space_fix(question + " </s>"),
            }

    train_dataset = load_dataset("drop", split="train")
    train_dataset = train_dataset.map(
        process_drop_row,
        remove_columns=[
            "passage",
            "question",
            "answers_spans",
        ],
    )
    dev_dataset = load_dataset("drop", split="validation")
    dev_dataset = dev_dataset.map(
        process_drop_row,
        remove_columns=[
            "passage",
            "question",
            "answers_spans",
        ],
    )
    return train_dataset, dev_dataset, dev_dataset


def create_question_dataset(
    tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    distributed=False,
    num_workers=0,
):
    """Function to mix and create the train/dev dataset for pytorch model."""

    def dataset_to_pytorch(train_dataset, dev_dataset, test_dataset):
        def process_data_to_model_inputs(batch):
            """Tokenize the inputs and labels."""
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
            batch["target_attention_mask"] = outputs.attention_mask
            batch["labels"] = outputs.input_ids

            # We have to make sure that the PAD token is ignored, -100 is being ignored in the loss function.

            labels = [
                [-100 if token == tokenizer.pad_token_id else token for token in labels]
                for labels in batch["labels"]
            ]
            batch["labels"] = labels

            return batch

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
                "target_attention_mask",
                "labels",
            ],
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
                "target_attention_mask",
                "labels",
            ],
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
                "target_attention_mask",
                "labels",
            ],
        )
        return train_dataset, dev_dataset, test_dataset

    drp_train_dataset, drp_dev_dataset, drp_test_dataset = q_read_drop_dataset()
    nq_train_dataset, nq_dev_dataset, nq_test_dataset = q_read_narrative_dataset()
    sq_train_dataset, sq_dev_dataset, sq_test_dataset = q_read_squad_dataset()

    drp_train_dataset, drp_dev_dataset, drp_test_dataset = dataset_to_pytorch(
        drp_train_dataset, drp_dev_dataset, drp_test_dataset
    )
    nq_train_dataset, nq_dev_dataset, nq_test_dataset = dataset_to_pytorch(
        nq_train_dataset, nq_dev_dataset, nq_test_dataset
    )
    sq_train_dataset, sq_dev_dataset, sq_test_dataset = dataset_to_pytorch(
        sq_train_dataset, sq_dev_dataset, sq_test_dataset
    )

    train_dataset = torch.utils.data.ConcatDataset(
        [
            drp_train_dataset,
            nq_train_dataset,
            sq_train_dataset,
        ]
    )
    dev_dataset = torch.utils.data.ConcatDataset(
        [
            drp_dev_dataset,
            nq_dev_dataset,
            sq_dev_dataset,
        ]
    )
    test_dataset = torch.utils.data.ConcatDataset(
        [
            drp_test_dataset,
            nq_test_dataset,
            sq_test_dataset,
        ]
    )

    # Training
    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
        )

    if not distributed:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        dev_dataset,
        test_dataset,
        train_sampler,
    )
