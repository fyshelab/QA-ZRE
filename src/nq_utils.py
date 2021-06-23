import random
import re

from datasets import load_dataset
from torch.utils.data import DataLoader


def white_space_fix(text):
    return " ".join(text.split())


def create_reverse_narrative_dataset(
    tokenizer, batch_size, source_max_length, decoder_max_length
):
    """Function to create the narrative dataset."""

    def process_reverse_narrative_row(row):
        """Helper function."""
        answer = random.choice(row["answers"])["text"]
        answer = white_space_fix(answer)

        question = row["question"]["text"]
        question = white_space_fix(question)

        article = row["document"]["summary"]["text"]
        article = white_space_fix(article)

        context = "answer: " + answer + " context: " + article + " </s>"
        # context = question + " \n " + article
        # context = context.lower()
        # context = re.sub("'(.*)'", r"\1", context)
        return {
            "article": context,
            "answer": question + " </s>",
        }

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

        batch["question_input_ids"] = inputs.input_ids
        batch["question_attention_mask"] = inputs.attention_mask
        batch["question_target_attention_mask"] = outputs.attention_mask
        batch["question_labels"] = outputs.input_ids.copy()

        # We have to make sure that the PAD token is ignored, -100 is being ignored in the loss function.

        labels = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["question_labels"]
        ]
        batch["question_labels"] = labels

        return batch

    train_dataset = load_dataset("narrativeqa", split="train")
    dev_dataset = load_dataset("narrativeqa", split="validation")
    test_dataset = load_dataset("narrativeqa", split="test")

    train_dataset = train_dataset.map(
        process_reverse_narrative_row,
        remove_columns=["document", "answers"],
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
            "question_input_ids",
            "question_attention_mask",
            "question_target_attention_mask",
            "question_labels",
        ],
    )

    dev_dataset = dev_dataset.map(
        process_reverse_narrative_row,
        remove_columns=["document", "answers"],
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
            "question_input_ids",
            "question_attention_mask",
            "question_target_attention_mask",
            "question_labels",
        ],
    )
    test_dataset = test_dataset.map(
        process_reverse_narrative_row,
        remove_columns=["document", "answers"],
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
            "question_input_ids",
            "question_attention_mask",
            "question_target_attention_mask",
            "question_labels",
        ],
    )

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
    )


def create_narrative_dataset(
    tokenizer, batch_size, source_max_length, decoder_max_length
):
    """Function to create the narrative dataset."""

    def process_narrative_row(row):
        """Helper function."""
        answer = random.choice(row["answers"])["text"]
        answer = white_space_fix(answer)

        question = row["question"]["text"]
        question = white_space_fix(question)

        article = row["document"]["summary"]["text"]
        article = white_space_fix(article)

        context = "question: " + question + " context: " + article + " </s>"
        # context = question + " \n " + article
        # context = context.lower()
        # context = re.sub("'(.*)'", r"\1", context)
        return {
            "article": context,
            "answer": answer + " </s>",
        }

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

    train_dataset = load_dataset("narrativeqa", split="train")
    dev_dataset = load_dataset("narrativeqa", split="validation")
    test_dataset = load_dataset("narrativeqa", split="test")

    train_dataset = train_dataset.map(
        process_narrative_row,
        remove_columns=["document", "answers"],
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
            "target_attention_mask",
            "labels",
        ],
    )

    dev_dataset = dev_dataset.map(
        process_narrative_row,
        remove_columns=["document", "answers"],
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
        process_narrative_row,
        remove_columns=["document", "answers"],
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
    )
