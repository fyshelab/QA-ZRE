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
import gensim
import gensim.corpora as corpora
import numpy as np
import pandas as pd
import spacy
import torch
import torch.distributed as dist
import torch.utils.data.distributed
from datasets import load_dataset
from spacy.lang.en.stop_words import STOP_WORDS
from torch.utils.data import DataLoader

from src.nq_utils import (create_narrative_dataset,
                          create_reverse_narrative_dataset)
from src.re_qa_model import REQA, HyperParameters, save


def question_read_squad(path):
    path = Path(path)
    with open(path, "rb") as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    for group in squad_dict["data"]:
        for passage in group["paragraphs"]:
            context = passage["context"]
            for qa in passage["qas"]:
                question = qa["question"]
                questions.append(question)
                contexts.append(context)
    return contexts, questions


def white_space_fix(text):
    return " ".join(text.split())


def read_docred(path, rel_info_path):
    path = Path(path)
    rel_info_path = Path(rel_info_path)

    with open(path, "rb") as fd:
        docred_dict = json.load(fd)

    with open(rel_info_path, "rb") as fd:
        rel_info_dict = json.load(fd)

    passages = []
    contexts = []
    answers = []
    for group in docred_dict:
        paragraph = ""
        for sent in group["sents"]:
            paragraph += " ".join(sent) + " "
        for relation in group["labels"]:
            head_entity = random.choice(group["vertexSet"][relation["h"]])["name"]
            tail_entity = random.choice(group["vertexSet"][relation["t"]])["name"]
            relation_type = relation["r"]
            passages.append(white_space_fix(paragraph))
            contexts.append(
                "answer: "
                + white_space_fix(head_entity)
                + " <SEP> "
                + white_space_fix(rel_info_dict[relation_type])
                + " context: "
                + white_space_fix(paragraph)
                + " </s>"
            )
            answers.append(white_space_fix(tail_entity) + " </s>")
    return passages, contexts, answers


def create_docred_dataset(
    answer_tokenizer,
    question_tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    distributed=False,
    num_workers=0,
    rank=0,
):
    """Function to create the docred dataset."""
    train_passages, train_contexts, train_answers = read_docred(
        "./docred/train_annotated.json", "./docred/rel_info.json"
    )
    val_passages, val_contexts, val_answers = read_docred(
        "./docred/dev.json", "./docred/rel_info.json"
    )
    """
    test_passages, test_contexts, test_answers = read_docred(
        "./docred/test.json", "./docred/rel_info.json"
    )
    """

    val_encodings = question_tokenizer(
        val_contexts,
        truncation=True,
        padding="max_length",
        max_length=source_max_length,
        add_special_tokens=False,
    )
    val_answer_encodings = answer_tokenizer(
        val_answers,
        truncation=True,
        padding="max_length",
        max_length=decoder_max_length,
        add_special_tokens=False,
    )

    train_encodings = question_tokenizer(
        train_contexts,
        truncation=True,
        padding="max_length",
        max_length=source_max_length,
        add_special_tokens=False,
    )
    train_answer_encodings = answer_tokenizer(
        train_answers,
        truncation=True,
        padding="max_length",
        max_length=decoder_max_length,
        add_special_tokens=False,
    )
    """
    test_encodings = question_tokenizer(
        test_contexts,
        truncation=True,
        padding="max_length",
        max_length=source_max_length,
        add_special_tokens=False,
    )
    test_answer_encodings = answer_tokenizer(
        test_answers,
        truncation=True,
        padding="max_length",
        max_length=decoder_max_length,
        add_special_tokens=False,
    )
    """
    train_encodings["passages"] = train_passages
    train_encodings["entity_relation_passage_input_ids"] = train_encodings.pop(
        "input_ids"
    )
    train_encodings["entity_relation_passage_attention_mask"] = train_encodings.pop(
        "attention_mask"
    )

    train_encodings["second_entity_labels"] = train_answer_encodings.pop("input_ids")
    train_encodings["second_entity_attention_mask"] = train_answer_encodings.pop(
        "attention_mask"
    )

    # because BERT automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored

    train_labels = [
        [-100 if token == answer_tokenizer.pad_token_id else token for token in labels]
        for labels in train_encodings["second_entity_labels"]
    ]
    train_encodings["second_entity_labels"] = train_labels

    val_encodings["passages"] = val_passages
    val_encodings["entity_relation_passage_input_ids"] = val_encodings.pop("input_ids")
    val_encodings["entity_relation_passage_attention_mask"] = val_encodings.pop(
        "attention_mask"
    )

    val_encodings["second_entity_labels"] = val_answer_encodings.pop("input_ids")
    val_encodings["second_entity_attention_mask"] = val_answer_encodings.pop(
        "attention_mask"
    )

    # because BERT automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored.

    val_labels = [
        [-100 if token == answer_tokenizer.pad_token_id else token for token in labels]
        for labels in val_encodings["second_entity_labels"]
    ]
    val_encodings["second_entity_labels"] = val_labels

    """
    test_encodings["passages"] = test_passages
    test_encodings["entity_relation_passage_input_ids"] = test_encodings.pop(
        "input_ids"
    )
    test_encodings["entity_relation_passage_attention_mask"] = test_encodings.pop(
        "attention_mask"
    )

    test_encodings["second_entity_labels"] = test_answer_encodings.pop("input_ids")
    test_encodings["second_entity_attention_mask"] = test_answer_encodings.pop(
        "attention_mask"
    )

    # because BERT automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored.

    test_labels = [
        [-100 if token == answer_tokenizer.pad_token_id else token for token in labels]
        for labels in test_encodings["second_entity_labels"]
    ]
    test_encodings["second_entity_labels"] = test_labels
    """

    class DocRedDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            row = {}
            for key, val in self.encodings.items():
                if key == "passages":
                    row[key] = val[idx]
                else:
                    row[key] = torch.tensor(val[idx])
            return row

        def __len__(self):
            return len(self.encodings.entity_relation_passage_input_ids)

    train_dataset = DocRedDataset(train_encodings)
    val_dataset = DocRedDataset(val_encodings)
    # test_dataset = SquadDataset(test_encodings)

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

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_sampler,
        train_loader,
        val_loader,
        None,
        # test_loader,
        train_dataset,
        val_dataset,
        None
        # test_dataset,
    )


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
                        "question: "
                        + white_space_fix(question)
                        + " context: "
                        + white_space_fix(context)
                        + " </s>"
                    )
                    answers.append(white_space_fix(answ["text"]) + " </s>")
                else:
                    contexts.append(
                        "question: "
                        + white_space_fix(question)
                        + " context: "
                        + white_space_fix(context)
                        + " </s>"
                    )
                    answers.append("no no answer </s>")

    return contexts, answers


def read_reverse_squad(path):
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
                        "answer: "
                        + white_space_fix(answ["text"])
                        + " context: "
                        + white_space_fix(context)
                        + " </s>"
                    )
                    answers.append(white_space_fix(question) + " </s>")
                else:
                    contexts.append(
                        "answer: "
                        + "no no answer"
                        + " context: "
                        + white_space_fix(context)
                        + " </s>"
                    )
                    answers.append(white_space_fix(question) + " </s>")

    return contexts, answers


def create_race_dataset(
    tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    distributed=False,
    num_workers=0,
    rank=0,
):
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

        answer = white_space_fix(row["options"][option_idx])

        question = white_space_fix(row["question"])

        article = white_space_fix(row["article"])

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

        batch["target_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids

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
            "target_attention_mask",
            "labels",
        ],
    )

    # Training
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
    )


def run_train_epoch(
    model, train_dataloader, current_device, phase="answer"
) -> Generator:
    """Train the model and return the loss for 'num_steps' given the
    'batch_size' and the train_dataset.

    Randomly pick a batch from the train_dataset.
    """
    step = 0
    answer_train_loader, question_train_loader, docred_train_loader = train_dataloader
    if phase == "answer":
        answer_iter = iter(answer_train_loader)
        for main_batch in docred_train_loader:
            answer_batch = next(answer_iter)
            main_batch.update(answer_batch)
            loss_values = model.module.train_step(
                main_batch, current_device, phase="answer"
            )
            step += 1
            yield step, loss_values["loss_value"]
    elif phase == "question":
        question_iter = iter(question_train_loader)
        for main_batch in docred_train_loader:
            answer_batch = next(question_iter)
            main_batch.update(answer_batch)
            loss_values = model.module.train_step(
                main_batch, current_device, phase="question"
            )
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
            for ret_row in model.module.predict_step(batch):
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
    rank=0,
    train_samplers=None,
    current_device=0,
) -> None:
    """Run the model on input data (for training or testing)"""

    model_path = config.model_path
    max_epochs = config.max_epochs
    mode = config.mode
    if mode == "train":
        print("\nRank: {0} | INFO: ML training\n".format(rank))
        first_start = time.time()
        epoch = 0
        question_inner_loop = config.question_training_steps
        answer_inner_loop = config.answer_training_steps
        while epoch < max_epochs:
            # let all processes sync up before starting with a new epoch of training
            dist.barrier()

            # make sure we get different orderings.
            for sampler in train_samplers:
                sampler.set_epoch(epoch)

            print("\nRank: {0} | Epoch:{1}\n".format(rank, epoch))
            start = time.time()
            for question_loop in range(question_inner_loop):
                total_loss = []
                print(
                    "\rRank: {0} | Info: Question Phase Training {1} | GPU Usage: {2}\n".format(
                        rank,
                        question_loop,
                        torch.cuda.memory_allocated(device=current_device),
                    )
                )
                for step, loss in run_train_epoch(
                    model, train_dataloader, current_device, phase="question"
                ):

                    if loss:
                        total_loss.append(loss)

                    if total_loss:
                        mean_loss = np.mean(total_loss)

                    else:
                        mean_loss = float("-inf")

                    print(
                        "\rRank: {0} | Batch:{1} | Loss:{2} | Mean Loss:{3} | GPU Usage: {4}\n".format(
                            rank,
                            step,
                            loss,
                            mean_loss,
                            torch.cuda.memory_allocated(device=current_device),
                        )
                    )
                    if rank == 0 and save_always and (step % 200 == 0):
                        save(
                            model.module.question_model,
                            model.module.model_path,
                            str(epoch) + "_question_step_" + str(step),
                        )

                    if save_always and (step % 200 == 0):
                        dist.barrier()

                if rank == 0 and save_always:
                    save(
                        model.module.question_model,
                        model.module.model_path,
                        str(epoch) + "_question_loop_" + str(question_loop),
                    )
                dist.barrier()

            dist.barrier()

            for answer_loop in range(answer_inner_loop):
                total_loss = []
                print(
                    "\rRank: {0} | Info: Answer Phase Training {1} | GPU Usage: {2}\n".format(
                        rank,
                        answer_loop,
                        torch.cuda.memory_allocated(device=current_device),
                    )
                )
                for step, loss in run_train_epoch(
                    model, train_dataloader, current_device, phase="answer"
                ):
                    if loss:
                        total_loss.append(loss)

                    if total_loss:
                        mean_loss = np.mean(total_loss)

                    else:
                        mean_loss = float("-inf")

                    print(
                        "\rRank: {0} | Batch:{1} | Loss:{2} | Mean Loss:{3} | GPU Usage: {4}\n".format(
                            rank,
                            step,
                            loss,
                            mean_loss,
                            torch.cuda.memory_allocated(device=current_device),
                        )
                    )
                    if rank == 0 and save_always and (step % 200 == 0):
                        save(
                            model.module.answer_model,
                            model.module.model_path,
                            str(epoch) + "_answer_step_" + str(step),
                        )

                    if save_always and (step % 200 == 0):
                        dist.barrier()

                if rank == 0 and save_always:
                    save(
                        model.module.answer_model,
                        model.module.model_path,
                        str(epoch) + "_answer_loop_" + str(answer_loop),
                    )
                dist.barrier()

            msg = "\nRank: {0} | Epoch training time: {1} seconds\n".format(
                rank, time.time() - start
            )
            print(msg)
            epoch += 1

        if rank == 0:
            save_config(config, model_path)
        msg = "\nRank: {0} | Total training time: {1} seconds\n".format(
            rank, time.time() - first_start
        )
        print(msg)

    elif mode == "test":
        print("Predicting...")
        start = time.time()
        run_predict(model, test_dataloader, config.prediction_file)
        msg = "\nTotal prediction time:{} seconds\n".format(time.time() - start)
        print(msg)


def read_squad_for_question_generation():
    train_contexts, train_questions = question_read_squad("./squad/train-v2.0.json")
    nlp = spacy.load("en_core_web_sm")
    # Add these interrogative words into the stop lists.
    stop_list = [
        "what",
        "for",
        "when",
        "what for",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "why don't",
        "how",
        "how far",
        "how long",
        "how much",
        "how many",
        "how old",
        "how come",
    ]

    STOP_WORDS.extend(stop_list)

    for word in STOP_WORDS:
        lexeme = nlp.vocab[word]
        lexeme.is_stop = True

    good_qs = []
    for index, q in enumerate(train_questions):
        q_doc = nlp(q)
        new_q_doc = [token.lemma_ for token in q_doc if token.lemma_ != "-PRON-"]
        new_q = u" ".join(new_q_doc)
        doc_to_remove_stop = nlp.make_doc(new_q)
        final_doc = [
            token.text
            for token in doc_to_remove_stop
            if token.is_stop != True and token.is_punct != True
        ]
        q_ents = [(e.text, e.label_) for e in q_doc.ents]
        # to collect high precision data.
        if len(q_ents) == 1:
            new_final_doc = []
            for token in final_doc:
                if token not in q_ents[0][0]:
                    new_final_doc.append(token)
            if new_final_doc:
                good_qs.append(
                    (train_contexts[index], q, " ".join(new_final_doc), q_ents[0][0])
                )

    contexts = []
    questions = []
    for row in good_qs:
        context = row[0]
        question = row[1]
        tokens = row[2].split(" ")
        if len(tokens) > 0 and len(tokens) < 4:
            relation_signal = " ".join(tokens)

        if len(tokens) >= 4:
            token_num = random.randint(1, 3)
            sampled_tokens = random.sample(tokens, token_num)
            relation_signal = " ".join(sampled_tokens)
        else:
            continue
        contexts.append(
            "answer: "
            + white_space_fix(row[3])
            + " <SEP> "
            + white_space_fix(relation_signal)
            + " context: "
            + white_space_fix(context)
            + " </s>"
        )
        questions.append(white_space_fix(question) + " </s>")

    return contexts, questions


def create_squad_dataset(
    tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    distributed=False,
    num_workers=0,
    rank=0,
):
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

    train_encodings["target_attention_mask"] = train_answer_encodings.attention_mask

    train_encodings["labels"] = train_answer_encodings.input_ids

    # because BERT automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored

    train_labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in train_encodings["labels"]
    ]
    train_encodings["labels"] = train_labels

    val_encodings["target_attention_mask"] = val_answer_encodings.attention_mask

    val_encodings["labels"] = val_answer_encodings.input_ids

    # because BERT automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored.

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

    # Training
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

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset, val_dataset


def create_rev_squad_dataset(
    tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    distributed=False,
    num_workers=0,
    rank=0,
):
    """Function to create the squad dataset."""
    train_contexts, train_answers = read_reverse_squad("./squad/train-v2.0.json")
    val_contexts, val_answers = read_reverse_squad("./squad/dev-v2.0.json")

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

    train_encodings["question_input_ids"] = train_encodings.pop("input_ids")
    train_encodings["question_attention_mask"] = train_encodings.pop("attention_mask")
    train_encodings[
        "question_target_attention_mask"
    ] = train_answer_encodings.attention_mask

    train_encodings["question_labels"] = train_answer_encodings.input_ids

    # because BERT automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored

    train_labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in train_encodings["question_labels"]
    ]
    train_encodings["question_labels"] = train_labels

    val_encodings["question_input_ids"] = val_encodings.pop("input_ids")
    val_encodings["question_attention_mask"] = val_encodings.pop("attention_mask")
    val_encodings[
        "question_target_attention_mask"
    ] = val_answer_encodings.attention_mask

    val_encodings["question_labels"] = val_answer_encodings.input_ids

    # because BERT automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored.

    val_labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in val_encodings["question_labels"]
    ]
    val_encodings["question_labels"] = val_labels

    class SquadDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings.question_input_ids)

    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)

    # Training
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

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset, val_dataset


def create_all_relation_qa_dataset(
    answer_tokenizer,
    question_tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    distributed=False,
    num_workers=0,
    rank=0,
):
    (
        squad_train_loader,
        squad_val_loader,
        squad_train_dataset,
        squad_val_dataset,
    ) = create_squad_dataset(
        answer_tokenizer,
        batch_size,
        source_max_length,
        decoder_max_length,
        distributed,
        num_workers,
        rank,
    )
    (
        rev_squad_train_loader,
        rev_squad_val_loader,
        rev_squad_train_dataset,
        rev_squad_val_dataset,
    ) = create_rev_squad_dataset(
        question_tokenizer,
        batch_size,
        source_max_length,
        decoder_max_length,
        distributed,
        num_workers,
        rank,
    )
    (
        nq_train_loader,
        nq_val_loader,
        nq_test_loader,
        nq_train_dataset,
        nq_dev_dataset,
        nq_test_dataset,
    ) = create_narrative_dataset(
        answer_tokenizer,
        batch_size,
        source_max_length,
        decoder_max_length,
        distributed,
        num_workers,
        rank,
    )
    (
        rev_nq_train_loader,
        rev_nq_val_loader,
        rev_nq_test_loader,
        rev_nq_train_dataset,
        rev_nq_dev_dataset,
        rev_nq_test_dataset,
    ) = create_reverse_narrative_dataset(
        question_tokenizer,
        batch_size,
        source_max_length,
        decoder_max_length,
        distributed,
        num_workers,
        rank,
    )
    (
        race_train_loader,
        race_val_loader,
        race_test_loader,
        race_train_dataset,
        race_dev_dataset,
        race_test_dataset,
    ) = create_race_dataset(
        answer_tokenizer,
        batch_size,
        source_max_length,
        decoder_max_length,
        distributed,
        num_workers,
        rank,
    )
    (
        docred_train_sampler,
        docred_train_loader,
        docred_val_loader,
        docred_test_loader,
        docred_train_dataset,
        docred_dev_dataset,
        docred_test_dataset,
    ) = create_docred_dataset(
        answer_tokenizer,
        question_tokenizer,
        batch_size,
        source_max_length,
        decoder_max_length,
        distributed,
        num_workers,
        rank,
    )

    answer_train_datasets = torch.utils.data.ConcatDataset(
        [squad_train_dataset, nq_train_dataset, race_train_dataset]
    )
    answer_eval_datasets = torch.utils.data.ConcatDataset(
        [squad_val_dataset, nq_dev_dataset, race_dev_dataset]
    )

    question_train_datasets = torch.utils.data.ConcatDataset(
        [rev_squad_train_dataset, rev_nq_train_dataset]
    )
    question_eval_datasets = torch.utils.data.ConcatDataset(
        [rev_squad_val_dataset, rev_nq_dev_dataset]
    )

    # Training
    answer_train_sampler = None
    if distributed:
        answer_train_sampler = torch.utils.data.distributed.DistributedSampler(
            answer_train_datasets
        )
        answer_train_loader = DataLoader(
            answer_train_datasets,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=answer_train_sampler,
        )

    if not distributed:
        answer_train_loader = DataLoader(
            answer_train_datasets, batch_size=batch_size, shuffle=True
        )

    answer_eval_loader = DataLoader(
        answer_eval_datasets, batch_size=batch_size, shuffle=False
    )

    # Training
    question_train_sampler = None
    if distributed:
        question_train_sampler = torch.utils.data.distributed.DistributedSampler(
            question_train_datasets
        )
        question_train_loader = DataLoader(
            question_train_datasets,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=question_train_sampler,
        )

    if not distributed:
        question_train_loader = DataLoader(
            question_train_datasets, batch_size=batch_size, shuffle=True
        )

    question_eval_loader = DataLoader(
        question_eval_datasets, batch_size=batch_size, shuffle=False
    )
    return (
        (answer_train_sampler, question_train_sampler, docred_train_sampler),
        (answer_train_loader, question_train_loader, docred_train_loader),
        (
            answer_eval_loader,
            question_eval_loader,
            docred_val_loader,
        ),
    )


def run_reqa(args):
    """Run the relation-extraction qa models."""

    # distributed code coppied from the compute canada guidelines:
    # https://docs.computecanada.ca/wiki/PyTorch

    ngpus_per_node = torch.cuda.device_count()

    """ This next line is the key to getting DistributedDataParallel working on SLURM:
		SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
 		current process inside a node and is also 0 or 1 in this example."""

    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank

    """ This next block parses CUDA_VISIBLE_DEVICES to find out which GPUs have been allocated to the job, then sets torch.device to the GPU corresponding       to the local rank (local rank 0 gets the first GPU, local rank 1 gets the second GPU etc) """

    available_gpus = list(os.environ.get("CUDA_VISIBLE_DEVICES").replace(",", ""))

    current_device = int(available_gpus[local_rank])

    torch.cuda.set_device(current_device)

    print("From Rank: {}, ==> Initializing Process Group...".format(rank))
    # init the process group
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.init_method,
        world_size=args.world_size,
        rank=rank,
    )
    print("process group ready!")

    print("From Rank: {}, ==> Making model..".format(rank))

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
        answer_checkpoint=args.answer_checkpoint,
        question_checkpoint=args.question_checkpoint,
        question_training_steps=args.question_training_steps,
        answer_training_steps=args.answer_training_steps,
        num_beams=args.num_beams,
        num_beam_groups=args.num_beam_groups,
        beam_diversity_penalty=args.beam_diversity_penalty,
    )
    model = REQA(config)
    model = model.to(current_device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[current_device]
    )

    print("From Rank: {}, ==> Preparing data..".format(rank))

    train_samplers, train_loaders, val_loaders = create_all_relation_qa_dataset(
        answer_tokenizer=model.module.answer_tokenizer,
        question_tokenizer=model.module.question_tokenizer,
        batch_size=config.batch_size // args.world_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
        distributed=True,
        num_workers=args.num_workers,
        rank=rank,
    )

    run_model(
        model,
        config=config,
        train_dataloader=train_loaders,
        dev_dataloader=val_loaders,
        test_dataloader=None,
        save_always=True,
        rank=rank,
        train_samplers=train_samplers,
        current_device=current_device,
    )


def run_main(args):
    """Decides what to do in the code."""
    if args.mode in ["reqa_train", "reqa_test"]:
        run_reqa(args)


def argument_parser():
    """augments arguments for model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="reqa_train | reqa_test",
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
    parser.add_argument("--num_beams", type=int, default=32, help="Number of beam size")
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=4,
        help="Number of beam groups for diverse beam.",
    )
    parser.add_argument(
        "--beam_diversity_penalty",
        type=float,
        default=0.5,
        help="Diversity penalty in diverse beam.",
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
        "--answer_checkpoint", type=str, help="checkpoint of the trained answer model."
    )
    parser.add_argument(
        "--question_checkpoint",
        type=str,
        help="checkpoint of the trained question model.",
    )
    parser.add_argument(
        "--answer_training_steps",
        type=int,
        help="number of epochs to train the answer model compared to the question model.",
    )
    parser.add_argument(
        "--question_training_steps",
        type=int,
        help="number of epochs to train the question model compared to the answer model.",
    )
    parser.add_argument(
        "--init_method",
        default="tcp://127.0.0.1:3456",
        type=str,
        help="I guess the address of the master",
    )
    parser.add_argument("--dist-backend", default="gloo", type=str, help="")
    parser.add_argument("--world_size", default=1, type=int, help="")
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="number of sub processes per main process of gpu to load data",
    )
    parser.add_argument("--distributed", action="store_true", help="")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    run_main(args)
