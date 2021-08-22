# from http://nlp.cs.washington.edu/zeroshot/evaluate.py
import codecs
import re
import string
import sys
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from docopt import docopt
from torch.utils.data import DataLoader

from src.nq_utils import white_space_fix


def zero_re_eval(test_file, answer_file):
    q_aprf = read_results(test_file, answer_file)
    return pretify(q_aprf)


def read_results(test_set, answer_file):
    with codecs.open(test_set, "r", "utf-8") as fin:
        data = [line.strip().split("\t") for line in fin]
    metadata = [x[:4] for x in data]
    gold = [set(x[4:]) for x in data]

    new_gold = []
    for g in gold:
        if g:
            new_gold.append(g)
    # new_gold = gold

    with codecs.open(answer_file, "r", "utf-8") as fin:
        answers = [line.strip() for line in fin]

    new_answers = []
    # ignore the header in answers
    for answer in answers[1:]:
        if answer != "no_answer":
            new_answers.append(answer)
        else:
            new_answers.append("")

    telemetry = []
    for m, g, a in zip(metadata, new_gold, new_answers):
        stats = score(g, a)
        telemetry.append([m[0], m[1], str(len(g) > 0), stats])
    return aprf(telemetry)


def parse_no_answers(results):
    p_answer = [
        a for i, a in sorted([(int(i), a) for i, a in results[0]["scores"].items()])
    ]
    p_no_answer = [
        a for i, a in sorted([(int(i), a) for i, a in results[0]["na"].items()])
    ]

    import numpy as np

    return [answer > no_answer for answer, no_answer in zip(p_answer, p_no_answer)]


def gb(collection, keyfunc):
    return [(k, list(g)) for k, g in groupby(sorted(collection, key=keyfunc), keyfunc)]


def aprf(g):
    tp, tn, sys_pos, real_pos = sum(map(lambda x: x[-1], g))
    total = len(g)
    # a = float(tp + tn) / total
    # nr = tn / float(total - real_pos)
    # npr = tn / float(total - sys_pos)
    if tp == 0:
        p = r = f = 0.0
    else:
        p = tp / float(sys_pos)
        r = tp / float(real_pos)
        f = 2 * p * r / (p + r)
    # return np.array((a, p, r, f, npr, nr))
    return np.array((p, r, f))


def score(gold, answer):
    if len(gold) > 0:
        gold = set.union(*[simplify(g) for g in gold])
    answer = simplify(answer)
    result = np.zeros(4)
    if answer == gold:
        if len(gold) > 0:
            result[0] += 1
        else:
            result[1] += 1
    if len(answer) > 0:
        result[2] += 1
    if len(gold) > 0:
        result[3] += 1
    return result


def simplify(answer):
    return set(
        "".join(c for c in t if c not in PUNCTUATION)
        for t in answer.strip().lower().split()
    ) - {"the", "a", "an", "and", ""}


def pretify(results):
    return " \t ".join(
        [
            ": ".join((k, v))
            for k, v in zip(
                ["Precision", "Recall", "F1"],
                map(lambda r: "{0:.2f}%".format(r * 100), results),
            )
        ]
    )


PUNCTUATION = set(string.punctuation)


def read_zero_re_gold_qa(path, concat=False):
    path = Path(path)
    with open(path, "r") as fd:
        contexts = []
        answers = []
        for line in fd:
            line = line.strip()
            line_arr = line.split("\t")
            if concat:
                gold_question = line_arr[2] + " " + line_arr[0]
            else:
                gold_question = line_arr[1].replace("XXX", " " + line_arr[2] + " ")
            passage = line_arr[3]
            if len(line_arr) > 4:
                gold_answers = line_arr[4:]
            else:
                gold_answers = ["no_answer"]
            contexts.append(
                "question: "
                + white_space_fix(gold_question)
                + " context: "
                + white_space_fix(passage)
                + " </s>"
            )
            answers.append(white_space_fix(" and ".join(gold_answers)) + " </s>")
    return contexts, answers


def read_zero_re_qa(path):
    path = Path(path)
    with open(path, "r") as fd:
        contexts = []
        answers = []
        passages = []
        for line in fd:
            line = line.strip()
            line_arr = line.split("\t")
            passage = line_arr[3]
            if len(line_arr) > 4:
                gold_answers = line_arr[4:]
            else:
                # ignore the unknown answers.
                continue
                # gold_answers = ["no_answer"]
            passages.append(passage)
            contexts.append(
                "answer: "
                + white_space_fix(line_arr[2])
                + " <SEP> "
                + white_space_fix(line_arr[0])
                + " context: "
                + white_space_fix(passage)
                + " </s>"
            )
            answers.append(white_space_fix(" and ".join(gold_answers)) + " </s>")
    return passages, contexts, answers


def create_zero_re_qa_dataset(
    question_tokenizer,
    answer_tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    train_file="./zero-shot-extraction/relation_splits/train.0",
    dev_file="./zero-shot-extraction/relation_splits/dev.0",
    distributed=True,
    num_workers=1,
):
    """Function to create the zero re qa dataset."""
    train_passages, train_contexts, train_answers = read_zero_re_qa(train_file)
    val_passages, val_contexts, val_answers = read_zero_re_qa(dev_file)

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

    # because HuggingFace automatically shifts the labels, the labels correspond exactly to `target_ids`.
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

    # because Huggingface automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored.

    val_labels = [
        [-100 if token == answer_tokenizer.pad_token_id else token for token in labels]
        for labels in val_encodings["second_entity_labels"]
    ]
    val_encodings["second_entity_labels"] = val_labels

    class HelperDataset(torch.utils.data.Dataset):
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

    train_dataset = HelperDataset(train_encodings)
    val_dataset = HelperDataset(val_encodings)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, train_dataset, val_dataset, train_sampler
    if not distributed:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, train_dataset, val_dataset, None


def create_zero_re_gold_qa_dataset(
    tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    train_file="./zero-shot-extraction/relation_splits/train.0",
    dev_file="./zero-shot-extraction/relation_splits/dev.0",
    concat=False,
    distributed=False,
    num_workers=1,
):
    """Function to create the zero re gold qa dataset."""
    train_contexts, train_answers = read_zero_re_gold_qa(train_file, concat=concat)
    val_contexts, val_answers = read_zero_re_gold_qa(dev_file, concat=concat)

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

    # because HuggingFace automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored

    train_labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in train_encodings["labels"]
    ]
    train_encodings["labels"] = train_labels

    val_encodings["target_attention_mask"] = val_answer_encodings.attention_mask

    val_encodings["labels"] = val_answer_encodings.input_ids

    # because HuggingFace automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored.

    val_labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in val_encodings["labels"]
    ]
    val_encodings["labels"] = val_labels

    class HelperDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings.input_ids)

    train_dataset = HelperDataset(train_encodings)
    val_dataset = HelperDataset(val_encodings)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, train_dataset, val_dataset, train_sampler
    if not distributed:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, train_dataset, val_dataset
