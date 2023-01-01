import json
import random
from pathlib import Path
from random import sample
from typing import Dict, List, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def white_space_fix(text):
    return " ".join(text.split())


def remove_prefix(text, prefix):
    """This function is used to remove prefix key from the text."""
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever


class QADataset(Dataset):
    """Subclass the pytorch's Dataset to build my own dataset for the question-
    response pre-training."""

    def __init__(self, data: Dict[str, Union[List[int], List[List[int]]]]) -> None:
        """store the reference to the tokenized data."""
        self.data = data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the elements for example index 'idx' as a dictionary with
        tensor values."""
        return {key: torch.tensor(val[idx]) for key, val in self.data.items()}

    def __len__(self) -> int:
        """Return the length of the data."""
        return len(self.data["input_ids"])


class ZREQADataset(Dataset):
    """Subclass the pytorch's Dataset to build my own dataset for the question-
    response fine-tuning on the relation extraction datasets."""

    def __init__(self, data: Dict[str, Union[List[int], List[List[int]]]]) -> None:
        """store the reference to the tokenized data."""
        self.data = data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the elements for example index 'idx' as a dictionary with
        tensor values."""
        row = {}
        for key, val in self.data.items():
            if key in ["passages", "entity_relations"]:
                row[key] = val[idx]
            else:
                row[key] = torch.tensor(val[idx])
        return row

    def __len__(self) -> int:
        """Return the length of the data."""
        return len(self.data["entity_relation_passage_input_ids"])


def read_fewrl_dataset(
    fewrel_file, relation_descriptions_file, seed=10, val_split_size=5
):
    path = Path(relation_descriptions_file)

    id_to_desc = {}
    id_to_label = {}
    label_to_id = {}
    with open(path, "r") as fd:
        re_desc_data = json.load(fd)
        for row in re_desc_data:
            re_id = row["relation_id"]
            id_to_desc[re_id] = row["relation_description"]
            id_to_label[re_id] = row["relation_label"]
            label_to_id[row["relation_label"]] = re_id

    train_contexts = []
    train_posterier_contexts = []
    train_answers = []
    train_passages = []
    train_entity_relations = []

    val_contexts = []
    val_posterier_contexts = []
    val_answers = []
    val_passages = []
    val_entity_relations = []
    val_actual_ids = []

    test_contexts = []
    test_posterier_contexts = []
    test_answers = []
    test_passages = []
    test_entity_relations = []
    test_actual_ids = []

    with open(fewrel_file, "r") as json_file:
        data = json.load(json_file)
        r_ids = list(data.keys())
        random.shuffle(r_ids)
        val_r_ids = r_ids[:val_split_size]
        test_r_ids = r_ids[val_split_size : 4 * val_split_size]
        train_r_ids = r_ids[4 * val_split_size :]

        train_id_df = pd.DataFrame(train_r_ids, columns=["relation_ids"])
        train_id_df.to_csv(
            f"/tmp/train_ids_{seed}.csv", sep=",", header=True, index=False
        )

        val_id_df = pd.DataFrame(val_r_ids, columns=["relation_ids"])
        val_id_df.to_csv(f"/tmp/val_ids_{seed}.csv", sep=",", header=True, index=False)

        test_id_df = pd.DataFrame(test_r_ids, columns=["relation_ids"])
        test_id_df.to_csv(
            f"/tmp/test_ids_{seed}.csv", sep=",", header=True, index=False
        )

        for r_id in val_r_ids:
            r_name = id_to_label[r_id]
            r_desc = id_to_desc[r_id]
            # validate on a smaller dev data for faster computation. Sample 50 sentences per relation.
            for sent in sample(data[r_id], 50):
                sentence = " ".join(sent["tokens"])
                head_entity_indices = sent["h"][2][0]
                tail_entity_indices = sent["t"][2][0]
                head_entity = " ".join([sent["tokens"][i] for i in head_entity_indices])
                tail_entity = " ".join([sent["tokens"][i] for i in tail_entity_indices])
                gold_answers = tail_entity
                for second_r_id in val_r_ids:
                    second_r_name = id_to_label[second_r_id]
                    second_r_desc = id_to_desc[second_r_id]
                    val_actual_ids.append(r_id)
                    val_passages.append(sentence)
                    val_entity_relations.append(
                        white_space_fix(head_entity + " <SEP> " + second_r_name)
                    )
                    val_contexts.append(
                        "answer: "
                        + white_space_fix(head_entity)
                        + " <SEP> "
                        + white_space_fix(second_r_name)
                        + " ; "
                        + white_space_fix(second_r_desc)
                        + " context: "
                        + white_space_fix(sentence)
                        + " </s>"
                    )
                    val_posterier_contexts.append(
                        "answer: "
                        + white_space_fix(head_entity)
                        + " <SEP> "
                        + white_space_fix(second_r_name)
                        + " ; "
                        + white_space_fix(second_r_desc)
                        + " "
                        + white_space_fix(gold_answers)
                        + " context: "
                        + white_space_fix(sentence)
                        + " </s>"
                    )
                    val_answers.append(white_space_fix(gold_answers) + " </s>")

        for r_id in test_r_ids:
            r_name = id_to_label[r_id]
            r_desc = id_to_desc[r_id]
            for sent in data[r_id]:
                sentence = " ".join(sent["tokens"])
                head_entity_indices = sent["h"][2][0]
                tail_entity_indices = sent["t"][2][0]
                head_entity = " ".join([sent["tokens"][i] for i in head_entity_indices])
                tail_entity = " ".join([sent["tokens"][i] for i in tail_entity_indices])
                gold_answers = tail_entity
                for second_r_id in test_r_ids:
                    second_r_name = id_to_label[second_r_id]
                    second_r_desc = id_to_desc[second_r_id]
                    test_actual_ids.append(r_id)
                    test_passages.append(sentence)
                    test_entity_relations.append(
                        white_space_fix(head_entity + " <SEP> " + second_r_name)
                    )
                    test_contexts.append(
                        "answer: "
                        + white_space_fix(head_entity)
                        + " <SEP> "
                        + white_space_fix(second_r_name)
                        + " ; "
                        + white_space_fix(second_r_desc)
                        + " context: "
                        + white_space_fix(sentence)
                        + " </s>"
                    )
                    test_posterier_contexts.append(
                        "answer: "
                        + white_space_fix(head_entity)
                        + " <SEP> "
                        + white_space_fix(second_r_name)
                        + " ; "
                        + white_space_fix(second_r_desc)
                        + " "
                        + white_space_fix(gold_answers)
                        + " context: "
                        + white_space_fix(sentence)
                        + " </s>"
                    )
                    test_answers.append(white_space_fix(gold_answers) + " </s>")

        for r_id in train_r_ids:
            r_name = id_to_label[r_id]
            r_desc = id_to_desc[r_id]

            for sent in data[r_id]:
                sentence = " ".join(sent["tokens"])
                head_entity_indices = sent["h"][2][0]
                tail_entity_indices = sent["t"][2][0]
                head_entity = " ".join([sent["tokens"][i] for i in head_entity_indices])
                tail_entity = " ".join([sent["tokens"][i] for i in tail_entity_indices])
                gold_answers = tail_entity
                train_passages.append(sentence)
                train_entity_relations.append(
                    white_space_fix(head_entity + " <SEP> " + r_name)
                )
                train_contexts.append(
                    "answer: "
                    + white_space_fix(head_entity)
                    + " <SEP> "
                    + white_space_fix(r_name)
                    + " ; "
                    + white_space_fix(r_desc)
                    + " context: "
                    + white_space_fix(sentence)
                    + " </s>"
                )
                train_posterier_contexts.append(
                    "answer: "
                    + white_space_fix(head_entity)
                    + " <SEP> "
                    + white_space_fix(r_name)
                    + " ; "
                    + white_space_fix(r_desc)
                    + " "
                    + white_space_fix(gold_answers)
                    + " context: "
                    + white_space_fix(sentence)
                    + " </s>"
                )
                train_answers.append(white_space_fix(gold_answers) + " </s>")

                # Add the negative example.
                temp_ids = list(train_r_ids)
                temp_ids.remove(r_id)
                other_r_id = random.sample(temp_ids, 1)[0]
                other_r_name = id_to_label[other_r_id]
                other_r_desc = id_to_desc[other_r_id]

                train_passages.append(sentence)
                train_entity_relations.append(
                    white_space_fix(head_entity + " <SEP> " + other_r_name)
                )
                train_contexts.append(
                    "answer: "
                    + white_space_fix(head_entity)
                    + " <SEP> "
                    + white_space_fix(other_r_name)
                    + " ; "
                    + white_space_fix(other_r_desc)
                    + " context: "
                    + white_space_fix(sentence)
                    + " </s>"
                )
                train_posterier_contexts.append(
                    "answer: "
                    + white_space_fix(head_entity)
                    + " <SEP> "
                    + white_space_fix(other_r_name)
                    + " ; "
                    + white_space_fix(other_r_desc)
                    + " "
                    + white_space_fix("no_answer")
                    + " context: "
                    + white_space_fix(sentence)
                    + " </s>"
                )
                train_answers.append(white_space_fix("no_answer") + " </s>")

    train_df = pd.DataFrame(
        {
            "passages": train_passages,
            "contexts": train_contexts,
            "answers": train_answers,
            "entity_relations": train_entity_relations,
            "posterier_contexts": train_posterier_contexts,
        }
    )

    val_df = pd.DataFrame(
        {
            "passages": val_passages,
            "contexts": val_contexts,
            "answers": val_answers,
            "entity_relations": val_entity_relations,
            "posterier_contexts": val_posterier_contexts,
            "actual_ids": val_actual_ids,
        }
    )

    test_df = pd.DataFrame(
        {
            "passages": test_passages,
            "contexts": test_contexts,
            "answers": test_answers,
            "entity_relations": test_entity_relations,
            "posterier_contexts": test_posterier_contexts,
            "actual_ids": test_actual_ids,
        }
    )

    train_df.to_csv(f"/tmp/train_data_{seed}.csv", sep=",", header=True, index=False)
    val_df.to_csv(f"/tmp/val_data_{seed}.csv", sep=",", header=True, index=False)
    test_df.to_csv(f"/tmp/test_data_{seed}.csv", sep=",", header=True, index=False)

    return train_df, val_df, test_df


def create_fewrl_dataset(
    question_tokenizer,
    answer_tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    dataframe,
    shuffle=True,
):
    """Function to create the fewrl dataset for fine-tuning the response and
    question generator."""

    passages = dataframe["passages"].tolist()
    contexts = dataframe["contexts"].tolist()
    answers = dataframe["answers"].tolist()
    entity_relations = dataframe["entity_relations"].tolist()
    posterier_contexts = dataframe["posterier_contexts"].tolist()

    input_encodings = question_tokenizer(
        contexts,
        truncation=True,
        padding="max_length",
        max_length=source_max_length,
        add_special_tokens=False,
    )
    output_encodings = answer_tokenizer(
        answers,
        truncation=True,
        padding="max_length",
        max_length=decoder_max_length,
        add_special_tokens=False,
    )
    posterier_encodings = question_tokenizer(
        posterier_contexts,
        truncation=True,
        padding="max_length",
        max_length=source_max_length,
        add_special_tokens=False,
    )
    data = {
        "passages": passages,
        "entity_relations": entity_relations,
        "posterier_input_ids": posterier_encodings.input_ids,
        "posterier_attention_mask": posterier_encodings.attention_mask,
        "entity_relation_passage_input_ids": input_encodings.input_ids,
        "entity_relation_passage_attention_mask": input_encodings.attention_mask,
        "second_entity_labels": output_encodings.input_ids,
        "second_entity_attention_mask": output_encodings.attention_mask,
    }
    dataloader = DataLoader(ZREQADataset(data), batch_size=batch_size, shuffle=shuffle)
    return dataloader
