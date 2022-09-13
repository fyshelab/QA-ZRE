import json
import random
from pathlib import Path
from re import I


import pandas as pd
import torch
# from datasets import load_dataset
from torch.utils.data import DataLoader

from random import sample
from src.re_qa_model import set_random_seed


def white_space_fix(text):
    return " ".join(text.split())

def find_sub_list(sl, l):
    i = 0
    while i <= len(l) - len(sl):
        sub_arr = l[i : i + len(sl)]
        check = True
        for j in range(len(sub_arr)):
            if sl[j] not in sub_arr[j]:
                check = False
        if check:
            return list(range(i, i + len(sl), 1))
        i += 1
    return []

def find_fewrel_ids(path):
    with open(path, "r") as json_file:
        data = json.load(json_file)
        r_ids = set(list(data.keys()))

    id_to_desc = {}
    with open("./relation_descriptions.json", "r") as fd:
        re_desc_data = json.load(fd)
        for row in re_desc_data:
            re_id = row["relation_id"]
            id_to_desc[re_id] = row["relation_description"]

    ids_not_found = []
    for id in r_ids:
        if id not in id_to_desc:
            ids_not_found.append(id)
    
    return ids_not_found

def find_wikizsl_ids(path):
    with open(path, "r") as json_file:
        training_data = json.load(json_file)

    all_keys = set()
    for i in training_data:
        label = i['edgeSet'][0]['kbID']
        if label not in all_keys:
            all_keys.add(label)

    id_to_desc = {}
    with open("./relation_descriptions.json", "r") as fd:
        re_desc_data = json.load(fd)
        for row in re_desc_data:
            re_id = row["relation_id"]
            id_to_desc[re_id] = row["relation_description"]

    ids_not_found = []
    for id in all_keys:
        if id not in id_to_desc:
            ids_not_found.append(id)
    
    return ids_not_found

def find_all_relation_ids_in_reqa(path):
    label_to_id = {}

    rel_desc_dict = {}
    with open("./props.json", "r") as fd:
        re_desc_data = json.load(fd)
        sentence_delimiters = [". ", ".\n", "? ", "?\n", "! ", "!\n"]
        for row in re_desc_data:
            desc = row["description"]
            if desc == {}:
                continue
            desc = desc.strip(".") + ". "
            pos = [desc.find(delimiter) for delimiter in sentence_delimiters]
            pos = min([p for p in pos if p >= 0])
            re_desc = desc[:pos]
            rel_desc_dict[row["id"]] = white_space_fix(re_desc)

    rel_dict = {}
    with open(path, "r") as fd:
        for line in fd:
            line = line.strip()
            line_arr = line.split("\t")
            rel_token = line_arr[0]
            if rel_token in label_to_id:
                rel_dict[rel_token] = label_to_id[rel_token]
            else:
                rel_dict[rel_token] = "none"
    return rel_dict, rel_desc_dict

import ujson
from pathlib import Path

def read_jsonl(file_path):
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
    with Path(file_path).open('r', encoding='utf8') as f:
        for line in f:
            try:  # hack to handle broken jsonl
                yield ujson.loads(line.strip())
            except ValueError:
                continue


def write_jsonl(file_path, lines):
    """Create a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
    Path(file_path).open('w', encoding='utf-8').write('\n'.join(data))

def convert_fewrel_to_promptZRE_format(path, output_path):
    path = Path(path)

    id_to_label = {}
    with open("./relation_descriptions.json", "r") as fd:
        re_desc_data = json.load(fd)
        for row in re_desc_data:
            re_label = row["relation_label"]
            re_id = row["relation_id"]
            id_to_label[re_id] = re_label

    with open(path, "r") as fd:
        fewrel_data = json.load(fd)
        data = []
        for k, v in fewrel_data.items():
            for i in v:
                i['relation'] = k
                data_row = {
                    "triplets" : [{
                        "tokens": i["tokens"],
                        "head": i["h"][2][0],
                        "tail": i["t"][2][0],
                        "label_id": i["relation"],
                        "label": id_to_label[i["relation"]]
                    }]
                }
                data.append(data_row)
    write_jsonl(output_path, data)

def convert_reqa_to_fewrel_format(path, output_path):
    path = Path(path)

    label_to_id = {}
    with open("./relation_descriptions.json", "r") as fd:
        re_desc_data = json.load(fd)
        for row in re_desc_data:
            re_label = row["relation_label"]
            re_id = row["relation_id"]
            label_to_id[re_label] = re_id

    output_json = {}
    with open(output_path, "w") as out_fd:
        with open(path, "r") as fd:
            index = -1
            for line in fd:
                index += 1
                line = line.strip()
                line_arr = line.split("\t")
                passage = line_arr[3]
                h = line_arr[2]
                if line_arr[4:]:
                    t = line_arr[4:][0]
                else:
                    continue
                rel_token = line_arr[0]
                rel_id = label_to_id.get(rel_token)
                list_update = output_json.get(rel_id, [])
                tokens = white_space_fix(passage).split(" ")
                h_tokens = white_space_fix(h).split(" ")
                t_tokens = white_space_fix(t).split(" ")
                h_indices = find_sub_list(h_tokens, tokens)
                t_indices = find_sub_list(t_tokens, tokens)
                if len(tokens) > 160:
                    print("skipped a long example")
                    continue
                ret_row = {
                    "example_index": index,
                    "tokens": tokens,
                    "h": [h, "dummy_id", [h_indices]],
                    "t": [t, "dummy_id", [t_indices]],
                }
                list_update.append(ret_row)
                output_json[rel_id] = list_update
        json.dump(output_json, out_fd)

def read_re_qa_relation_data_prompt_format(path, train=False):
    """Create data for the UnifiedQA model with a prompt format."""
    path = Path(path)

    rel_dict = {}
    with open("./props.json", "r") as fd:
        re_desc_data = json.load(fd)
        sentence_delimiters = [". ", ".\n", "? ", "?\n", "! ", "!\n"]
        for row in re_desc_data:
            desc = row["description"]
            if desc == {}:
                continue
            desc = desc.strip(".") + ". "
            pos = [desc.find(delimiter) for delimiter in sentence_delimiters]
            pos = min([p for p in pos if p >= 0])
            re_desc = desc[:pos]
            re_id = row["label"]
            rel_dict[white_space_fix(re_id).lower()] = white_space_fix(re_desc)

    all_relations = {}
    with open(path, "r") as fd:
        for line in fd:
            line = line.strip()
            line_arr = line.split("\t")
            if line_arr[0] not in all_relations:
                all_relations[line_arr[0]] = {line_arr[1]}
            else:
                all_relations[line_arr[0]].add(line_arr[1])

    with open(path, "r") as fd:
        contexts = []
        answers = []
        for line in fd:
            line = line.strip()
            line_arr = line.split("\t")
            passage = line_arr[3]
            if len(line_arr) > 4:
                gold_answers = line_arr[4:]
                head_entity = line_arr[2]
                tail_entity = " and ".join(gold_answers)
                prompt = head_entity + " to " + tail_entity + " ? \\n "
                if not train:
                    for rel_class in all_relations.keys():
                        prompt += "(R)" + " " + rel_class + " "

                    prompt += "\\n " + passage
                else:
                    gold_relation = line_arr[0]
                    # sample 9 other relation types.
                    all_relations_list = list(all_relations.keys())
                    all_relations_list.remove(gold_relation)
                    sampled_relations = random.sample(all_relations_list, k=9)
                    sampled_relations.append(gold_relation)
                    random.shuffle(sampled_relations)
                    for rel_class in sampled_relations:
                        prompt += "(R)" + " " + rel_class + " "

                    prompt += "\\n " + passage

            else:
                # negative example.
                continue
            
            gold_relation = line_arr[0]
            contexts.append(white_space_fix(prompt + " </s>"))
            answers.append(white_space_fix(gold_relation + " </s>"))

    data_df = pd.DataFrame(
            {
                "contexts": contexts,
                "answers": answers,
            }
    )
    data_df.to_csv(str(path) + ".prompt_data.csv", sep=",", header=True, index=False)
    return contexts, answers


def read_gold_re_qa_relation_data(path, concat=False, for_question_generation=False):
    """Create val data for relation classification considering all the data and
    gold_templates."""
    path = Path(path)

    label_to_desc = {}
    with open("./relation_descriptions.json", "r") as fd:
        re_desc_data = json.load(fd)
        for row in re_desc_data:
            re_label = row["relation_label"]
            label_to_desc[re_label] = row["relation_description"]

    all_relations = {}
    with open(path, "r") as fd:
        for line in fd:
            line = line.strip()
            line_arr = line.split("\t")
            if line_arr[0] not in all_relations:
                all_relations[line_arr[0]] = {line_arr[1]}
            else:
                all_relations[line_arr[0]].add(line_arr[1])

    with open(path, "r") as fd:
        contexts = []
        posterier_contexts = []
        answers = []
        passages = []
        entities = []
        entity_relations = []
        correct_indices = []
        rel_types = []
        for line in fd:
            line = line.strip()
            line_arr = line.split("\t")
            passage = line_arr[3]
            for rel_type in all_relations.keys():
                if concat:
                    gold_question = line_arr[2] + " <SEP> " + rel_type
                else:
                    gold_template = next(iter(all_relations[rel_type]))
                    gold_question = gold_template.replace(
                        "XXX", " " + line_arr[2] + " "
                    )
                if len(line_arr) > 4:
                    gold_answers = line_arr[4:]
                    if rel_type == line_arr[0]:
                        correct_indices.append(True)
                    else:
                        correct_indices.append(False)
                else:
                    continue
                rel_types.append(rel_type)
                passages.append(passage)
                entity_relations.append(white_space_fix(line_arr[2] + " " + rel_type))
                entities.append(white_space_fix(line_arr[2]))
                if for_question_generation:
                    if rel_type in label_to_desc:
                        contexts.append(
                            "answer: "
                            + white_space_fix(line_arr[2])
                            + " <SEP> "
                            + white_space_fix(rel_type)
                            + " ; "
                            + label_to_desc[rel_type]
                            + " context: "
                            + white_space_fix(passage)
                            + " </s>"
                        )
                        posterier_contexts.append(
                            "answer: "
                            + white_space_fix(line_arr[2])
                            + " <SEP> "
                            + white_space_fix(rel_type)
                            + " ; "
                            + label_to_desc[rel_type]
                            + " "
                            + white_space_fix(" and ".join(gold_answers))
                            + " context: "
                            + white_space_fix(passage)
                            + " </s>"
                        )
                else:
                    contexts.append(
                        "question: "
                        + white_space_fix(gold_question)
                        + " context: "
                        + white_space_fix(passage)
                        + " </s>"
                    )
                answers.append(white_space_fix(" and ".join(gold_answers)) + " </s>")

    if for_question_generation:
        data_df = pd.DataFrame(
            {
                "passages": passages,
                "contexts": contexts,
                "posterier_contexts": posterier_contexts,
                "answers": answers,
                "entity_relations": entity_relations,
                "entities": entities,
                "correct_indices": correct_indices,
                "rel_types": rel_types,
            }
        )
    else:
        data_df = pd.DataFrame(
            {
                "passages": passages,
                "contexts": contexts,
                "answers": answers,
                "entity_relations": entity_relations,
                "entities": entities,
                "correct_indices": correct_indices,
                "rel_types": rel_types,
            }
        )
    if concat:
        data_df.to_csv(
            str(path) + ".concat.relation_data.csv", sep=",", header=True, index=False
        )
    elif for_question_generation:
        data_df.to_csv(
            str(path) + ".qq.relation_data.csv", sep=",", header=True, index=False
        )
    else:
        data_df.to_csv(
            str(path) + ".relation_data.csv", sep=",", header=True, index=False
        )
    return passages, contexts, posterier_contexts, answers, entity_relations, entities


def create_zero_re_qa_gold_dataset(
    question_tokenizer,
    answer_tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    file=None,
    concat=False,
):
    """Function to create the zero re qa dataset."""
    (
        val_passages,
        val_contexts,
        val_posterier_contexts,
        val_answers,
        val_entity_relations,
        _,
    ) = read_gold_re_qa_relation_data(file, concat=concat)

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

    val_encodings["target_attention_mask"] = val_answer_encodings.attention_mask

    val_encodings["labels"] = val_answer_encodings.input_ids

    # because HuggingFace automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored.

    val_labels = [
        [-100 if token == answer_tokenizer.pad_token_id else token for token in labels]
        for labels in val_encodings["labels"]
    ]
    val_encodings["labels"] = val_labels

    class HelperDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            row = {}
            for key, val in self.encodings.items():
                if key in ["passages", "entity_relations"]:
                    row[key] = val[idx]
                else:
                    row[key] = torch.tensor(val[idx])
            return row

        def __len__(self):
            if "entity_relation_passage_input_ids" in self.encodings:
                return len(self.encodings.entity_relation_passage_input_ids)
            if "input_ids" in self.encodings:
                return len(self.encodings.input_ids)

    val_dataset = HelperDataset(val_encodings)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader, val_dataset


def read_zero_re_qa(path, ignore_unknowns=False, gold_question=False, concat=False, only_unknowns=False):
    """Main function to read the zero re qa dataset."""
    path = Path(path)

    label_to_desc = {}
    with open("./relation_descriptions.json", "r") as fd:
        re_desc_data = json.load(fd)
        for row in re_desc_data:
            re_label = row["relation_label"]
            label_to_desc[re_label] = row["relation_description"]

    with open(path, "r") as fd:
        uniq_relations = set()
        for line in fd:
            line = line.strip()
            line_arr = line.split("\t")
            uniq_relations.add(white_space_fix(line_arr[0]))

        relation_to_id = {}
        i = 0
        for rel in uniq_relations:
            relation_to_id[rel] = i
            i += 1
        id_to_relation = {id: rel for rel, id in relation_to_id.items()}

    contexts = []
    posterier_contexts = []
    answers = []
    passages = []
    entities = []
    entity_relations = []
    relations = []
    with open(path, "r") as fd:
        for line in fd:
            line = line.strip()
            line_arr = line.split("\t")
            passage = line_arr[3]
            if concat:
                gold_question_str = line_arr[2] + " <SEP> " + line_arr[0]
            elif gold_question:
                gold_question_str = line_arr[1].replace("XXX", " " + line_arr[2] + " ")
            if len(line_arr) > 4:
                gold_answers = line_arr[4:]
                if only_unknowns:
                    continue
            elif ignore_unknowns:
                continue
            else:
                gold_answers = ["no_answer"]
            passages.append(passage)
            entity_relations.append(white_space_fix(line_arr[2] + " " + line_arr[0]))
            entities.append(white_space_fix(line_arr[2]))
            relations.append(relation_to_id[white_space_fix(line_arr[0])])
            if concat or gold_question:
                contexts.append(
                    "question: "
                    + white_space_fix(gold_question_str)
                    + " context: "
                    + white_space_fix(passage)
                    + " </s>"
                )
            else:
                if line_arr[0] in label_to_desc:
                    contexts.append(
                        "answer: "
                        + white_space_fix(line_arr[2])
                        + " <SEP> "
                        + white_space_fix(line_arr[0])
                        + " ; "
                        + label_to_desc[line_arr[0]]
                        + " context: "
                        + white_space_fix(passage)
                        + " </s>"
                    )
                    posterier_contexts.append(
                        "answer: "
                        + white_space_fix(line_arr[2])
                        + " <SEP> "
                        + white_space_fix(line_arr[0])
                        + " ; "
                        + label_to_desc[line_arr[0]]
                        + " "
                        + white_space_fix(" and ".join(gold_answers))
                        + " context: "
                        + white_space_fix(passage)
                        + " </s>"
                    )

            answers.append(white_space_fix(" and ".join(gold_answers)) + " </s>")
    
    data_df = pd.DataFrame(
            {
                "passages": passages,
                "contexts": contexts,
                "answers": answers,
                "entity_relations": entity_relations,
                "entities": entities,
            }
    )
    data_df.to_csv(
            str(path) + ".unknowns.csv", sep=",", header=True, index=False
    )
    return (
        passages,
        contexts,
        answers,
        entity_relations,
        entities,
        posterier_contexts,
        (relations, relation_to_id, id_to_relation),
    )


def create_prompt_zero_re_qa_dataset(
    tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    file_path=None,
    for_evaluation=False,
):
    """Function to create the prompt-based zero re qa dataset."""

    data_df = pd.read_csv(file_path, sep=",")
    data_contexts = data_df["contexts"].tolist()
    data_answers = data_df["answers"].tolist()

    encodings = tokenizer(
        data_contexts,
        truncation=True,
        padding="max_length",
        max_length=source_max_length,
        add_special_tokens=False,
    )
    answer_encodings = tokenizer(
        data_answers,
        truncation=True,
        padding="max_length",
        max_length=decoder_max_length,
        add_special_tokens=False,
    )
    encodings["target_attention_mask"] = answer_encodings.attention_mask
    encodings["labels"] = answer_encodings.input_ids

    # because HuggingFace automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored
    labels = [
        [
            -100 if token == tokenizer.pad_token_id else token
            for token in labels
        ]
        for labels in encodings["labels"]
    ]
    encodings["labels"] = labels

    class HelperDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            row = {}
            for key, val in self.encodings.items():
                row[key] = torch.tensor(val[idx])
            return row

        def __len__(self):
            if "input_ids" in self.encodings:
                return len(self.encodings.input_ids)

    loader = None
    dataset = HelperDataset(encodings)
    if not for_evaluation:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, dataset


def create_zero_re_qa_dataset(
    question_tokenizer,
    answer_tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    train_file=None,
    dev_file=None,
    ignore_unknowns=True,
    concat=False,
    gold_questions=False,
    for_evaluation=False,
):
    """Function to create the zero re qa dataset."""
    if not for_evaluation:
        (
            train_passages,
            train_contexts,
            train_answers,
            train_entity_relations,
            train_entities,
            train_posterier_contexts,
            relation_info,
        ) = read_zero_re_qa(
            train_file,
            ignore_unknowns=ignore_unknowns,
            gold_question=gold_questions,
            concat=concat,
        )
    (
        val_passages,
        val_contexts,
        val_answers,
        val_entity_relations,
        _,
        _,
        _,
    ) = read_zero_re_qa(
        dev_file,
        ignore_unknowns=ignore_unknowns,
        gold_question=gold_questions,
        concat=concat,
    )

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

    if not for_evaluation:
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
        train_entity_encodings = question_tokenizer(
            train_entities,
            truncation=True,
            padding="max_length",
            max_length=decoder_max_length,
            add_special_tokens=False,
        )

        if not (gold_questions or concat):
            train_posterier_encodings = question_tokenizer(
                train_posterier_contexts,
                truncation=True,
                padding="max_length",
                max_length=source_max_length,
                add_special_tokens=False,
            )

    if gold_questions or concat:

        if not for_evaluation:
            train_encodings[
                "target_attention_mask"
            ] = train_answer_encodings.attention_mask

            train_encodings["labels"] = train_answer_encodings.input_ids

            # because HuggingFace automatically shifts the labels, the labels correspond exactly to `target_ids`.
            # We have to make sure that the PAD token is ignored

            train_labels = [
                [
                    -100 if token == answer_tokenizer.pad_token_id else token
                    for token in labels
                ]
                for labels in train_encodings["labels"]
            ]
            train_encodings["labels"] = train_labels

        val_encodings["target_attention_mask"] = val_answer_encodings.attention_mask

        val_encodings["labels"] = val_answer_encodings.input_ids

        # because HuggingFace automatically shifts the labels, the labels correspond exactly to `target_ids`.
        # We have to make sure that the PAD token is ignored.

        val_labels = [
            [
                -100 if token == answer_tokenizer.pad_token_id else token
                for token in labels
            ]
            for labels in val_encodings["labels"]
        ]
        val_encodings["labels"] = val_labels

    else:
        if not for_evaluation:
            train_encodings["passages"] = train_passages
            train_encodings["entity_relations"] = train_entity_relations
            train_encodings["posterier_input_ids"] = train_posterier_encodings.pop(
                "input_ids"
            )
            train_encodings["posterier_attention_mask"] = train_posterier_encodings.pop(
                "attention_mask"
            )
            train_encodings["entity_input_ids"] = train_entity_encodings.pop(
                "input_ids"
            )
            train_encodings["entity_attention_mask"] = train_entity_encodings.pop(
                "attention_mask"
            )

            train_encodings["entity_relation_passage_input_ids"] = train_encodings.pop(
                "input_ids"
            )
            train_encodings[
                "entity_relation_passage_attention_mask"
            ] = train_encodings.pop("attention_mask")

            train_encodings["second_entity_labels"] = train_answer_encodings.pop(
                "input_ids"
            )
            train_encodings[
                "second_entity_attention_mask"
            ] = train_answer_encodings.pop("attention_mask")

            # because HuggingFace automatically shifts the labels, the labels correspond exactly to `target_ids`.
            # We have to make sure that the PAD token is ignored

            train_labels = [
                [
                    -100 if token == answer_tokenizer.pad_token_id else token
                    for token in labels
                ]
                for labels in train_encodings["second_entity_labels"]
            ]
            train_encodings["second_entity_labels"] = train_labels
            train_encodings["relation_labels"] = relation_info[0]

        val_encodings["passages"] = val_passages
        val_encodings["entity_relations"] = val_entity_relations
        val_encodings["entity_relation_passage_input_ids"] = val_encodings.pop(
            "input_ids"
        )
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
            [
                -100 if token == answer_tokenizer.pad_token_id else token
                for token in labels
            ]
            for labels in val_encodings["second_entity_labels"]
        ]
        val_encodings["second_entity_labels"] = val_labels

    class HelperDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            row = {}
            for key, val in self.encodings.items():
                if key in ["passages", "entity_relations"]:
                    row[key] = val[idx]
                else:
                    row[key] = torch.tensor(val[idx])
            return row

        def __len__(self):
            if "entity_relation_passage_input_ids" in self.encodings:
                return len(self.encodings.entity_relation_passage_input_ids)
            if "input_ids" in self.encodings:
                return len(self.encodings.input_ids)

    train_dataset = None
    train_loader = None
    if not for_evaluation:
        train_dataset = HelperDataset(train_encodings)
    val_dataset = HelperDataset(val_encodings)

    if not for_evaluation:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_dataset, val_dataset

def read_wikizsl_dataset(zsl_path, seed=10, m=5):
    rel_names = {}
    rel_descs = {}
    with open("./props.json", "r") as fd:
        re_desc_data = json.load(fd)
        sentence_delimiters = [". ", ".\n", "? ", "?\n", "! ", "!\n"]
        for row in re_desc_data:
            desc = row["description"]
            if desc == {}:
                desc = "no relation description"
            desc = desc.strip(".") + ". "
            pos = [desc.find(delimiter) for delimiter in sentence_delimiters]
            pos = min([p for p in pos if p >= 0])
            re_desc = desc[:pos]
            re_id = row["id"]
            rel_names[re_id] = white_space_fix(row["label"]).lower()
            rel_descs[re_id] = white_space_fix(re_desc)

    set_random_seed(seed)

    train_contexts = []
    train_posterier_contexts = []
    train_answers = []
    train_passages = []
    train_entities = []
    train_entity_relations = []

    val_contexts = []
    val_posterier_contexts = []
    val_answers = []
    val_passages = []
    val_entities = []
    val_entity_relations = []
    val_actual_ids = []

    test_contexts = []
    test_posterier_contexts = []
    test_answers = []
    test_passages = []
    test_entities = []
    test_entity_relations = []
    test_actual_ids = []

    with open(zsl_path, "r") as json_file:
        data = json.load(json_file)
        r_ids = set()
        for row in data:
            relation_id = row["edgeSet"][0]["kbID"]
            r_ids.add(relation_id)

        r_ids = list(r_ids)
        random.shuffle(r_ids)
        val_r_ids = r_ids[:m]
        test_r_ids = r_ids[m : 4 * m]
        train_r_ids = r_ids[4 * m :]

        set_val_r_ids = set(val_r_ids)
        set_test_r_ids = set(test_r_ids)
        set_train_r_ids = set(train_r_ids)

        train_id_df = pd.DataFrame(train_r_ids, columns=["relation_ids"])
        train_id_df.to_csv(
            "./train_ids_" + str(seed) + ".csv", sep=",", header=True, index=False
        )

        val_id_df = pd.DataFrame(val_r_ids, columns=["relation_ids"])
        val_id_df.to_csv(
            "./val_ids_" + str(seed) + ".csv", sep=",", header=True, index=False
        )

        test_id_df = pd.DataFrame(test_r_ids, columns=["relation_ids"])
        test_id_df.to_csv(
            "./test_ids_" + str(seed) + ".csv", sep=",", header=True, index=False
        )

        random.shuffle(data)
        for row in data:
            sentence = " ".join(row["tokens"])
            relation_id = row["edgeSet"][0]["kbID"]
            head_entity = " ".join(
                [row["tokens"][int(i)] for i in row["edgeSet"][0]["left"]]
            )
            tail_entity = " ".join(
                [row["tokens"][int(i)] for i in row["edgeSet"][0]["right"]]
            )
            r_name = rel_names[relation_id]
            re_desc = rel_descs[relation_id]
            gold_answers = [tail_entity]
            if relation_id in set_val_r_ids:
                for second_relation_id in set_val_r_ids:
                    val_actual_ids.append(relation_id)
                    r_name = rel_names[second_relation_id]
                    re_desc = rel_descs[second_relation_id]
                    val_passages.append(sentence)
                    val_entity_relations.append(white_space_fix(head_entity + " <SEP> " + r_name))
                    val_entities.append(white_space_fix(head_entity))
                    val_contexts.append(
                        "answer: "
                        + white_space_fix(head_entity)
                        + " <SEP> "
                        + white_space_fix(r_name)
                        + " ; "
                        + white_space_fix(re_desc)
                        + " context: "
                        + white_space_fix(sentence)
                        + " </s>"
                    )
                    val_posterier_contexts.append(
                        "answer: "
                        + white_space_fix(head_entity)
                        + " <SEP> "
                        + white_space_fix(r_name)
                        + " ; "
                        + white_space_fix(re_desc)
                        + " "
                        + white_space_fix(" and ".join(gold_answers))
                        + " context: "
                        + white_space_fix(sentence)
                        + " </s>"
                    )
                    val_answers.append(
                        white_space_fix(" and ".join(gold_answers)) + " </s>"
                    )

            elif relation_id in set_test_r_ids:
                for second_relation_id in set_test_r_ids:
                    test_actual_ids.append(relation_id)
                    test_passages.append(sentence)
                    r_name = rel_names[second_relation_id]
                    re_desc = rel_descs[second_relation_id]
                    test_entity_relations.append(
                        white_space_fix(head_entity + " <SEP> " + r_name)
                    )
                    test_entities.append(white_space_fix(head_entity))
                    test_contexts.append(
                        "answer: "
                        + white_space_fix(head_entity)
                        + " <SEP> "
                        + white_space_fix(r_name)
                        + " ; "
                        + white_space_fix(re_desc)
                        + " context: "
                        + white_space_fix(sentence)
                        + " </s>"
                    )
                    test_posterier_contexts.append(
                        "answer: "
                        + white_space_fix(head_entity)
                        + " <SEP> "
                        + white_space_fix(r_name)
                        + " ; "
                        + white_space_fix(re_desc)
                        + " "
                        + white_space_fix(" and ".join(gold_answers))
                        + " context: "
                        + white_space_fix(sentence)
                        + " </s>"
                    )
                    test_answers.append(
                        white_space_fix(" and ".join(gold_answers)) + " </s>"
                    )

            elif relation_id in set_train_r_ids:
                train_passages.append(sentence)
                train_entity_relations.append(
                    white_space_fix(head_entity + " <SEP> " + r_name)
                )
                train_entities.append(white_space_fix(head_entity))
                train_contexts.append(
                    "answer: "
                    + white_space_fix(head_entity)
                    + " <SEP> "
                    + white_space_fix(r_name)
                    + " ; "
                    + white_space_fix(re_desc)
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
                    + white_space_fix(re_desc)
                    + " "
                    + white_space_fix(" and ".join(gold_answers))
                    + " context: "
                    + white_space_fix(sentence)
                    + " </s>"
                )
                train_answers.append(
                    white_space_fix(" and ".join(gold_answers)) + " </s>"
                )

    train_df = pd.DataFrame(
        {
            "passages": train_passages,
            "contexts": train_contexts,
            "answers": train_answers,
            "entity_relations": train_entity_relations,
            "entities": train_entities,
            "posterier_contexts": train_posterier_contexts,
        }
    )

    val_df = pd.DataFrame(
        {
            "passages": val_passages,
            "contexts": val_contexts,
            "answers": val_answers,
            "entity_relations": val_entity_relations,
            "entities": val_entities,
            "posterier_contexts": val_posterier_contexts,
            "actual_ids": val_actual_ids
        }
    )

    test_df = pd.DataFrame(
        {
            "passages": test_passages,
            "contexts": test_contexts,
            "answers": test_answers,
            "entity_relations": test_entity_relations,
            "entities": test_entities,
            "posterier_contexts": test_posterier_contexts,
            "actual_ids": test_actual_ids
        }
    )

    train_df.to_csv(
        "./train_data_" + str(seed) + ".csv", sep=",", header=True, index=False
    )
    val_df.to_csv("./val_data_" + str(seed) + ".csv", sep=",", header=True, index=False)
    test_df.to_csv(
        "./test_data_" + str(seed) + ".csv", sep=",", header=True, index=False
    )

    return (
        (
            train_passages,
            train_contexts,
            train_answers,
            train_entity_relations,
            train_entities,
            train_posterier_contexts,
        ),
        (
            val_passages,
            val_contexts,
            val_answers,
            val_entity_relations,
            val_entities,
            val_posterier_contexts,
        ),
        (
            test_passages,
            test_contexts,
            test_answers,
            test_entity_relations,
            test_entities,
            test_posterier_contexts,
        ),
    )


def read_fewrl_dataset(fewrel_path, seed=10, m=5):#, augment_with_unks=True):

    set_random_seed(seed)

    path = Path("./relation_descriptions.json")

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

    '''
    if augment_with_unks:
        unk_relation_ids = {}
        with open("./zero-shot-extraction/relation_splits/train.0", "r") as fd:
            for line in fd:
                line = line.strip()
                line_arr = line.split("\t")
                if len(line_arr) <= 4:
                    # negative example
                    if label_to_id[line_arr[0]] not in unk_relation_ids:
                        unk_relation_ids[label_to_id[line_arr[0]]] = [line_arr]
                    else:
                        unk_relation_ids[label_to_id[line_arr[0]]].append(line_arr)

        with open("./zero-shot-extraction/relation_splits/dev.0", "r") as fd:
            for line in fd:
                line = line.strip()
                line_arr = line.split("\t")
                if len(line_arr) <= 4:
                    # negative example
                    if label_to_id[line_arr[0]] not in unk_relation_ids:
                        unk_relation_ids[label_to_id[line_arr[0]]] = [line_arr]
                    else:
                        unk_relation_ids[label_to_id[line_arr[0]]].append(line_arr)

        with open("./zero-shot-extraction/relation_splits/test.0", "r") as fd:
            for line in fd:
                line = line.strip()
                line_arr = line.split("\t")
                if len(line_arr) <= 4:
                    # negative example
                    if label_to_id[line_arr[0]] not in unk_relation_ids:
                        unk_relation_ids[label_to_id[line_arr[0]]] = [line_arr]
                    else:
                        unk_relation_ids[label_to_id[line_arr[0]]].append(line_arr)
    '''

    train_contexts = []
    train_posterier_contexts = []
    train_answers = []
    train_passages = []
    train_entities = []
    train_entity_relations = []

    val_contexts = []
    val_posterier_contexts = []
    val_answers = []
    val_passages = []
    val_entities = []
    val_entity_relations = []
    val_actual_ids = []

    test_contexts = []
    test_posterier_contexts = []
    test_answers = []
    test_passages = []
    test_entities = []
    test_entity_relations = []
    test_actual_ids = []

    with open(fewrel_path, "r") as json_file:
        data = json.load(json_file)
        r_ids = list(data.keys())
        random.shuffle(r_ids)
        val_r_ids = r_ids[: m]
        test_r_ids = r_ids[m: 4 * m]
        train_r_ids = r_ids[4 * m:]

        train_id_df = pd.DataFrame(train_r_ids, columns=["relation_ids"])
        train_id_df.to_csv(
            "./train_ids_" + str(seed) + ".csv", sep=",", header=True, index=False
        )

        val_id_df = pd.DataFrame(val_r_ids, columns=["relation_ids"])
        val_id_df.to_csv(
            "./val_ids_" + str(seed) + ".csv", sep=",", header=True, index=False
        )

        test_id_df = pd.DataFrame(test_r_ids, columns=["relation_ids"])
        test_id_df.to_csv(
            "./test_ids_" + str(seed) + ".csv", sep=",", header=True, index=False
        )

        for r_id in val_r_ids:
            r_name = id_to_label[r_id]
            r_desc = id_to_desc[r_id]

            # validate on a smaller dev data for faster computation. Sample 50 sentences per relation.
            for sent in sample(data[r_id], 50):
                for second_r_id in val_r_ids:
                    second_r_name = id_to_label[second_r_id]
                    second_r_desc = id_to_desc[second_r_id]

                    val_actual_ids.append(r_id)
                    sentence = " ".join(sent["tokens"])
                    head_entity_indices = sent["h"][2][0]
                    tail_entity_indices = sent["t"][2][0]
                    head_entity = " ".join([sent["tokens"][i] for i in head_entity_indices])
                    tail_entity = " ".join([sent["tokens"][i] for i in tail_entity_indices])
                    gold_answers = tail_entity
                    val_passages.append(sentence)
                    val_entity_relations.append(
                        white_space_fix(head_entity + " <SEP> " + second_r_name)
                    )
                    val_entities.append(white_space_fix(head_entity))
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
                    val_answers.append(
                        white_space_fix(gold_answers) + " </s>"
                    )

        for r_id in test_r_ids:
            r_name = id_to_label[r_id]
            r_desc = id_to_desc[r_id]
            for sent in data[r_id]:
                for second_r_id in test_r_ids:
                    second_r_name = id_to_label[second_r_id]
                    second_r_desc = id_to_desc[second_r_id]

                    test_actual_ids.append(r_id)
                    sentence = " ".join(sent["tokens"])
                    head_entity_indices = sent["h"][2][0]
                    tail_entity_indices = sent["t"][2][0]
                    head_entity = " ".join([sent["tokens"][i] for i in head_entity_indices])
                    tail_entity = " ".join([sent["tokens"][i] for i in tail_entity_indices])
                    gold_answers = tail_entity
                    test_passages.append(sentence)
                    test_entity_relations.append(
                        white_space_fix(head_entity + " <SEP> " + second_r_name)
                    )
                    test_entities.append(white_space_fix(head_entity))
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
                        + white_space_fix(" and ".join(gold_answers))
                        + " context: "
                        + white_space_fix(sentence)
                        + " </s>"
                    )
                    test_answers.append(
                        white_space_fix(gold_answers) + " </s>"
                    )

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
                train_entities.append(white_space_fix(head_entity))
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
                train_answers.append(
                    white_space_fix(gold_answers) + " </s>"
                )
            
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
                train_entities.append(white_space_fix(head_entity))
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
                train_answers.append(
                    white_space_fix("no_answer") + " </s>"
                )

            '''
            if augment_with_unks:
                # add negative examples for the train r_id if exists in the RE-QA dataset.
                if r_id in unk_relation_ids:
                    num_samples = min([len(data[r_id]), len(unk_relation_ids[r_id])])
                    for line_arr in random.sample(unk_relation_ids[r_id], num_samples):
                        sentence = line_arr[3]
                        head_entity = line_arr[2]
                        tail_entity = "no_answer"
                        gold_answers = "no_answer"
                        train_passages.append(sentence)
                        train_entity_relations.append(white_space_fix(head_entity + " <SEP> " + r_name))
                        train_entities.append(white_space_fix(head_entity))
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
                        train_answers.append(
                            white_space_fix(gold_answers) + " </s>"
                        )
            '''

    train_df = pd.DataFrame(
        {
            "passages": train_passages,
            "contexts": train_contexts,
            "answers": train_answers,
            "entity_relations": train_entity_relations,
            "entities": train_entities,
            "posterier_contexts": train_posterier_contexts,
        }
    )

    val_df = pd.DataFrame(
        {
            "passages": val_passages,
            "contexts": val_contexts,
            "answers": val_answers,
            "entity_relations": val_entity_relations,
            "entities": val_entities,
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
            "entities": test_entities,
            "posterier_contexts": test_posterier_contexts,
            "actual_ids": test_actual_ids,
        }
    )

    train_df.to_csv(
        "./train_data_" + str(seed) + ".csv", sep=",", header=True, index=False
    )
    val_df.to_csv("./val_data_" + str(seed) + ".csv", sep=",", header=True, index=False)
    test_df.to_csv(
        "./test_data_" + str(seed) + ".csv", sep=",", header=True, index=False
    )

    return


def create_fewrl_dataset(
    question_tokenizer,
    answer_tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    train_fewrel_path=None,
    dev_fewrel_path=None,
    test_fewrel_path=None,
    concat=False,
):
    """Function to create the fewrl dataset."""
    train_df = pd.read_csv(train_fewrel_path, sep=",")
    dev_df = pd.read_csv(dev_fewrel_path, sep=",")
    test_df = pd.read_csv(test_fewrel_path, sep=",")

    train_passages = train_df["passages"].tolist()
    train_contexts = train_df["contexts"].tolist()
    train_answers = train_df["answers"].tolist()
    train_entity_relations = train_df["entity_relations"].tolist()
    train_entities = [str(row) for row in train_df["entities"].tolist()]
    train_posterier_contexts = train_df["posterier_contexts"].tolist()

    val_passages = dev_df["passages"].tolist()
    val_contexts = dev_df["contexts"].tolist()
    val_answers = dev_df["answers"].tolist()
    val_entity_relations = dev_df["entity_relations"].tolist()
    val_entities = [str(row) for row in dev_df["entities"].tolist()]
    val_posterier_contexts = dev_df["posterier_contexts"].tolist()

    test_passages = test_df["passages"].tolist()
    test_contexts = test_df["contexts"].tolist()
    test_answers = test_df["answers"].tolist()
    test_entity_relations = test_df["entity_relations"].tolist()
    test_entities = [str(row) for row in test_df["entities"].tolist()]
    test_posterier_contexts = test_df["posterier_contexts"].tolist()

    if concat:
        for i in range(len(train_contexts)):
            ctx = train_contexts[i]
            ctx_str = ctx.split("context: ")[1]
            ent_rel_str = train_entity_relations[i]
            new_train_context = white_space_fix(
                "question: " + ent_rel_str + " context: " + ctx_str
            )
            train_contexts[i] = new_train_context

        for i in range(len(val_contexts)):
            ctx = val_contexts[i]
            ctx_str = ctx.split("context: ")[1]
            ent_rel_str = val_entity_relations[i]
            new_val_context = white_space_fix(
                "question: " + ent_rel_str + " context: " + ctx_str
            )
            val_contexts[i] = new_val_context


        for i in range(len(test_contexts)):
            ctx = test_contexts[i]
            ctx_str = ctx.split("context: ")[1]
            ent_rel_str = test_entity_relations[i]
            new_test_context = white_space_fix(
                "question: " + ent_rel_str + " context: " + ctx_str
            )
            test_contexts[i] = new_test_context
            print(test_contexts[i])

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
    train_entity_encodings = question_tokenizer(
        train_entities,
        truncation=True,
        padding="max_length",
        max_length=decoder_max_length,
        add_special_tokens=False,
    )

    train_posterier_encodings = question_tokenizer(
        train_posterier_contexts,
        truncation=True,
        padding="max_length",
        max_length=source_max_length,
        add_special_tokens=False,
    )
    train_encodings["passages"] = train_passages
    train_encodings["entity_relations"] = train_entity_relations
    train_encodings["posterier_input_ids"] = train_posterier_encodings.pop("input_ids")
    train_encodings["posterier_attention_mask"] = train_posterier_encodings.pop(
        "attention_mask"
    )
    train_encodings["entity_input_ids"] = train_entity_encodings.pop("input_ids")
    train_encodings["entity_attention_mask"] = train_entity_encodings.pop(
        "attention_mask"
    )

    train_encodings["entity_relation_passage_input_ids"] = train_encodings["input_ids"]
    train_encodings["entity_relation_passage_attention_mask"] = train_encodings[
        "attention_mask"
    ]

    train_encodings["target_attention_mask"] = train_answer_encodings["attention_mask"]
    train_encodings["second_entity_labels"] = train_answer_encodings["input_ids"]
    train_encodings["second_entity_attention_mask"] = train_answer_encodings[
        "attention_mask"
    ]

    # because HuggingFace automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored

    train_labels = [
        [-100 if token == answer_tokenizer.pad_token_id else token for token in labels]
        for labels in train_encodings["second_entity_labels"]
    ]
    train_encodings["second_entity_labels"] = train_labels
    train_encodings["labels"] = train_labels

    val_encodings["passages"] = val_passages
    val_encodings["entity_relations"] = val_entity_relations

    val_encodings["entity_relation_passage_input_ids"] = val_encodings["input_ids"]
    val_encodings["entity_relation_passage_attention_mask"] = val_encodings[
        "attention_mask"
    ]
    val_encodings["target_attention_mask"] = val_answer_encodings["attention_mask"]

    val_encodings["second_entity_labels"] = val_answer_encodings["input_ids"]
    val_encodings["second_entity_attention_mask"] = val_answer_encodings[
        "attention_mask"
    ]

    # because Huggingface automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored.

    val_labels = [
        [-100 if token == answer_tokenizer.pad_token_id else token for token in labels]
        for labels in val_encodings["second_entity_labels"]
    ]
    val_encodings["second_entity_labels"] = val_labels
    val_encodings["labels"] = val_labels

    test_encodings["passages"] = test_passages
    test_encodings["entity_relations"] = test_entity_relations
    test_encodings["entity_relation_passage_input_ids"] = test_encodings["input_ids"]

    test_encodings["entity_relation_passage_attention_mask"] = test_encodings[
        "attention_mask"
    ]
    test_encodings["target_attention_mask"] = test_answer_encodings["attention_mask"]
    test_encodings["second_entity_labels"] = test_answer_encodings["input_ids"]
    test_encodings["second_entity_attention_mask"] = test_answer_encodings[
        "attention_mask"
    ]

    # because Huggingface automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored.

    test_labels = [
        [-100 if token == answer_tokenizer.pad_token_id else token for token in labels]
        for labels in test_encodings["second_entity_labels"]
    ]
    test_encodings["second_entity_labels"] = test_labels
    test_encodings["labels"] = test_labels

    class HelperDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            row = {}
            for key, val in self.encodings.items():
                if key in ["passages", "entity_relations"]:
                    row[key] = val[idx]
                else:
                    row[key] = torch.tensor(val[idx])
            return row

        def __len__(self):
            if "entity_relation_passage_input_ids" in self.encodings:
                return len(self.encodings.entity_relation_passage_input_ids)
            if "input_ids" in self.encodings:
                return len(self.encodings.input_ids)

    train_dataset = HelperDataset(train_encodings)
    val_dataset = HelperDataset(val_encodings)
    test_dataset = HelperDataset(test_encodings)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )


def create_relation_qq_dataset(
    question_tokenizer,
    answer_tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    train_fewrel_path=None,
    concat=False,
    shuffle=False,
    for_fewrel_dataset=False,
):
    """Function to create the fewrl dataset for training with negative
    samples."""

    if for_fewrel_dataset:
        train_df = pd.read_csv(train_fewrel_path, sep=",")

        train_passages = train_df["passages"].tolist()
        train_contexts = train_df["contexts"].tolist()
        train_answers = train_df["answers"].tolist()
        train_entity_relations = train_df["entity_relations"].tolist()
        train_entities = [str(row) for row in train_df["entities"].tolist()]
        train_posterier_contexts = train_df["posterier_contexts"].tolist()
    else:
        (
        train_passages,
        train_contexts,
        train_posterier_contexts,
        train_answers,
        train_entity_relations,
        train_entities,
        ) = read_gold_re_qa_relation_data(
            train_fewrel_path, concat=False, for_question_generation=True
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
    train_entity_encodings = question_tokenizer(
        train_entities,
        truncation=True,
        padding="max_length",
        max_length=decoder_max_length,
        add_special_tokens=False,
    )

    train_posterier_encodings = question_tokenizer(
        train_posterier_contexts,
        truncation=True,
        padding="max_length",
        max_length=source_max_length,
        add_special_tokens=False,
    )
    train_encodings["passages"] = train_passages
    train_encodings["entity_relations"] = train_entity_relations
    train_encodings["posterier_input_ids"] = train_posterier_encodings.pop("input_ids")
    train_encodings["posterier_attention_mask"] = train_posterier_encodings.pop(
        "attention_mask"
    )
    train_encodings["entity_input_ids"] = train_entity_encodings.pop("input_ids")
    train_encodings["entity_attention_mask"] = train_entity_encodings.pop(
        "attention_mask"
    )

    train_encodings["entity_relation_passage_input_ids"] = train_encodings["input_ids"]
    train_encodings["entity_relation_passage_attention_mask"] = train_encodings[
        "attention_mask"
    ]

    train_encodings["target_attention_mask"] = train_answer_encodings["attention_mask"]
    train_encodings["second_entity_labels"] = train_answer_encodings["input_ids"]
    train_encodings["second_entity_attention_mask"] = train_answer_encodings[
        "attention_mask"
    ]

    # because HuggingFace automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # We have to make sure that the PAD token is ignored

    train_labels = [
        [-100 if token == answer_tokenizer.pad_token_id else token for token in labels]
        for labels in train_encodings["second_entity_labels"]
    ]
    train_encodings["second_entity_labels"] = train_labels
    train_encodings["labels"] = train_labels

    class HelperDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            row = {}
            for key, val in self.encodings.items():
                if key in ["passages", "entity_relations"]:
                    row[key] = val[idx]
                else:
                    row[key] = torch.tensor(val[idx])
            return row

        def __len__(self):
            if "entity_relation_passage_input_ids" in self.encodings:
                return len(self.encodings.entity_relation_passage_input_ids)
            if "input_ids" in self.encodings:
                return len(self.encodings.input_ids)

    train_dataset = HelperDataset(train_encodings)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return (
        train_loader,
        train_dataset,
    )
