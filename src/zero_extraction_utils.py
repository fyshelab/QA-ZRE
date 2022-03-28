import json
import random
from pathlib import Path

import pandas as pd
import torch
# from datasets import load_dataset
from torch.utils.data import DataLoader

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


def convert_reqa_to_fewrel_format(path, output_path, train=False):
    path = Path(path)

    label_to_id = {}
    with open("./props.json", "r") as fd:
        re_desc_data = json.load(fd)
        for row in re_desc_data:
            re_label = row["label"]
            re_id = row["id"]
            label_to_id[re_label] = re_id

    output_json = {}
    with open(output_path, "w") as out_fd:
        with open(path, "r") as fd:
            for line in fd:
                line = line.strip()
                line_arr = line.split("\t")
                passage = line_arr[3]
                h = line_arr[2]
                if line_arr[4:]:
                    t = line_arr[4:][0]
                else:
                    continue
                rel_desc = line_arr[0]
                rel_id = label_to_id.get(rel_desc.lower(), rel_desc.lower())
                list_update = output_json.get(rel_id, [])
                tokens = white_space_fix(passage).split(" ")
                h_tokens = white_space_fix(h).split(" ")
                t_tokens = white_space_fix(t).split(" ")
                h_indices = find_sub_list(h_tokens, tokens)
                t_indices = find_sub_list(t_tokens, tokens)
                # if train and len(tokens) > 160:
                #    print("skipped an example in train")
                #    continue
                if len(tokens) > 160:
                    continue
                    print("skipped an example in test")
                ret_row = {
                    "tokens": tokens,
                    "h": [h, "dummy_id", [h_indices]],
                    "t": [t, "dummy_id", [t_indices]],
                }
                list_update.append(ret_row)
                output_json[rel_id] = list_update
        json.dump(output_json, out_fd)


def read_gold_re_qa_relation_data(path, concat=False, for_question_generation=False):
    """Create val data for relation classification considering all the data and
    gold_templates."""
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
                    if white_space_fix(rel_type).lower() in rel_dict:
                        contexts.append(
                            "answer: "
                            + white_space_fix(line_arr[2])
                            + " <SEP> "
                            + white_space_fix(rel_type)
                            + " ; "
                            + rel_dict[white_space_fix(rel_type).lower()]
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
                            + rel_dict[white_space_fix(rel_type).lower()]
                            + " "
                            + white_space_fix(" and ".join(gold_answers))
                            + " context: "
                            + white_space_fix(passage)
                            + " </s>"
                        )
                    else:
                        contexts.append(
                            "answer: "
                            + white_space_fix(line_arr[2])
                            + " <SEP> "
                            + white_space_fix(rel_type)
                            + " context: "
                            + white_space_fix(passage)
                            + " </s>"
                        )
                        posterier_contexts.append(
                            "answer: "
                            + white_space_fix(line_arr[2])
                            + " <SEP> "
                            + white_space_fix(rel_type)
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


def read_zero_re_qa(path, ignore_unknowns=True, gold_question=False, concat=False):
    """Main function to read the zero re qa dataset."""
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

    with open(path, "r") as fd:
        contexts = []
        posterier_contexts = []
        answers = []
        passages = []
        entities = []
        entity_relations = []
        for line in fd:
            line = line.strip()
            line_arr = line.split("\t")
            passage = line_arr[3]
            if concat:
                gold_question = line_arr[2] + " <SEP> " + line_arr[0]
            elif gold_question:
                gold_question = line_arr[1].replace("XXX", " " + line_arr[2] + " ")
            if len(line_arr) > 4:
                gold_answers = line_arr[4:]
            elif ignore_unknowns:
                continue
            else:
                gold_answers = ["no_answer"]
            passages.append(passage)
            entity_relations.append(white_space_fix(line_arr[2] + " " + line_arr[0]))
            entities.append(white_space_fix(line_arr[2]))
            if concat or gold_question:
                contexts.append(
                    "question: "
                    + white_space_fix(gold_question)
                    + " context: "
                    + white_space_fix(passage)
                    + " </s>"
                )
            else:
                if white_space_fix(line_arr[0]).lower() in rel_dict:
                    contexts.append(
                        "answer: "
                        + white_space_fix(line_arr[2])
                        + " <SEP> "
                        + white_space_fix(line_arr[0])
                        + " ; "
                        + rel_dict[white_space_fix(line_arr[0]).lower()]
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
                        + rel_dict[white_space_fix(line_arr[0]).lower()]
                        + " "
                        + white_space_fix(" and ".join(gold_answers))
                        + " context: "
                        + white_space_fix(passage)
                        + " </s>"
                    )
                else:
                    contexts.append(
                        "answer: "
                        + white_space_fix(line_arr[2])
                        + " <SEP> "
                        + white_space_fix(line_arr[0])
                        + " context: "
                        + white_space_fix(passage)
                        + " </s>"
                    )
                    posterier_contexts.append(
                        "answer: "
                        + white_space_fix(line_arr[2])
                        + " <SEP> "
                        + white_space_fix(line_arr[0])
                        + " "
                        + white_space_fix(" and ".join(gold_answers))
                        + " context: "
                        + white_space_fix(passage)
                        + " </s>"
                    )

            answers.append(white_space_fix(" and ".join(gold_answers)) + " </s>")
    return passages, contexts, answers, entity_relations, entities, posterier_contexts


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


def read_fewrl_names(split):
    """Read the few rel dataset."""

    def process_few_rel_row(row):
        """Helper functions for fewrel Dataset."""
        return {"r_id": row["relation"], "r_name": row["names"][0]}

    few_rel = load_dataset("few_rel", "default")[split]

    rel_names = few_rel.map(
        process_few_rel_row,
        remove_columns=["relation", "tokens", "head", "tail", "names"],
    )
    return rel_names


rel_dict = {
    "P931": "place served by transport hub",
    "P4552": "mountain range",
    "P140": "religion",
    "P1923": "participating team",
    "P150": "contains administrative territorial entity",
    "P6": "head of government",
    "P27": "country of citizenship",
    "P449": "original network",
    "P1435": "heritage designation",
    "P175": "performer",
    "P1344": "participant of",
    "P39": "position held",
    "P527": "has part",
    "P740": "location of formation",
    "P706": "located on terrain feature",
    "P84": "architect",
    "P495": "country of origin",
    "P123": "publisher",
    "P57": "director",
    "P22": "father",
    "P178": "developer",
    "P241": "military branch",
    "P403": "mouth of the watercourse",
    "P1411": "nominated for",
    "P135": "movement",
    "P991": "successful candidate",
    "P156": "followed by",
    "P176": "manufacturer",
    "P31": "instance of",
    "P1877": "after a work by",
    "P102": "member of political party",
    "P1408": "licensed to broadcast to",
    "P159": "headquarters location",
    "P3373": "sibling",
    "P1303": "instrument",
    "P17": "country",
    "P106": "occupation",
    "P551": "residence",
    "P937": "work location",
    "P355": "subsidiary",
    "P710": "participant",
    "P137": "operator",
    "P674": "characters",
    "P466": "occupant",
    "P136": "genre",
    "P306": "operating system",
    "P127": "owned by",
    "P400": "platform",
    "P974": "tributary",
    "P1346": "winner",
    "P460": "said to be the same as",
    "P86": "composer",
    "P118": "league",
    "P264": "record label",
    "P750": "distributor",
    "P58": "screenwriter",
    "P3450": "sports season of league or competition",
    "P105": "taxon rank",
    "P276": "location",
    "P101": "field of work",
    "P407": "language of work or name",
    "P1001": "applies to jurisdiction",
    "P800": "notable work",
    "P131": "located in the administrative territorial entity",
    "P177": "crosses",
    "P364": "original language of film or TV show",
    "P2094": "competition class",
    "P361": "part of",
    "P641": "sport",
    "P59": "constellation",
    "P413": "position played on team / speciality",
    "P206": "located in or next to body of water",
    "P412": "voice type",
    "P155": "follows",
    "P26": "spouse",
    "P410": "military rank",
    "P25": "mother",
    "P463": "member of",
    "P40": "child",
    "P921": "main subject",
    "P1056": "product or material produced",
}

rel_desc = {
    "P931": "territorial entity or entities served by this transport hub (airport, train station, etc.)",
    "P4552": "range or subrange to which the geographical item belongs",
    "P140": "religion of a person, organization or religious building, or associated with this subject",
    "P1923": "Like 'Participant' (P710) but for teams. For an event like a cycle race or a football match you can use this property to list the teams and P710 to list the individuals (with 'member of sports team' (P54) as a qualifier for the individuals)",
    "P150": "(list of) direct subdivisions of an administrative territorial entity",
    "P6": "head of the executive power of this town, city, municipality, state, country, or other governmental body",
    "P27": "the object is a country that recognizes the subject as its citizen",
    "P449": "network(s) the radio or television show was originally aired on, not including later re-runs or additional syndication",
    "P1435": "heritage designation of a cultural or natural site",
    "P175": "actor, musician, band or other performer associated with this role or musical work",
    "P1344": "event a person or an organization was/is a participant in, inverse of P710 or P1923",
    "P39": "subject currently or formerly holds the object position or public office",
    "P527": 'part of this subject; inverse property of "part of" (P361). See also "has parts of the class" (P2670).',
    "P740": "location where a group or organization was formed",
    "P706": "located on the specified landform. Should not be used when the value is only political/administrative (P131) or a mountain range (P4552).",
    "P84": "person or architectural firm that designed this building",
    "P495": "country of origin of this item (creative work, food, phrase, product, etc.)",
    "P123": "organization or person responsible for publishing books, periodicals, games or software",
    "P57": "director(s) of film, TV-series, stageplay, video game or similar",
    "P22": 'male parent of the subject. For stepfather, use "stepparent" (P3448)',
    "P178": "organisation or person that developed the item",
    "P241": "branch to which this military unit, award, office, or person belongs, e.g. Royal Navy",
    "P403": "the body of water to which the watercourse drains",
    "P1411": 'award nomination received by a person, organisation or creative work (inspired from "award received" (Property:P166))',
    "P135": "literary, artistic, scientific or philosophical movement associated with this person or work",
    "P991": "person(s) elected after the election",
    "P156": 'immediately following item in a series of which the subject is a part [if the subject has been replaced, e.g. political offices, use "replaced by" (P1366)]',
    "P176": "manufacturer or producer of this product",
    "P31": "that class of which this subject is a particular example and member (subject typically an individual member with a proper name label); different from P279; using this property as a qualifier is deprecated—use P2868 or P3831 instead",
    "P1877": "artist whose work strongly inspired/ was copied in this item",
    "P102": "the political party of which this politician is or has been a member",
    "P1408": "place that a radio/TV station is licensed/required to broadcast to",
    "P159": 'specific location where an organization\'s headquarters is or has been situated. Inverse property of "occupant" (P466).',
    "P3373": 'the subject has the object as their sibling (brother, sister, etc.). Use "relative" (P1038) for siblings-in-law (brother-in-law, sister-in-law, etc.) and step-siblings (step-brothers, step-sisters, etc.)',
    "P1303": "musical instrument that a person plays",
    "P17": "sovereign state of this item; don't use on humans",
    "P106": 'occupation of a person; see also "field of work" (Property:P101), "position held" (Property:P39)',
    "P551": "the place where the person is or has been, resident",
    "P937": "location where persons were active",
    "P355": "subsidiary of a company or organization, opposite of parent organization (P749)",
    "P710": 'person, group of people or organization (object) that actively takes/took part in an event or process (subject).  Preferably qualify with "object has role" (P3831). Use P1923 for participants that are teams.',
    "P137": "person, profession, or organization that operates the equipment, facility, or service; use country for diplomatic missions",
    "P674": "characters which appear in this item (like plays, operas, operettas, books, comics, films, TV series, video games)",
    "P466": "a person or organization occupying property",
    "P136": "creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic",
    "P306": "operating system (OS) on which a software works or the OS installed on hardware",
    "P127": "owner of the subject",
    "P400": "platform for which a work was developed or released, or the specific platform version of a software product",
    "P974": "stream or river that flows into this main stem (or parent) river",
    "P1346": "winner of an event - do not use for awards (use P166 instead), nor for wars or battles",
    "P460": "this item is said to be the same as that item, but the statement is disputed",
    "P86": 'person(s) who wrote the music [for lyricist, use "lyrics by" (P676)]',
    "P118": "league in which team or player plays or has played in",
    "P264": "brand and trademark associated with the marketing of subject music recordings and music videos",
    "P750": "distributor of a creative work; distributor for a record label",
    "P58": "person(s) who wrote the script for subject item",
    "P3450": 'property that shows the competition of which the item is a season. Use P5138 for "season of club or team".',
    "P105": "level in a taxonomic hierarchy",
    "P276": "location of the item, physical object or event is within. In case of an administrative entity use P131. In case of a distinct terrain feature use P706.",
    "P101": "specialization of a person or organization; see P106 for the occupation",
    "P407": "language associated with this creative work (such as books, shows, songs, or websites) or a name (for persons use P103 and P1412)",
    "P1001": "the item (an institution, law, public office ...) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, ...)",
    "P800": "notable scientific, artistic or literary work, or other work of significance among subject's works",
    "P131": "the item is located on the territory of the following administrative entity. Use P276 (location) for specifying the location of non-administrative places and for items about events",
    "P177": "obstacle (body of water, road, ...) which this bridge crosses over or this tunnel goes under",
    "P364": 'language in which a film or a performance work was originally created. Deprecated for written works; use P407 ("language of work or name") instead.',
    "P2094": "official classification by a regulating body under which the subject (events, teams, participants, or equipment) qualifies for inclusion",
    "P361": 'object of which the subject is a part (it\'s not useful to link objects which are themselves parts of other objects already listed as parts of the subject). Inverse property of "has part" (P527, see also "has parts of the class" (P2670)).',
    "P641": "sport in which the subject participates or belongs to",
    "P59": "the area of the celestial sphere of which the subject is a part (from a scientific standpoint, not an astrological one)",
    "P413": "position or specialism of a player on a team, e.g. Small Forward",
    "P206": "sea, lake or river",
    "P412": "person's voice type. expected values: soprano, mezzo-soprano, contralto, countertenor, tenor, baritone, bass (and derivatives)",
    "P155": 'immediately prior item in a series of which the subject is a part [if the subject has replaced the preceding item, e.g. political offices, use "replaces" (P1365)]',
    "P26": 'the subject has the object as their spouse (husband, wife, partner, etc.). Use "partner" (P451) for non-married companions',
    "P410": 'military rank achieved by a person (should usually have a "start time" qualifier), or military rank associated with a position',
    "P25": 'female parent of the subject. For stepmother, use "stepparent" (P3448)',
    "P463": "organization or club to which the subject belongs. Do not use for membership in ethnic or social groups, nor for holding a position such as a member of parliament (use P39 for that).",
    "P40": "subject has object as biological, foster, and/or adoptive child",
    "P921": "primary topic of a work (see also P180: depicts)",
    "P1056": "material or product produced by a government agency, business, industry, facility, or process",
}


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

    test_contexts = []
    test_posterier_contexts = []
    test_answers = []
    test_passages = []
    test_entities = []
    test_entity_relations = []

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
                val_passages.append(sentence)
                val_entity_relations.append(white_space_fix(head_entity + " " + r_name))
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
                test_passages.append(sentence)
                test_entity_relations.append(
                    white_space_fix(head_entity + " " + r_name)
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
                    white_space_fix(head_entity + " " + r_name)
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


def read_fewrl_dataset(fewrel_path, seed=10, m=5):
    sentence_delimiters = [". ", ".\n", "? ", "?\n", "! ", "!\n"]

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

    with open(fewrel_path, "r") as json_file:
        data = json.load(json_file)
        r_ids = list(data.keys())
        random.shuffle(r_ids)
        val_r_ids = r_ids[:m]
        test_r_ids = r_ids[m : 4 * m]
        train_r_ids = r_ids[4 * m :]

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
            r_name = rel_dict[r_id]
            r_desc = rel_desc[r_id]
            desc = r_desc.strip(".") + ". "
            pos = [desc.find(delimiter) for delimiter in sentence_delimiters]
            pos = min([p for p in pos if p >= 0])
            re_desc = desc[:pos]

            for sent in data[r_id]:
                for second_r_id in val_r_ids:
                    second_r_name = rel_dict[second_r_id]
                    second_r_desc = rel_desc[second_r_id]
                    second_desc = second_r_desc.strip(".") + ". "
                    second_pos = [
                        second_desc.find(delimiter) for delimiter in sentence_delimiters
                    ]
                    second_pos = min([p for p in second_pos if p >= 0])
                    second_re_desc = second_desc[:second_pos]

                    val_actual_ids.append(r_id)
                    sentence = " ".join(sent["tokens"])
                    head_entity = sent["h"][0]
                    tail_entity = sent["t"][0]
                    gold_answers = [tail_entity]
                    val_passages.append(sentence)
                    val_entity_relations.append(
                        white_space_fix(head_entity + " " + second_r_name)
                    )
                    val_entities.append(white_space_fix(head_entity))
                    val_contexts.append(
                        "answer: "
                        + white_space_fix(head_entity)
                        + " <SEP> "
                        + white_space_fix(second_r_name)
                        + " ; "
                        + white_space_fix(second_re_desc)
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
                        + white_space_fix(second_re_desc)
                        + " "
                        + white_space_fix(" and ".join(gold_answers))
                        + " context: "
                        + white_space_fix(sentence)
                        + " </s>"
                    )
                    val_answers.append(
                        white_space_fix(" and ".join(gold_answers)) + " </s>"
                    )

        for r_id in test_r_ids:
            r_name = rel_dict[r_id]
            r_desc = rel_desc[r_id]
            desc = r_desc.strip(".") + ". "
            pos = [desc.find(delimiter) for delimiter in sentence_delimiters]
            pos = min([p for p in pos if p >= 0])
            re_desc = desc[:pos]

            for sent in data[r_id]:
                for second_r_id in test_r_ids:
                    second_r_name = rel_dict[second_r_id]
                    second_r_desc = rel_desc[second_r_id]
                    second_desc = second_r_desc.strip(".") + ". "
                    second_pos = [
                        second_desc.find(delimiter) for delimiter in sentence_delimiters
                    ]
                    second_pos = min([p for p in second_pos if p >= 0])
                    second_re_desc = second_desc[:second_pos]

                    test_actual_ids.append(r_id)
                    sentence = " ".join(sent["tokens"])
                    head_entity = sent["h"][0]
                    tail_entity = sent["t"][0]
                    gold_answers = [tail_entity]
                    test_passages.append(sentence)
                    test_entity_relations.append(
                        white_space_fix(head_entity + " " + second_r_name)
                    )
                    test_entities.append(white_space_fix(head_entity))
                    test_contexts.append(
                        "answer: "
                        + white_space_fix(head_entity)
                        + " <SEP> "
                        + white_space_fix(second_r_name)
                        + " ; "
                        + white_space_fix(second_re_desc)
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
                        + white_space_fix(second_re_desc)
                        + " "
                        + white_space_fix(" and ".join(gold_answers))
                        + " context: "
                        + white_space_fix(sentence)
                        + " </s>"
                    )
                    test_answers.append(
                        white_space_fix(" and ".join(gold_answers)) + " </s>"
                    )

        for r_id in train_r_ids:
            r_name = rel_dict[r_id]
            r_desc = rel_desc[r_id]
            desc = r_desc.strip(".") + ". "
            pos = [desc.find(delimiter) for delimiter in sentence_delimiters]
            pos = min([p for p in pos if p >= 0])
            re_desc = desc[:pos]

            for sent in data[r_id]:
                sentence = " ".join(sent["tokens"])
                head_entity = sent["h"][0]
                tail_entity = sent["t"][0]
                gold_answers = [tail_entity]
                train_passages.append(sentence)
                train_entity_relations.append(
                    white_space_fix(head_entity + " " + r_name)
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
    val_entities = dev_df["entities"].tolist()
    val_posterier_contexts = dev_df["posterier_contexts"].tolist()

    test_passages = test_df["passages"].tolist()
    test_contexts = test_df["contexts"].tolist()
    test_answers = test_df["answers"].tolist()
    test_entity_relations = test_df["entity_relations"].tolist()
    test_entities = test_df["entities"].tolist()
    test_posterier_contexts = test_df["posterier_contexts"].tolist()

    if concat:
        for i in range(len(train_contexts)):
            ctx = train_contexts[i]
            ctx_str = ctx.split("context: ")[1]
            ent_rel_str = train_entity_relations[i]
            new_train_context = white_space_fix(
                "answer: " + ent_rel_str + " context: " + ctx_str
            )
            train_contexts[i] = new_train_context

        for i in range(len(val_contexts)):
            ctx = val_contexts[i]
            ctx_str = ctx.split("context: ")[1]
            ent_rel_str = val_entity_relations[i]
            new_val_context = white_space_fix(
                "answer: " + ent_rel_str + " context: " + ctx_str
            )
            val_contexts[i] = new_val_context

        for i in range(len(test_contexts)):
            ctx = test_contexts[i]
            ctx_str = ctx.split("context: ")[1]
            ent_rel_str = test_entity_relations[i]
            new_test_context = white_space_fix(
                "answer: " + ent_rel_str + " context: " + ctx_str
            )
            test_contexts[i] = new_test_context

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
        test_loader,
    )


def create_relation_train_dataset_fewrl(fewrel_path, seed=10, m=5, neg_samples=1):
    sentence_delimiters = [". ", ".\n", "? ", "?\n", "! ", "!\n"]

    set_random_seed(seed)

    train_contexts = []
    train_posterier_contexts = []
    train_answers = []
    train_passages = []
    train_entities = []
    train_entity_relations = []

    with open(fewrel_path, "r") as json_file:
        data = json.load(json_file)
        r_ids = list(data.keys())
        random.shuffle(r_ids)
        train_r_ids = r_ids[4 * m :]

        train_id_df = pd.DataFrame(train_r_ids, columns=["relation_ids"])
        train_id_df.to_csv(
            "./train_ids_" + str(seed) + ".csv", sep=",", header=True, index=False
        )

        for r_id in train_r_ids:
            r_name = rel_dict[r_id]
            r_desc = rel_desc[r_id]
            desc = r_desc.strip(".") + ". "
            pos = [desc.find(delimiter) for delimiter in sentence_delimiters]
            pos = min([p for p in pos if p >= 0])
            re_desc = desc[:pos]

            # Create a randomly shuffled dataset.
            sentences = data[r_id]
            random.shuffle(sentences)
            for sent in sentences:
                # positive sample
                sentence = " ".join(sent["tokens"])
                head_entity = sent["h"][0]
                tail_entity = sent["t"][0]
                gold_answers = [tail_entity]
                train_passages.append(sentence)
                train_entity_relations.append(
                    white_space_fix(head_entity + " " + r_name)
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

                # add the rest with neg_samples
                for i in range(neg_samples):
                    neg_sent = random.choice(sentences)
                    neg_sentence = " ".join(neg_sent["tokens"])
                    neg_head_entity = neg_sent["h"][0]
                    neg_tail_entity = neg_sent["t"][0]
                    random_r_id = random.choice(list(set(train_r_ids) - {r_id}))
                    r_r_name = rel_dict[random_r_id]
                    r_r_desc = rel_desc[random_r_id]
                    r_desc = r_r_desc.strip(".") + ". "
                    r_pos = [
                        r_desc.find(delimiter) for delimiter in sentence_delimiters
                    ]
                    r_pos = min([p for p in r_pos if p >= 0])
                    r_re_desc = r_desc[:r_pos]

                    train_passages.append(neg_sentence)
                    train_entity_relations.append(
                        white_space_fix(neg_head_entity + " " + r_r_name)
                    )

                    train_entities.append(white_space_fix(neg_head_entity))
                    train_contexts.append(
                        "answer: "
                        + white_space_fix(neg_head_entity)
                        + " <SEP> "
                        + white_space_fix(r_r_name)
                        + " ; "
                        + white_space_fix(r_re_desc)
                        + " context: "
                        + white_space_fix(neg_sentence)
                        + " </s>"
                    )
                    train_posterier_contexts.append(
                        "answer: "
                        + white_space_fix(neg_head_entity)
                        + " <SEP> "
                        + white_space_fix(r_r_name)
                        + " ; "
                        + white_space_fix(r_re_desc)
                        + " "
                        + white_space_fix(neg_tail_entity)
                        + " context: "
                        + white_space_fix(neg_sentence)
                        + " </s>"
                    )
                    train_answers.append(white_space_fix("no_answer") + " </s>")

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

    train_df.to_csv(
        "./unk_train_data_" + str(seed) + ".csv", sep=",", header=True, index=False
    )

    return (
        train_passages,
        train_contexts,
        train_answers,
        train_entity_relations,
        train_entities,
        train_posterier_contexts,
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
):
    """Function to create the fewrl dataset for training with negative
    samples."""

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

    """
    train_df = pd.read_csv(train_fewrel_path, sep=",")

    train_passages = train_df["passages"].tolist()
    train_contexts = train_df["contexts"].tolist()
    train_answers = train_df["answers"].tolist()
    train_entity_relations = train_df["entity_relations"].tolist()
    train_entities = [str(row) for row in train_df["entities"].tolist()]
    train_posterier_contexts = train_df["posterier_contexts"].tolist()

    if concat:
        for i in range(len(train_contexts)):
            ctx = train_contexts[i]
            ctx_str = ctx.split("context: ")[1]
            ent_rel_str = train_entity_relations[i]
            new_train_context = white_space_fix(
                "answer: " + ent_rel_str + " context: " + ctx_str
            )
            train_contexts[i] = new_train_context
    """
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


def read_relation_extraction_lm_reqa_data(path):
    """Main function to create the relation extraction LM data from the RE-QA
    dataset."""
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

    with open(path, "r") as fd:
        contexts = []
        answers = []
        for line in fd:
            line = line.strip()
            line_arr = line.split("\t")
            passage = line_arr[3]
            h_entity = line_arr[2]
            output_relation_lm = line_arr[0]
            if len(line_arr) > 4:
                tail_entity = line_arr[4:]
            else:
                tail_entity = ["no_answer"]
            contexts.append(
                "head: "
                + white_space_fix(h_entity)
                + " tail: "
                + white_space_fix(" and ".join(tail_entity))
                + " context: "
                + white_space_fix(passage)
                + " </s>"
            )

            output_lm = output_relation_lm
            if output_relation_lm.lower() in rel_dict:
                output_lm = rel_dict[output_relation_lm.lower()]
            answers.append(white_space_fix(output_lm) + " </s>")

    return contexts, answers


def create_relation_extraction_lm_dataset(
    tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    data_file,
    shuffle=False,
):
    contexts, answers = read_relation_extraction_lm_reqa_data(data_file)
    train_encodings = tokenizer(
        contexts,
        truncation=True,
        padding="max_length",
        max_length=source_max_length,
        add_special_tokens=False,
    )
    train_answer_encodings = tokenizer(
        answers,
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
