import json
import os
import random
from pathlib import Path

import pandas as pd
import torch
# from datasets import load_dataset
from torch.utils.data import DataLoader


def white_space_fix(text):
    return " ".join(text.split())


def read_zero_re_qa(path, ignore_unknowns=True, gold_question=False, concat=False):
    """Main function to read the zero re qa dataset."""
    path = Path(path)
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


def test_read_zero_re_qa():
    """Test code for the re dataset reading file.

    The function tests three modes: gold questions! concat questions!
    question generator data!
    """
    passages, contexts, answers, entity_relations = read_zero_re_qa(
        "./zero-shot-extraction/relation_splits/train.very_small.0",
        ignore_unknowns=True,
        gold_question=True,
        concat=False,
    )
    assert len(contexts) == len(passages) == len(answers) == 8445

    expected_context = "question: Which is the body of water by Świecie ? context: Świecie is located on the west bank of river Vistula at the mouth of river Wda, approximately 40 kilometers north-east of Bydgoszcz, 105 kilometers south of Gdańsk and 190 kilometers south-west of Kaliningrad. </s>"
    assert contexts[100] == expected_context
    assert answers[100] == "Vistula and Wda </s>"
    assert (
        passages[100]
        == "Świecie is located on the west bank of river Vistula at the mouth of river Wda, approximately 40 kilometers north-east of Bydgoszcz, 105 kilometers south of Gdańsk and 190 kilometers south-west of Kaliningrad."
    )

    passages, contexts, answers, entity_relations = read_zero_re_qa(
        "./zero-shot-extraction/relation_splits/train.very_small.0",
        ignore_unknowns=False,
        gold_question=True,
        concat=False,
    )
    assert len(contexts) == len(passages) == len(answers) == 16800
    assert answers[101] == "no_answer </s>"
    assert (
        contexts[101]
        == "question: What olympics was Shakira ? context: Shakira released her first studio albums, Magia and Peligro, in the early 1990s, failing to attain commercial success; however, she rose to prominence in Latin America with her major-label debut, Pies Descalzos (1996), and her fourth album, Dónde Están los Ladrones? (1998). </s>"
    )

    passages, contexts, answers, entity_relations = read_zero_re_qa(
        "./zero-shot-extraction/relation_splits/train.very_small.0",
        ignore_unknowns=True,
        gold_question=False,
        concat=True,
    )
    assert len(contexts) == len(passages) == len(answers) == 8445
    expected_context = "question: Świecie <SEP> located next to body of water context: Świecie is located on the west bank of river Vistula at the mouth of river Wda, approximately 40 kilometers north-east of Bydgoszcz, 105 kilometers south of Gdańsk and 190 kilometers south-west of Kaliningrad. </s>"
    assert contexts[100] == expected_context

    passages, contexts, answers, entity_relations = read_zero_re_qa(
        "./zero-shot-extraction/relation_splits/train.very_small.0",
        ignore_unknowns=True,
        gold_question=False,
        concat=False,
    )
    assert len(contexts) == len(passages) == len(answers) == 8445
    expected_answer = "answer: Świecie <SEP> located next to body of water context: Świecie is located on the west bank of river Vistula at the mouth of river Wda, approximately 40 kilometers north-east of Bydgoszcz, 105 kilometers south of Gdańsk and 190 kilometers south-west of Kaliningrad. </s>"
    assert contexts[100] == expected_answer


def create_zero_re_qa_dataset(
    question_tokenizer,
    answer_tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    train_file=None,
    dev_file=None,
    distributed=True,
    num_workers=1,
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
    train_sampler = None
    train_loader = None
    if not for_evaluation:
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
        if not for_evaluation:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, train_dataset, val_dataset, None


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
}


def read_fewrl_dataset(fewrel_path, m=5):
    # rel_dict = {}
    # rel_names = read_fewrl_names("train_wiki")
    # for row in rel_names:
    #    rel_dict[row["r_id"]] = row["r_name"]

    # rel_names = read_fewrl_names("val_wiki")
    # for row in rel_names:
    #    rel_dict[row["r_id"]] = row["r_name"]

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
    with open(fewrel_path, "r") as json_file:
        data = json.load(json_file)
        r_ids = list(data.keys())
        random.shuffle(r_ids)
        val_r_ids = r_ids[:m]
        test_r_ids = r_ids[m : 3 * m]
        train_r_ids = r_ids[3 * m :]

        for r_id in val_r_ids:
            r_name = rel_dict[r_id]
            for sent in data[r_id]:
                sentence = " ".join(sent["tokens"])
                head_entity = sent["h"][0]
                tail_entity = sent["t"][0]
                gold_answers = [tail_entity]
                val_passages.append(sentence)
                val_entity_relations.append(white_space_fix(head_entity + " " + r_name))
                val_entities.append(white_space_fix(head_entity))
                val_contexts.append(
                    "answer: "
                    + white_space_fix(head_entity)
                    + " <SEP> "
                    + white_space_fix(r_name)
                    + " context: "
                    + white_space_fix(sentence)
                    + " </s>"
                )
                val_posterier_contexts.append(
                    "answer: "
                    + white_space_fix(head_entity)
                    + " <SEP> "
                    + white_space_fix(r_name)
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
            for sent in data[r_id]:
                sentence = " ".join(sent["tokens"])
                head_entity = sent["h"][0]
                tail_entity = sent["t"][0]
                gold_answers = [tail_entity]
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
                    + " context: "
                    + white_space_fix(sentence)
                    + " </s>"
                )
                test_posterier_contexts.append(
                    "answer: "
                    + white_space_fix(head_entity)
                    + " <SEP> "
                    + white_space_fix(r_name)
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
                    + " context: "
                    + white_space_fix(sentence)
                    + " </s>"
                )
                train_posterier_contexts.append(
                    "answer: "
                    + white_space_fix(head_entity)
                    + " <SEP> "
                    + white_space_fix(r_name)
                    + " "
                    + white_space_fix(" and ".join(gold_answers))
                    + " context: "
                    + white_space_fix(sentence)
                    + " </s>"
                )
                train_answers.append(
                    white_space_fix(" and ".join(gold_answers)) + " </s>"
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
    train_df = pd.read_csv(train_fewrel_path, sep="\t")
    dev_df = pd.read_csv(dev_fewrel_path, sep="\t")
    test_df = pd.read_csv(test_fewrel_path, sep="\t")

    train_passages = train_df["passages"].tolist()
    train_contexts = train_df["contexts"].tolist()
    train_answers = train_df["answers"].tolist()
    train_entity_relations = train_df["entity_relations"].tolist()
    train_entities = train_df["entities"].tolist()
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
        train_contexts = [
            ctx.replace("answer: ", "question: ") for ctx in train_contexts
        ]
        val_contexts = [ctx.replace("answer: ", "question: ") for ctx in val_contexts]
        test_contexts = [ctx.replace("answer: ", "question: ") for ctx in test_contexts]

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
    test_encodings["entity_relation_passage_input_ids"] = test_encodings.pop(
        "input_ids"
    )
    test_encodings["entity_relation_passage_attention_mask"] = test_encodings.pop(
        "attention_mask"
    )

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
