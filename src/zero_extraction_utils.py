from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.question_response_generation.question_utils import \
    create_data_for_question_generation


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
