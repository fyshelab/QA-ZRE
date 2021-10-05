import random

import spacy
import torch
from datasets import load_dataset
from spacy.lang.en.stop_words import STOP_WORDS
from torch.utils.data import DataLoader


def white_space_fix(text):
    return " ".join(text.split())


def q_only_read_narrative_dataset():
    """Read the narrative qa dataset for question generation"""

    def process_narrative_row(row):
        """Helper functions for NarrativeQA Dataset."""
        question = row["question"]["text"]
        article = row["document"]["summary"]["text"]
        return {
            "article": white_space_fix(article),
            "question": white_space_fix(question),
        }

    train_dataset = load_dataset("narrativeqa", split="train")
    dev_dataset = load_dataset("narrativeqa", split="validation")
    test_dataset = load_dataset("narrativeqa", split="test")

    train_dataset = train_dataset.map(
        process_narrative_row,
        remove_columns=["document", "answers"],
    )

    dev_dataset = dev_dataset.map(
        process_narrative_row,
        remove_columns=["document", "answers"],
    )

    test_dataset = test_dataset.map(
        process_narrative_row,
        remove_columns=["document", "answers"],
    )
    return train_dataset, dev_dataset, test_dataset


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


def q_only_read_squad_dataset():
    def process_squad_row(row):
        context = row["context"]
        question = row["question"]
        return {
            "article": white_space_fix(context),
            "question": white_space_fix(question),
        }

    train_dataset = load_dataset("squad_v2", split="train")
    train_dataset = train_dataset.map(
        process_squad_row,
        remove_columns=["id", "title", "context", "answers"],
    ).filter(lambda row: row["article"] != "NONE")
    dev_dataset = load_dataset("squad_v2", split="validation")
    dev_dataset = dev_dataset.map(
        process_squad_row,
        remove_columns=["id", "title", "context", "answers"],
    ).filter(lambda row: row["article"] != "NONE")
    return train_dataset, dev_dataset, dev_dataset


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
        else:
            return {
                "article": "NONE",
                "answer": "NONE",
            }

    train_dataset = load_dataset("squad_v2", split="train")
    train_dataset = train_dataset.map(
        process_squad_row,
        remove_columns=["id", "title", "context", "question", "answers"],
    ).filter(lambda row: row["article"] != "NONE")
    dev_dataset = load_dataset("squad_v2", split="validation")
    dev_dataset = dev_dataset.map(
        process_squad_row,
        remove_columns=["id", "title", "context", "question", "answers"],
    ).filter(lambda row: row["article"] != "NONE")
    return train_dataset, dev_dataset, dev_dataset


def q_only_read_drop_dataset():
    def process_drop_row(row):
        context = row["passage"]
        question = row["question"]
        return {
            "article": white_space_fix(context),
            "question": white_space_fix(question),
        }

    train_dataset = load_dataset("drop", split="train")
    train_dataset = train_dataset.map(
        process_drop_row,
        remove_columns=[
            "passage",
            "answers_spans",
        ],
    ).filter(lambda row: row["article"] != "NONE")
    dev_dataset = load_dataset("drop", split="validation")
    dev_dataset = dev_dataset.map(
        process_drop_row,
        remove_columns=[
            "passage",
            "answers_spans",
        ],
    ).filter(lambda row: row["article"] != "NONE")
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
        else:
            return {
                "article": "NONE",
                "answer": "NONE",
            }

    train_dataset = load_dataset("drop", split="train")
    train_dataset = train_dataset.map(
        process_drop_row,
        remove_columns=[
            "passage",
            "question",
            "answers_spans",
        ],
    ).filter(lambda row: row["article"] != "NONE")
    dev_dataset = load_dataset("drop", split="validation")
    dev_dataset = dev_dataset.map(
        process_drop_row,
        remove_columns=[
            "passage",
            "question",
            "answers_spans",
        ],
    ).filter(lambda row: row["article"] != "NONE")
    return train_dataset, dev_dataset, dev_dataset


def create_question_dataset(
    tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    distributed=False,
    num_workers=0,
    dataset="all",
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

    if dataset == "all":
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
    elif dataset == "squad_v2":
        train_dataset, dev_dataset, test_dataset = q_read_squad_dataset()
        train_dataset, dev_dataset, test_dataset = dataset_to_pytorch(
            train_dataset, dev_dataset, test_dataset
        )
    elif dataset == "narrativeqa":
        train_dataset, dev_dataset, test_dataset = q_read_narrative_dataset()
        train_dataset, dev_dataset, test_dataset = dataset_to_pytorch(
            train_dataset, dev_dataset, test_dataset
        )
    elif dataset == "drop":
        train_dataset, dev_dataset, test_dataset = q_read_drop_dataset()
        train_dataset, dev_dataset, test_dataset = dataset_to_pytorch(
            train_dataset, dev_dataset, test_dataset
        )
    else:
        raise ("Unknown dataset {0}".format(dataset))

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


def create_data_for_question_generation():
    nq_train_dataset, _, _ = q_only_read_narrative_dataset()
    squad_train_dataset, _, _ = q_only_read_squad_dataset()
    drop_train_dataset, _, _ = q_only_read_drop_dataset()

    train_contexts = []
    train_questions = []
    for row in nq_train_dataset:
        train_contexts.append(row["article"])
        train_questions.append(row["question"])

    for row in squad_train_dataset:
        train_contexts.append(row["article"])
        train_questions.append(row["question"])

    for row in drop_train_dataset:
        train_contexts.append(row["article"])
        train_questions.append(row["question"])

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

    STOP_WORDS.update(stop_list)

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
        if len(tokens) > 0 and len(tokens) < 6:
            relation_signal = " ".join(tokens)

        if len(tokens) >= 6:
            token_num = random.randint(3, len(tokens))
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


def create_question_generation_dataset(
    question_tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    distributed=True,
    num_workers=1,
):
    """Function to create the question input-outputs to do the second stage training of the question generator."""

    contexts, questions = create_data_for_question_generation()
    train_encodings = question_tokenizer(
        contexts,
        truncation=True,
        padding="max_length",
        max_length=source_max_length,
        add_special_tokens=False,
    )
    train_answer_encodings = question_tokenizer(
        questions,
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
        [
            -100 if token == question_tokenizer.pad_token_id else token
            for token in labels
        ]
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

    train_dataset = None
    train_sampler = None
    train_loader = None
    train_dataset = HelperDataset(train_encodings)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
        )
        return train_loader, train_dataset, train_sampler
    if not distributed:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, train_dataset, None
