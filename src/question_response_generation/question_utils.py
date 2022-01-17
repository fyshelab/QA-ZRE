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
        answer = random.choice(row["answers"])["text"]
        return {
            "article": white_space_fix(article),
            "question": white_space_fix(question),
            "answer": white_space_fix(answer),
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


def q_only_read_squad_dataset():
    def process_squad_row(row):
        context = row["context"]
        question = row["question"]
        if row["answers"]["text"]:
            answ = random.choice(row["answers"]["text"])
            return {
                "article": white_space_fix(context),
                "question": white_space_fix(question),
                "answer": white_space_fix(answ),
            }
        else:
            return {
                "article": "NONE",
                "answer": "NONE",
                "question": "NONE",
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


def q_only_read_drop_dataset():
    def process_drop_row(row):
        context = row["passage"]
        question = row["question"]
        if row["answers_spans"]["spans"]:
            answ = random.choice(row["answers_spans"]["spans"])
            return {
                "article": white_space_fix(context),
                "question": white_space_fix(question),
                "answer": white_space_fix(answ),
            }
        else:
            return {
                "article": "NONE",
                "answer": "NONE",
                "question": "NONE",
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


def create_data_for_question_pretrain():
    nq_train_dataset, _, _ = q_only_read_narrative_dataset()
    squad_train_dataset, _, _ = q_only_read_squad_dataset()
    drop_train_dataset, _, _ = q_only_read_drop_dataset()

    train_contexts = []
    train_questions = []
    train_answers = []
    for row in nq_train_dataset:
        train_contexts.append(row["article"])
        train_questions.append(row["question"])
        train_answers.append(row["answer"])

    for row in squad_train_dataset:
        train_contexts.append(row["article"])
        train_questions.append(row["question"])
        train_answers.append(row["answer"])

    for row in drop_train_dataset:
        train_contexts.append(row["article"])
        train_questions.append(row["question"])
        train_answers.append(row["answer"])

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
        if len(q_ents) > 0:
            q_entity = random.choice(q_ents)[0]
            all_entities = " ".join([ent[0] for ent in q_ents])
            new_final_doc = []
            for token in final_doc:
                if token not in all_entities:
                    new_final_doc.append(token)
            if new_final_doc:
                good_qs.append(
                    (
                        train_contexts[index],
                        q,
                        " ".join(new_final_doc),
                        q_entity,
                        train_answers[index],
                    )
                )

    print(len(good_qs))
    contexts = []
    questions = []
    for row in good_qs:
        context = row[0]
        question = row[1]
        answer = row[4]
        tokens = row[2].split(" ")
        if len(tokens) > 0 and len(tokens) < 4:
            relation_signal = " ".join(tokens)

        if len(tokens) >= 4:
            token_num = random.randint(1, 4)
            sampled_tokens = random.sample(tokens, token_num)
            relation_signal = " ".join(sampled_tokens)
        else:
            continue
        contexts.append(
            "answer: "
            + white_space_fix(row[3])
            + " <SEP> "
            + white_space_fix(relation_signal)
            + " "
            + white_space_fix(answer)
            + " context: "
            + white_space_fix(context)
            + " </s>"
        )
        questions.append(white_space_fix(question) + " </s>")

    return contexts, questions


def create_question_pretrain_dataset(
    question_tokenizer,
    batch_size,
    source_max_length,
    decoder_max_length,
    distributed=True,
    num_workers=1,
):
    """Function to create the question input-outputs to do the pretraining of the question generator."""

    contexts, questions = create_data_for_question_pretrain()
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
