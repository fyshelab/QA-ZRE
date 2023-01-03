import collections
import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd

# implementation of the squad evaluation. Basic Average F1 from the squad v2 evaluation script.


def removesuffix(self: str, suffix: str) -> str:
    if self.endswith(suffix):
        return self[: -len(suffix)]
    else:
        return self[:]


def read_pred_file(pred_file_name):
    df = pd.read_csv(pred_file_name).astype(str)
    predictions = df["predictions_str"].tolist()
    normal_preds = [removesuffix(pred, " </s>") for pred in predictions]
    return normal_preds


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def read_squad_refs(path):
    path = Path(path)
    with open(path, "rb") as f:
        squad_dict = json.load(f)

    all_refs = []
    for group in squad_dict["data"]:
        for passage in group["paragraphs"]:
            for qa in passage["qas"]:
                gold_answers = [
                    a["text"] for a in qa["answers"] if normalize_answer(a["text"])
                ]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = [""]
                all_refs.append(gold_answers)

    return all_refs


def get_raw_scores(squad_path, preds):
    exact_scores = {}
    f1_scores = {}
    all_refs = read_squad_refs(squad_path)
    for i, gold_answers in enumerate(all_refs):
        a_pred = preds[i]
        if a_pred == "no_answer":
            # our model will generate the special no_answer token.
            # map it to the empty string if you want to compare with squad's gold output for unanswerable questions.
            a_pred = ""
        # Take max over all gold answers
        exact_scores[i] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[i] = max(compute_f1(a, a_pred) for a in gold_answers)

    mean_f1 = 100.0 * sum(f1_scores[k] for k in range(len(all_refs))) / len(all_refs)
    return exact_scores, f1_scores, mean_f1


def compute_response_f1(gold_file, prediction_file):
    """Compute the mean_f1 on the squads eval data."""
    preds = read_pred_file(prediction_file)
    exact_scores, f1_scores, mean_f1 = get_raw_scores(gold_file, preds)
    return mean_f1


def compute_macro_PRF(predicted_idx, gold_idx, i=-1, empty_label=None):
    """This evaluation function follows work from Sorokin and
    Gurevych(https://www.aclweb.org/anthology/D17-1188.pdf) code borrowed from
    the following link:

    https://github.com/UKPLab/emnlp2017-relation-
    extraction/blob/master/relation_extraction/evaluation/metrics.py
    """
    if i == -1:
        i = len(predicted_idx)

    complete_rel_set = set(gold_idx) - {empty_label}
    avg_prec = 0.0
    avg_rec = 0.0

    for r in complete_rel_set:
        r_indices = predicted_idx[:i] == r
        tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
        tp_fp = len(r_indices.nonzero()[0])
        tp_fn = len((gold_idx == r).nonzero()[0])
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        # print(id_to_labels[r], prec, rec, 2.0 * prec * rec / (prec + rec))
        avg_prec += prec
        avg_rec += rec
    f1 = 0
    avg_prec = avg_prec / len(set(predicted_idx[:i]))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec + avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

    return avg_prec, avg_rec, f1


def compute_relation_score(gold_file, prediction_file):
    df = pd.read_csv(gold_file, sep=",")
    gold_indices = np.array(df["gold_indices"].tolist())
    num_unseen_relations = np.amax(gold_indices) + 1
    gold_indices = np.reshape(gold_indices, (-1, num_unseen_relations))
    num_examples, num_unseen_relations = gold_indices.shape
    pred_log_ps = pd.read_csv(prediction_file, sep=",")["answer_log_p"].tolist()
    pred_log_ps = np.log(
        np.mean(
            np.reshape(
                np.exp(np.array(pred_log_ps)), (num_examples, num_unseen_relations, -1)
            ),
            axis=2,
        )
    )
    pred_ids = np.argmax(pred_log_ps, axis=1)
    prec, rec, f1 = compute_macro_PRF(pred_ids, gold_indices)
    return prec, rec, f1
