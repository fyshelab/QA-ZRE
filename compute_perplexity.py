# Based on http://nlp.cs.washington.edu/zeroshot/evaluate.py
import pandas as pd
import os
import codecs
import re
import string
import sys
import numpy as np

PUNCTUATION = set(string.punctuation)

import re

def remove_latin(text):
    return re.sub(r'[^\x00-\x7f]',r'', text)

def unk_zero_re_eval(test_file, answer_file):
    q_aprf = unk_read_results(test_file, answer_file)
    return pretify(q_aprf)

def unk_read_results(test_set, answer_file):
    with codecs.open(test_set, "r", "utf-8") as fin:
        data = [line.strip().split("\t") for line in fin]
    metadata = [x[:4] for x in data]
    gold = [set(x[4:]) for x in data]

    with codecs.open(answer_file, "r", "utf-8") as fin:
        answers = [line.strip() for line in fin]

    new_answers = []
    for answer in answers[1:]:
        if answer != "no_answer":
            new_answers.append(answer)
        else:
            new_answers.append("")

    telemetry = []
    for m, g, a in zip(metadata, gold, new_answers):
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

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda"
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

import torch
def compute_perplexity_for_questions(main_path, file):
    ppls = []
    df = pd.read_csv(os.path.join(main_path, file), sep=',')
    questions = df["question_predictions"].tolist()
    for question in questions:
        encodings = tokenizer(question, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        b_sz, length = input_ids.size()
        target_ids = input_ids.clone()
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0]
            ppl = torch.exp(neg_log_likelihood)
        ppls.append(ppl)
    ppl = torch.stack(ppls).mean()
    return ppl

def gold_compute_perplexity_for_questions(main_path, file):
    ppls = []
    df = pd.read_csv(os.path.join(main_path, file), sep=',')
    inputs = df["input_str"].tolist()
    for inp in inputs:
        question = inp.split("context:")[0].replace("question:", "").strip()
        encodings = tokenizer(question, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        b_sz, length = input_ids.size()
        target_ids = input_ids.clone()
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0]
            ppl = torch.exp(neg_log_likelihood)
        ppls.append(ppl)
    ppl = torch.stack(ppls).mean()
    return ppl

def preprocess_the_prediction_files(main_path, list_of_files):
    for file in list_of_files:
        df = pd.read_csv(os.path.join(main_path, file), sep=',')
        df["predictions_str"].to_csv(os.path.join("/tmp/", file), sep='\t', header=True, index=False)

def unk_eval_the_prediction_files(list_of_files, gold_file):
    scores = {}
    scores_list = []
    precision_list = []
    recall_list = []
    for file in list_of_files:
        score = unk_zero_re_eval(gold_file, os.path.join("/tmp/", file))
        arr = score.split()
        f1_score = float(arr[-1][0:-1])
        precision = float(arr[1][0:-1])
        recall = float(arr[3][0:-1])
        scores[f1_score] = file
        scores_list.append(f1_score)
        precision_list.append(precision)
        recall_list.append(recall)

    f1s = np.array(scores_list)
    precisions = np.array(precision_list)
    recalls = np.array(recall_list)
    max_f1 = max(scores.keys())
    return scores[max_f1],  max_f1, f1s, scores, precisions, recalls

results = {}
for fold_i in range(1, 11, 1):
    results[fold_i] = {'mml-pgg-off-sim': {},
                       'mml-pgg-on-sim': {},
                       'mml-mml-off-sim': {},
                       'mml-mml-on-sim': {}}

print("# Evaluating the dev predictions on the RE-QA dataset on all folds for the tail entity generation task.")
# Evaluating the dev predictions on the RE-QA dataset on all folds for the tail entity generation task.
folders = ["mml-pgg-off-sim", "mml-pgg-on-sim", "mml-mml-off-sim", "mml-mml-on-sim"]

for fold_i in range(1, 11, 1):
    for folder in folders:
        fold_gold_file = "./zero-shot-extraction/relation_splits/dev.{}".format(fold_i-1)
        fold_path = "~/reqa-predictions/fold_{}/{}/".format(fold_i, folder)
        if fold_i == 1:
            fold_files = ["{}.fold.{}.dev.predictions.step.{}.csv".format(folder, fold_i, 100 * i) for i in range(1, 101, 1)]
        elif 2 <= fold_i <= 4:
            if folder == "mml-pgg-off-sim":
                fold_files = ["{}.fold.{}.dev.predictions.step.{}.csv".format(folder, fold_i, 100 * i) for i in range(1, 101, 1)]
            else:
                fold_files = ["{}.dev.predictions.fold.{}.step.{}.csv".format(folder, fold_i, 100 * i) for i in range(1, 101, 1)]
        else:
            if folder == "mml-pgg-off-sim":
                fold_files = ["{}.fold.{}.dev.predictions.step.{}.csv".format(folder, fold_i, 100 * i) for i in range(1, 201, 1)]
            else:
                fold_files = ["{}.dev.predictions.fold.{}.step.{}.csv".format(folder, fold_i, 100 * i) for i in range(1, 201, 1)]

        preprocess_the_prediction_files(fold_path, fold_files)
        max_file,  max_f1, f1s, scores, precisions, recalls = unk_eval_the_prediction_files(fold_files, fold_gold_file)
        print(folder, fold_i, max_file, max_f1)
        print("\n")
        results[fold_i][folder] = max_file
    print("NEXT")

print("# Evaluating the test predictions on the RE-QA dataset on all folds for the tail entity generation task.")
# Evaluating the test predictions on the RE-QA dataset on all folds for the tail entity generation task.
folders = ["mml-pgg-on-sim", "mml-mml-off-sim", "mml-mml-on-sim", "mml-pgg-off-sim"]
for folder in folders:
    avg_f1 = {"mml-mml-off-sim": 0, "mml-mml-on-sim": 0, "mml-pgg-on-sim": 0, "mml-pgg-off-sim": 0}
    avg_p = {"mml-mml-off-sim": 0, "mml-mml-on-sim": 0, "mml-pgg-on-sim": 0, "mml-pgg-off-sim": 0}
    avg_r = {"mml-mml-off-sim": 0, "mml-mml-on-sim": 0, "mml-pgg-on-sim": 0, "mml-pgg-off-sim": 0}
    for fold_i in range(1, 11, 1):
        fold_gold_file = "./zero-shot-extraction/relation_splits/test.{}".format(fold_i-1)
        fold_path = "~/reqa-predictions/fold_{}/{}".format(fold_i, folder)
        old_dev_file = results[fold_i][folder]
        new_test_file = old_dev_file.replace(".fold.{}.dev.predictions.".format(fold_i), ".test.predictions.fold.{}.".format(fold_i))
        new_test_file = new_test_file.replace(".dev.predictions.fold.{}.".format(fold_i), ".test.predictions.fold.{}.".format(fold_i))
        fold_files = [new_test_file]
        preprocess_the_prediction_files(fold_path, fold_files)
        max_file,  max_f1, f1s, scores, precisions, recalls = unk_eval_the_prediction_files(fold_files, fold_gold_file)
        print(folder, fold_i, max_file, max_f1)
        avg_f1[folder] += max_f1
        avg_p[folder] += precisions[0]
        avg_r[folder] += recalls[0]
        print("\n")

    print(folder, "f1", avg_f1[folder] / 10.0)
    print(folder, "p", avg_p[folder] / 10.0)
    print(folder, "r", avg_r[folder] / 10.0)
    print("NEXT")


print("# Compute perplexity over the test generated questions for the following method on the RE-QA dataset.")
# Compute perplexity over the test generated questions for the following method on the RE-QA dataset.
folders = ["mml-mml-off-sim"]

for folder in folders:
    avg_pp = {"mml-mml-off-sim": 0}
    for fold_i in range(9, 11, 1):
        fold_path = "~/reqa-predictions/fold_{}/{}".format(fold_i, folder)
        old_dev_file = results[fold_i][folder]
        new_test_file = old_dev_file.replace(".fold.{}.dev.predictions.".format(fold_i), ".test.predictions.fold.{}.".format(fold_i))
        new_test_file = new_test_file.replace(".dev.predictions.fold.{}.".format(fold_i), ".test.predictions.fold.{}.".format(fold_i))
        fold_file = new_test_file
        pp = compute_perplexity_for_questions(fold_path, fold_file)
        avg_pp[folder] += pp
        print(fold_file, pp)
    print("\n")
    print(folder, "pp", avg_pp[folder] / 10.0)


print("# Compute perplexity over the test generated questions for the following method on the RE-QA dataset.")
# Compute perplexity over the test generated questions for the following method on the RE-QA dataset.
folders = ["mml-pgg-off-sim"]

for folder in folders:
    avg_pp = {"mml-pgg-off-sim": 0}
    for fold_i in range(1, 11, 1):
        fold_path = "~/reqa-predictions/fold_{}/{}".format(fold_i, folder)
        old_dev_file = results[fold_i][folder]
        new_test_file = old_dev_file.replace(".fold.{}.dev.predictions.".format(fold_i), ".test.predictions.fold.{}.".format(fold_i))
        new_test_file = new_test_file.replace(".dev.predictions.fold.{}.".format(fold_i), ".test.predictions.fold.{}.".format(fold_i))
        fold_file = new_test_file
        pp = compute_perplexity_for_questions(fold_path, fold_file)
        avg_pp[folder] += pp
        print(fold_file, pp)
    print("\n")
    print(folder, "pp", avg_pp[folder] / 10.0)

print("# Compute perplexity over the test generated questions for the following method on the RE-QA dataset.")
# Compute perplexity over the test generated questions for the following method on the RE-QA dataset.
folders = ["mml-pgg-on-sim"]

for folder in folders:
    avg_pp = {"mml-pgg-on-sim": 0}
    for fold_i in range(1, 11, 1):
        fold_path = "~/reqa-predictions/fold_{}/{}".format(fold_i, folder)
        old_dev_file = results[fold_i][folder]
        new_test_file = old_dev_file.replace(".fold.{}.dev.predictions.".format(fold_i), ".test.predictions.fold.{}.".format(fold_i))
        new_test_file = new_test_file.replace(".dev.predictions.fold.{}.".format(fold_i), ".test.predictions.fold.{}.".format(fold_i))
        fold_file = new_test_file
        pp = compute_perplexity_for_questions(fold_path, fold_file)
        avg_pp[folder] += pp
        print(fold_file, pp)
    print("\n")
    print(folder, "pp", avg_pp[folder] / 10.0)

print("# Compute perplexity over the test generated questions for the following method on the RE-QA dataset.")
# Compute perplexity over the test generated questions for the following method on the RE-QA dataset.
folders = ["mml-mml-on-sim"]

for folder in folders:
    avg_pp = {"mml-mml-on-sim": 0}
    for fold_i in range(1, 11, 1):
        fold_path = "~/reqa-predictions/fold_{}/{}".format(fold_i, folder)
        old_dev_file = results[fold_i][folder]
        new_test_file = old_dev_file.replace(".fold.{}.dev.predictions.".format(fold_i), ".test.predictions.fold.{}.".format(fold_i))
        new_test_file = new_test_file.replace(".dev.predictions.fold.{}.".format(fold_i), ".test.predictions.fold.{}.".format(fold_i))
        fold_file = new_test_file
        pp = compute_perplexity_for_questions(fold_path, fold_file)
        avg_pp[folder] += pp
        print(fold_file, pp)
    print("\n")
    print(folder, "pp", avg_pp[folder] / 10.0)

print("# PP for the base-base predictions on the RE-QA dataset.")
avg_pp = 0.0
for fold_i in range(1, 11, 1):
    fold_path = "~/reqa-predictions/fold_{}/".format(fold_i)
    fold_file = "base-base.test.predictions.fold.{}.csv".format(fold_i)
    pp = compute_perplexity_for_questions(fold_path, fold_file)
    avg_pp += pp
    print(fold_file, pp)

print("\n")
print("pp", avg_pp / 10.0)
