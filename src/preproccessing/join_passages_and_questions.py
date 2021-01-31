#%%
# Preprocess Dreamscape Dataset
# VS Code Notebook

import argparse
#%% Imports
import json
from io import StringIO
from pathlib import Path

import pandas as pd


def filter_data(args):
    repo_root = Path(__file__).parent.parent.parent
    output_dir = repo_root / "data/dreamscape-clean"
    #%% Generate clean passages.csv
    passages_df = pd.read_csv(
        repo_root / "data/dreamscape/passages-jan29-2021.csv",
    )

    passages_out_df = passages_df[passages_df["gradeId"] == args.grade]

    passages_out_df[["id", "genreId", "gradeId", "text"]].to_csv(
        output_dir / "passages-jan29-2021.csv", encoding="utf8", index=False
    )

    #%% Generate clean questions.csv
    questions_df = pd.read_csv(repo_root / "data/dreamscape/questions-jan29-2021.csv")

    questions_out_df = questions_df[questions_df["questionTypeId"] == 1]
    questions_out_df = questions_df[questions_df["Question Type"] == args.type]
    questions_out_df = questions_out_df[
        questions_df["passageId"].isin(passages_out_df["id"])
    ]

    questions_out_df[
        [
            "id",
            "passageId",
            "questionTypeId",
            "question",
            "potentialAnswers",
            "correctAnswers",
            "Question Type",
        ]
    ].to_csv(
        output_dir / "literal-questions-jan29-2021.csv", encoding="utf8", index=False
    )
    #%% Join dfs and parse json data
    join_df = passages_out_df.merge(
        questions_out_df, left_on="id", right_on="passageId"
    )

    def transform_to_questions_answers_format(row):
        row["correctAnswers"] = json.loads(row["correctAnswers"])
        row["potentialAnswers"] = json.loads(row["potentialAnswers"])
        row["wrongAnswers"] = [
            x for x in row["potentialAnswers"] if x not in row["correctAnswers"]
        ]
        return row

    question_answer_df = join_df.apply(transform_to_questions_answers_format, axis=1)
    # %% Generate final json output
    groups = question_answer_df.groupby(by=["passageId", "text"])

    final_json_out = []

    for group_name, group_df in groups:
        group_df[["question", "correctAnswers", "wrongAnswers"]]

        final_json_out.append(
            {
                "passageId": int(group_df.iloc[0]["passageId"]),
                "text": group_df.iloc[0]["text"],
                "questions_answers": list(
                    group_df[["question", "correctAnswers", "wrongAnswers"]].to_dict(
                        "records"
                    )
                ),
            }
        )

    with open(
        output_dir / "literal-jan29-2021-passages_questions_answers.csv", "w"
    ) as out_file:
        out_file.write(json.dumps(final_json_out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grade",
        type=int,
        required=True,
    )
    parser.add_argument("--type", type=str, required=True, help="Literal|Inferential")
    parser.add_argument(
        "--output_name",
        type=str,
        required=False,
    )
    args, _ = parser.parse_known_args()
    filter_data(args)
# %%
