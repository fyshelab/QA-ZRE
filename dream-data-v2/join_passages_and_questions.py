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
    output_dir = repo_root / "."
    #%% Generate clean passages.csv
    passages_df = pd.read_csv(
        repo_root / "./passages-jan29-2021.csv",
    )

    passages_out_df = passages_df

    passages_out_df[["id", "genreId", "gradeId", "text"]].to_csv(
        output_dir / "passages-cleaned.csv", encoding="utf8", index=False
    )

    #%% Generate clean questions.csv
    questions_df = pd.read_csv(repo_root / "./questions-jan29-2021.csv")

    questions_out_df = questions_df
    questions_out_df = questions_out_df[
        questions_df["passageId"].isin(passages_out_df["id"])
    ]


    questions_out_df = questions_out_df[(questions_out_df.skillId == 3) & (questions_out_df.questionTypeId == 1) & (questions_out_df.QuestionType == "Inferential")]
    questions_out_df[
        [
            "id",
            "skillId",
            "passageId",
            "questionTypeId",
            "question",
            "potentialAnswers",
            "correctAnswers",
            "QuestionType",
        ]
    ].to_csv(
        output_dir / "questions-cleaned.csv", encoding="utf8", index=False
    )
    #%% Join dfs and parse json data
    join_df = passages_out_df[["id", "genreId", "gradeId", "text"]].merge(
        questions_out_df[
        [
            "passageId",
            "skillId",
            "questionTypeId",
            "question",
            "potentialAnswers",
            "correctAnswers",
            "QuestionType",
        ]
    ], left_on="id", right_on="passageId"
    )

    join_df.to_csv(
        output_dir / "summary-passages-questions-cleaned.inferentials.csv", encoding="utf8", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    filter_data(args)
# %%
