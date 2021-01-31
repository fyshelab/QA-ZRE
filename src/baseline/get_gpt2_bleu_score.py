#%%
# Generate bleu and nist scores for gpt2 daat
# VS Code Notebook

import re
from pathlib import Path

#%%
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.nist_score import sentence_nist

clean_regex = re.compile(r"[^\w\s]")
split_regex = re.compile(r"\s+")

repo_root = Path(__file__).parent.parent.parent
clean_dir = repo_root / "data/dreamscape-clean"

chencherry = SmoothingFunction()


def tokenize_string(string):
    """Breaks strings up into tokens and cleans them."""
    clean_string = clean_regex.sub("", string.lower())
    return split_regex.split(clean_string)


df = pd.read_csv(clean_dir / "gpt2_predictions.csv")
# %%
def get_bleu_score(row):
    """Generate bleu score for row."""
    correct_tokens = tokenize_string(row["correct_answer"])
    gpt2_tokens = tokenize_string(row["gpt2_answer"])

    # Maybe there is a better smoothing method other than 3
    return sentence_bleu(
        [correct_tokens], gpt2_tokens, smoothing_function=chencherry.method3
    )


df["bleu_score"] = df.apply(get_bleu_score, axis=1)
print(f"Mean: {df['bleu_score'].mean()}")
print(f"Median: {df['bleu_score'].median()}")
#%%
def get_nist_score(row):
    """Generate nist score for row."""
    correct_tokens = tokenize_string(row["correct_answer"])
    gpt2_tokens = tokenize_string(row["gpt2_answer"])

    return sentence_nist(
        [correct_tokens], gpt2_tokens, n=min(len(correct_tokens), len(gpt2_tokens), 5)
    )


df["nist_score"] = df.apply(get_nist_score, axis=1)
print(f"Mean: {df['nist_score'].mean()}")
print(f"Median: {df['nist_score'].median()}")
# %%
# %%
df.to_csv(clean_dir / "gpt2_scores.csv", encoding="utf8", index=False)
# %%
