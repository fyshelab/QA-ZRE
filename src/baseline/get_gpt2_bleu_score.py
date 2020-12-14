#%%
import pandas as pd
import re

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pathlib import Path

clean_regex = re.compile(r'[^\w\s]')
split_regex = re.compile(r'\s+')

repo_root = Path(__file__).parent.parent.parent
clean_dir = repo_root/'data/dreamscape-clean'

chencherry = SmoothingFunction()

def tokenize_string(string):
    clean_string = clean_regex.sub('',string.lower())
    return split_regex.split(clean_string)

def get_bleu_score(row):
    correct_tokens = tokenize_string(row['correct_answer'])
    gpt2_tokens = tokenize_string(row['gpt2_answer'])

    # Handle division by 0 case with smoothing
    # if len(gpt2_tokens)==1:
    #     return sentence_bleu([correct_tokens],gpt2_tokens, weights=(1,))

    # Method 4 should penalize longer answers less
    return sentence_bleu([correct_tokens],gpt2_tokens,smoothing_function=chencherry.method3)

df = pd.read_csv(clean_dir/'gpt2_predictions.csv')
# %%
df['bleu_score'] = df.apply(get_bleu_score,axis=1)
# %%
print(f"Mean: {df['bleu_score'].mean()}")
print(f"Median: {df['bleu_score'].median()}")
# %%
df.to_csv(clean_dir/'gpt2_bleu_scores.csv',encoding='utf8', index = False)
# %%
