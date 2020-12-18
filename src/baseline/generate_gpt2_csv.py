# Generate GPT2 responses to questions

import json
import pandas as pd

from joblib import Parallel, delayed
from pathlib import Path

from gpt2 import get_gpt2_answer

repo_root = Path(__file__).parent.parent.parent
clean_dir = repo_root/'data/dreamscape-clean'

with open(clean_dir/'passages_questions_answers.json') as input_file:
    data = json.loads(input_file.read())


def get_rows(passage):
    """
    Generates rows for a given passage
    """
    rows = []
    passage_id= passage['passageId']
    print(f"Processing: {passage_id}")
    text = passage['text']
    for question in passage['questions_answers']:
        question_text = question['question']

        # Retry if connection issue
        try:
            gpt2_answer = get_gpt2_answer(text,question_text)
        except:
            gpt2_answer = get_gpt2_answer(text,question_text)

        correct_answer = question['correctAnswers'][0] # Only 1 correct answers in set used

        rows.append({
            "passage_id":passage_id,
            "passage_text":text,
            "question_text":question_text,
            "correct_answer":correct_answer,
            'gpt2_answer': gpt2_answer,
        })
    return rows

rows = []


results = Parallel(n_jobs=16)(delayed(get_rows)(passage) for passage in data)

results_flat = [x for y in results for x in y]

out_df = pd.DataFrame(results_flat)
out_df.to_csv(clean_dir/'gpt2_predictions.csv',encoding='utf8', index = False)