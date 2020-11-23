# Spencer's Hacky GPT-2 Question Answering Baseline
#
# Notes: 
# It is slow and takes about a minute or two to run, but can be parallelized
# It uses an undocumented API (kind of like webscraping, so not stable)
# It is non-deterministic
# I have done some basic param tuning. If you change the parameters too much you probably 
# won't get good results
# This will not work for all types of quesitons. For example if you want fill in the blank 
# questions you need to phrase differently
# It seems to do best with information extraction questions

import json
import requests

from collections import Counter

gpt2_url = "https://transformer.huggingface.co/autocomplete/gpt2/large"

def get_gpt2_answer(context, question, samples=10, temperature=0.2, top_p=1, max_time=5):
    payload= json.dumps({
        "context": f"{context}\nQ: {question}\nA: ",
        "model_size": "gpt2/large",
        "top_p": top_p,
        "temperature": temperature,
        "max_time": max_time
    })
    possible_answers = []
    for _ in range(samples):
        response = requests.post(gpt2_url, data=payload)
        for sentence in response.json()['sentences']:
            possible_answers.append(sentence['value'])

    answer_counter = Counter(possible_answers)
    return answer_counter.most_common(1)[0][0].split('\n')[0].strip()

if __name__ == "__main__":
    example_context_1 = 'No birds have lips. All birds have either a bill or a beak.'
    example_question_1 = 'Based on the text, what do no birds have?'
    example_answer_1 = get_gpt2_answer(example_context_1,example_question_1)
    
    print(f'Context: {example_context_1}')
    print(f'Q: {example_question_1}')
    print(f'A: {example_answer_1}')

    example_context_2 = 'Hypnos is a Greek god. He is the god of sleep. He lives in the cave where it is said that day and night meet.'
    example_question_2 = 'Who is Hypnos?'
    example_answer_2 = get_gpt2_answer(example_context_2,example_question_2)
    
    print(f'Context: {example_context_2}')
    print(f'Q: {example_question_2}')
    print(f'A: {example_answer_2}')
