from transformers.models.albert.modeling_albert import AlbertGenerationDecoder, GenerationAlbertModel
import torch
from transformers import EncoderDecoderModel
from transformers import AlbertTokenizer, AlbertConfig
from transformers import EncoderDecoderConfig
import torch
import numpy as np
import torch.optim as optim
import datasets
from dataclasses import dataclass, field
from typing import Optional
from transformers.seq2seq_trainer import Seq2SeqTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# set the tokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
encoder_max_length=512
decoder_max_length=128
batch_size = 4

def process_data_to_model_inputs(batch, gpu=False):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["inputs"], padding="longest", truncation=False, return_tensors="pt")
  outputs = tokenizer(batch["outputs"], padding="longest", truncation=False, return_tensors="pt")

  batch["input_ids"] = inputs.input_ids if not gpu else inputs.input_ids.to('cuda')
  batch["attention_mask"] = inputs.attention_mask if not gpu else inputs.attention_mask.to('cuda')
  batch["decoder_input_ids"] = outputs.input_ids if not gpu else outputs.input_ids.to('cuda')
  batch["decoder_attention_mask"] = outputs.attention_mask if not gpu else outputs.attention_mask.to('cuda')
  batch["labels"] = outputs.input_ids.clone() if not gpu else outputs.input_ids.clone().to('cuda')

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
  # We have to make sure that the PAD token is ignored
 
  labels = torch.tensor([[torch.tensor(-100) if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]])
  batch["labels"] = labels if not gpu else labels.to('cuda')
  return batch

def glue_passage_question(bos_token, eos_token, passage, question=None):
    entries = {'cls': bos_token,
               'sep': eos_token,
               'passage': passage,
               'question': question}
    if question is not None:
        return '{passage} {sep} {question}'.format(**entries)
    return '{passage}'.format(**entries)

def create_albert2albert(tokenizer, gpu=False):
    albert_xxlarge_configuration = AlbertConfig()
    albert_xxlarge_configuration.decoder_start_token_id = tokenizer.bos_token_id
    albert_xxlarge_configuration.eos_token_id = tokenizer.eos_token_id
    albert_xxlarge_configuration.pad_token_id = tokenizer.pad_token_id
    albert_xxlarge_configuration.tie_encoder_decoder=False
    albert_xxlarge_configuration.architectures=["GenerationAlbertModel"]

    encoder_config = albert_xxlarge_configuration
    encoder = GenerationAlbertModel.from_pretrained("albert-xxlarge-v2", config=encoder_config)
    if gpu:
        encoder.to('cuda')

    albert_xxlarge_configuration.is_decoder=True
    albert_xxlarge_configuration.add_cross_attention=True
    albert_xxlarge_configuration.architectures=["AlbertGenerationDecoder"]
    decoder_config = albert_xxlarge_configuration
    decoder = AlbertGenerationDecoder.from_pretrained("albert-xxlarge-v2", config=decoder_config)
    if gpu:
        decoder.to('cuda')

    # This is to fix an issue in the generation version of the albert decoder.
    decoder.lm_head.lm_decoder = torch.nn.Linear(decoder_config.hidden_size, decoder_config.vocab_size, bias=True)

    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    albert2albert = EncoderDecoderModel(config=config, encoder=encoder, decoder=decoder)

    # set special tokens
    albert2albert.config.decoder_start_token_id = tokenizer.bos_token_id
    albert2albert.config.eos_token_id = tokenizer.eos_token_id
    albert2albert.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    albert2albert.config.vocab_size = albert2albert.config.decoder.vocab_size
    albert2albert.config.max_length = 142
    albert2albert.config.min_length = 56
    albert2albert.config.no_repeat_ngram_size = 3
    albert2albert.config.early_stopping = True
    albert2albert.config.length_penalty = 2.0
    albert2albert.config.num_beams = 4
    if gpu:
        albert2albert.to('cuda')
    return albert2albert

def process_race_row(row):
    option_code = row['answer']
    if option_code == 'A':
        option_idx = 0
    elif option_code == 'B':
        option_idx = 1
    elif option_code == 'C':
        option_idx = 2
    elif option_code == 'D':
        option_idx = 3

    answer = row['options'][option_idx]
    answer = ' '.join(answer.split())

    question = row['question']
    question = ' '.join(question.split())

    article = row['article']
    article = ' '.join(article.split())

    input_str = glue_passage_question(tokenizer.bos_token, tokenizer.eos_token, article, question)
    output_str = glue_passage_question(tokenizer.bos_token, tokenizer.eos_token, answer)

    return {'inputs': input_str, 'outputs': output_str}

# load rouge for validation
rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear", metadata={"help": f"Which lr scheduler to use."}
    )

# set training arguments - these params are not really tuned, feel free to change
'''
training_args = Seq2SeqTrainingArguments(
    output_dir="./",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    evaluate_during_training=True,
    do_train=True,
    do_eval=True,
    logging_steps=2,  # set to 1000 for full training
    save_steps=16,  # set to 500 for full training
    eval_steps=4,  # set to 8000 for full training
    warmup_steps=1,  # set to 2000 for full training
    max_steps=16, # delete for full training
    overwrite_output_dir=True,
    save_total_limit=3 
)'''

albert2albert = create_albert2albert(tokenizer, gpu=False)

train_dataset = load_dataset('race', 'all', split='train')
train_dataset = [process_race_row(train_dataset[i]) for i in range(len(train_dataset))]

dev_dataset = load_dataset('race', 'all', split='validation')
dev_dataset = [process_race_row(dev_dataset[i]) for i in range(len(dev_dataset))]

test_dataset = load_dataset('race', 'all', split='test')
test_dataset = [process_race_row(test_dataset[i]) for i in range(len(test_dataset))]
print(test_dataset[0])

exit()

train_data = train_dataset.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["article", "highlights", "id"]
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

exit()
# instantiate trainer
trainer = Seq2SeqTrainer(
    model=albert2albert,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()


passage = 'This is a long article to summarize'
question = 'What are we doing in this article ?'
answer = 'to summarize'
passage2 = 'This is a long article'
question2 = 'What are we doing here ?'
answer2 = 'to clearup my mind.'
batch = {}
batch['inputs'] = [glue_passage_question(tokenizer.bos_token, tokenizer.eos_token, passage, question)]
batch['outputs'] = [glue_passage_question(tokenizer.bos_token, tokenizer.eos_token, answer)]
batch['inputs'].append(glue_passage_question(tokenizer.bos_token, tokenizer.eos_token, passage2, question2))
batch['outputs'].append(glue_passage_question(tokenizer.bos_token, tokenizer.eos_token, answer2))
new_batch = process_data_to_model_inputs(batch, tokenizer)
del new_batch['inputs']
del new_batch['outputs']
optimizer = optim.Adam(filter(lambda p: p.requires_grad, albert2albert.parameters()), lr=0.001)
seq2seq = albert2albert(**new_batch)
loss = seq2seq.loss
predictions = seq2seq.logits.detach().cpu().tolist()
preds = predictions[0]
preds = np.argmax(preds, axis=1)
print(preds)
print(loss)
for i in range(3):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    seq2seq = albert2albert(**new_batch)
    loss = seq2seq.loss
    predictions = seq2seq.logits.detach().cpu().tolist()
    preds = predictions[0]
    preds = np.argmax(preds, axis=1)
    print(preds)
    print(loss)
