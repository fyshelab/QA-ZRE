from transformers.models.albert.modeling_albert import AlbertGenerationDecoder, GenerationAlbertModel
import torch
from transformers import EncoderDecoderModel
from transformers import AlbertTokenizer, AlbertConfig
from transformers import EncoderDecoderConfig
import torch
import numpy as np
import torch.optim as optim



# set the tokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
albert_xxlarge_configuration = AlbertConfig()
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
encoder_max_length=512
decoder_max_length=128

def process_data_to_model_inputs(batch, tokenizer):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["inputs"], padding="longest", truncation=False, return_tensors="pt")
  outputs = tokenizer(batch["outputs"], padding="longest", truncation=False, return_tensors="pt")

  batch["input_ids"] = inputs.input_ids.to('cuda')
  batch["attention_mask"] = inputs.attention_mask.to('cuda')
  batch["decoder_input_ids"] = outputs.input_ids.to('cuda')
  batch["decoder_attention_mask"] = outputs.attention_mask.to('cuda')
  batch["labels"] = outputs.input_ids.clone().to('cuda')

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  batch["labels"] = torch.tensor([[torch.tensor(-100) if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]).to('cuda')

  return batch

def glue_passage_question(bos_token, eos_token, passage, question=None):
    entries = {'cls': bos_token,
               'sep': eos_token,
               'passage': passage,
               'question': question}
    if question is not None:
        return '{passage} {sep} {question}'.format(**entries)
    return '{passage}'.format(**entries)

albert_xxlarge_configuration.tie_encoder_decoder=False
albert_xxlarge_configuration.architectures=["GenerationAlbertModel"]
encoder_config = albert_xxlarge_configuration
encoder = GenerationAlbertModel.from_pretrained("albert-xxlarge-v2", config=encoder_config)
encoder.to('cuda')
albert_xxlarge_configuration.is_decoder=True
albert_xxlarge_configuration.add_cross_attention=True
albert_xxlarge_configuration.architectures=["AlbertGenerationDecoder"]
decoder_config = albert_xxlarge_configuration
print(decoder_config)
decoder = AlbertGenerationDecoder.from_pretrained("albert-xxlarge-v2", config=decoder_config)
decoder.to('cuda')
decoder.lm_head.lm_decoder = torch.nn.Linear(decoder_config.hidden_size, decoder_config.vocab_size, bias=True)

config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
albert2albert = EncoderDecoderModel(config=config, encoder=encoder, decoder=decoder)


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
albert2albert.to('cuda')
optimizer = optim.Adam(filter(lambda p: p.requires_grad, albert2albert.parameters()), lr=0.001)
seq2seq = albert2albert(**new_batch)
loss = seq2seq.loss
predictions = seq2seq.logits.detach().cpu().tolist()
preds = predictions[0]
preds = np.argmax(preds, axis=1)
print(preds)
print(loss)
for i in range(10):
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
