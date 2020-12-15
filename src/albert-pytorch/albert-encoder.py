from transformers.models.albert.modeling_albert import AlbertGenerationDecoder, GenerationAlbertModel
from transformers import EncoderDecoderModel
from transformers import AlbertTokenizer, AlbertConfig
from transformers import EncoderDecoderConfig
import torch
import numpy as np
import torch.optim as optim

albert_xxlarge_configuration = AlbertConfig()
albert_xxlarge_configuration.tie_encoder_decoder=False
albert_xxlarge_configuration.bos_token_id=101
albert_xxlarge_configuration.eos_token_id=101
albert_xxlarge_configuration.architectures=["GenerationAlbertModel"]
encoder_config = albert_xxlarge_configuration

encoder = GenerationAlbertModel.from_pretrained("albert-xxlarge-v2", config=encoder_config)
encoder.to('cuda')
albert_xxlarge_configuration.is_decoder=True
albert_xxlarge_configuration.add_cross_attention=True
albert_xxlarge_configuration.architectures=["AlbertGenerationDecoder"]
decoder_config = albert_xxlarge_configuration

decoder = AlbertGenerationDecoder.from_pretrained("albert-xxlarge-v2", config=decoder_config)
decoder.to('cuda')
decoder.lm_head.lm_decoder = torch.nn.Linear(decoder_config.hidden_size, decoder_config.vocab_size, bias=True)

config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
albert2albert = EncoderDecoderModel(config=config, encoder=encoder, decoder=decoder)
tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
input_ids = tokenizer('This is a long article to summarize', add_special_tokens=False, return_tensors="pt")
labels = tokenizer('This is a short summary', return_tensors="pt")
input_ids.to('cuda')
labels.to('cuda')
albert2albert.to('cuda')
albert2albert.eval()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, albert2albert.parameters()), lr=0.001)
seq2seq = albert2albert(input_ids=input_ids.input_ids, decoder_input_ids=labels.input_ids, labels=labels.input_ids)
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
    seq2seq = albert2albert(input_ids=input_ids.input_ids, decoder_input_ids=labels.input_ids, labels=labels.input_ids)
    loss = seq2seq.loss
    predictions = seq2seq.logits.detach().cpu().tolist()
    preds = predictions[0]
    preds = np.argmax(preds, axis=1)
    print(preds)
    print(loss)
