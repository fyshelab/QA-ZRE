"""Implementation of the albert Module for Response Generation.

Some parts are from:
https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer
https://github.com/google-research/albert
"""
import copy
import io
import json
import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy
import numpy as np
import six
import torch
import torch.nn as nn

from src.initialize import xavier_param_init


def set_random_seed(seed: int) -> Any:
    """Set the random seed, which initializes the random number generator.

    Ensures that runs are reproducible and eliminates differences due to
    randomness.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AlbertTokenEmbedding(nn.Module):
    """Creates embedding tables for different token types used in the Albert
    Model."""

    def __init__(
        self,
        pad_id: int,
        vocab_size: Optional[int] = 1,
        dim_embeddings: Optional[int] = 1,
        lookup_table: Optional[numpy.ndarray] = None,
    ) -> None:
        """Constructs token embeddings for words, toke_types, and position
        features."""
        super(AlbertTokenEmbedding, self).__init__()

        self.pad_id = pad_id
        if lookup_table is not None:
            lookup_t = torch.FloatTensor(lookup_table)
            self.token_embs = nn.Embedding.from_pretrained(lookup_t)
        else:
            self.token_embs = nn.Embedding(vocab_size, dim_embeddings)

        self.token_embs.weight.data[self.pad_id].fill_(0.0)

    def set_trainable(self, trainable: Optional[bool] = True) -> None:
        """Set the freeze embeddings during training or not."""
        self.token_embs.weight.requires_grad = trainable

    def forward(self, *args: List[Any]) -> torch.FloatTensor:
        """Zero the vector for the pad tokens, and then retreive the vector of
        each token."""
        token_indices = args[0]
        self.token_embs.weight.data[self.pad_id].fill_(0.0)
        return self.token_embs(token_indices)


class AlbertEmbedding(nn.Module):
    """AlbertEmbedding: word embeddings, token_type embeddings, position embeddings."""

    def __init__(
        self,
        vocab_size: int,
        word_pad_id: int,
        token_type_pad_id: int,
        embedding_size: Optional[int] = 128,
        token_type_vocab_size: Optional[int] = 16,
        use_position_embeddings: Optional[bool] = True,
        max_position_embeddings: Optional[int] = 512,
        dropout: Optional[float] = 0.1,
    ) -> None:

        """The designed 3 embeddings of Albert."""
        super(AlbertEmbedding, self).__init__()

        self.word_embedder = AlbertTokenEmbedding(
            pad_id=word_pad_id, vocab_size=vocab_size, dim_embeddings=embedding_size
        )
        self.word_embedder.set_trainable(trainable=True)

        # For token_type features.
        self.token_type_embedder = AlbertTokenEmbedding(
            pad_id=token_type_pad_id,
            vocab_size=token_type_vocab_size,
            dim_embeddings=embedding_size,
        )
        self.token_type_embedder.set_trainable(trainable=True)

        if use_position_embeddings:
            self.pos_embedder = nn.Parameter(
                torch.Tensor(512, embedding_size),
                requires_grad=True,
            )

        self.dropout = nn.Dropout(dropout)

        xavier_param_init(self)

        # Layer normalization: :cite:`https://arxiv.org/abs/1607.06450`
        # eps from huggingface transformers implementation
        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-12)

    def forward(self, input_ids, token_type_ids) -> torch.FloatTensor:
        """Build the embeddings."""
        embeddings = self.word_embedder(input_ids) + self.token_type_embedder(
            token_type_ids
        )

        _, s_len, _ = embeddings.size()

        # [0, 1, 2, ..., s_len-1]
        length_indices = torch.tensor(
            list(range(s_len)), dtype=torch.long, device=input_ids.device
        )

        # size: (1, s_len, hidden_size)
        position_embeddings = torch.index_select(
            self.pos_embedder, 0, length_indices
        ).unsqueeze(0)

        return self.dropout(self.layer_norm(embeddings + position_embeddings))


@dataclass
class AttentionConfig:
    """Configurations for the MultiHeadAttention Module.

    Notes:
           num_heads (int): the number of attention heads
           dim_model (int): the dimension (hidden units) of the model
           dim_query (int): the dimension of the "query" vectors
           dim_key (int): the dimension of the "key" vectors
           dim_value (int): the dimension of the "value" vectors
    """

    num_heads: int
    dim_model: int
    dim_query: int
    dim_key: int
    dim_value: int
    mask_future: Optional[bool] = False
    cross_attention: Optional[bool] = False


@dataclass
class AttentionData:
    """Data for the MultiHeadAttention Module.

    Dimension Notes: query (FloatTensor): ``(batch_size,
    sequence_length, hidden_units)`` key (FloatTensor): ``(batch_size,
    sequence_length, hidden_units)`` value (FloatTensor): ``(batch_size,
    sequence_length, hidden_units)`` mask (ByteTensor): ``(batch_size,
    sequence_length)``
    """

    query: torch.FloatTensor
    key: torch.FloatTensor
    value: torch.FloatTensor
    mask: Optional[torch.ByteTensor] = None


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    :cite:`https://arxiv.org/abs/1706.03762`
    """

    def __init__(self, config: AttentionConfig, dropout: Optional[float] = 0.0) -> None:
        """Create linear layers needed for the multi-head attention module."""

        super(MultiHeadAttention, self).__init__()

        # Matrices for linear projection of "query", "key", "value"
        self.w_qs = nn.Linear(config.dim_model, config.num_heads * config.dim_query)
        self.w_ks = nn.Linear(config.dim_model, config.num_heads * config.dim_key)
        self.w_vs = nn.Linear(config.dim_model, config.num_heads * config.dim_value)

        # For getting a single vector from multiple head of attention values
        self.dense_layer = nn.Linear(
            config.num_heads * config.dim_value, config.dim_model
        )
        self.dropout = nn.Dropout(dropout)

        xavier_param_init(self)

        # Layer normalization: :cite:`https://arxiv.org/abs/1607.06450`
        self.layer_norm = nn.LayerNorm(config.dim_model, eps=1e-12)

        self.config = config

    def scaled_dot_product_attention(
        self,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
        mask: Optional[torch.ByteTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Scaled Dot-Product Attention.

        mask_future: used to mask future steps in the decoder.
        """
        attn = query.bmm(key.transpose(1, 2))

        # denominator for normalizing attention scores
        attn = attn / np.power(self.config.dim_key, 0.5)
        attn = attn.softmax(dim=2)
        if mask is not None:
            attn = attn * (1.0 - mask)

        if self.config.mask_future:
            _, q_length, kv_length = attn.size()
            # we assume q_length = kv_length in self-attention over decoder inputs.
            """ Example:
            if q_length = kv_length = 3:
            future_bias = [[0, 1.0, 1.0],
                           [0,  0,  1.0],
                           [0,  0,    0]]
            """
            future_mask = np.ones((q_length, kv_length))
            future_mask = torch.tensor(
                np.triu(future_mask, k=1), dtype=torch.float, device=attn.device
            )

            future_mask = future_mask.expand_as(attn)
            attn = attn * (1.0 - future_mask)

        attn = self.dropout(attn)
        output = attn.bmm(value)

        return output, attn

    def forward(self, *args) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute the multi-head attention.
        Returns:
            (FloatTensor, FloatTensor):
                * output ``(batch_size, sequence_length, hidden_units)``
                * attn ``(num_heads * batch_size, sequence_length, sequence_length)``
        """
        attention_data = args[0]
        b_sz, q_len, _ = attention_data.query.size()
        _, k_len, _ = attention_data.key.size()
        _, v_len, _ = attention_data.value.size()

        _, mask_len = attention_data.mask.size()

        attn_mask = attention_data.mask.repeat(1, q_len).view(-1, q_len, mask_len)

        if not self.config.cross_attention:
            attn_mask = attn_mask.masked_fill_(
                attention_data.mask.unsqueeze(dim=2).to(torch.bool), 1
            )
        attn_mask = attn_mask.repeat(self.config.num_heads, 1, 1).float()

        multi_q = (
            self.w_qs(attention_data.query)
            .view(b_sz, q_len, self.config.num_heads, self.config.dim_query)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, q_len, self.config.dim_query)
        )
        multi_k = (
            self.w_ks(attention_data.key)
            .view(b_sz, k_len, self.config.num_heads, self.config.dim_key)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, k_len, self.config.dim_key)
        )
        multi_v = (
            self.w_vs(attention_data.value)
            .view(b_sz, v_len, self.config.num_heads, self.config.dim_value)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, v_len, self.config.dim_value)
        )

        output, attn = self.scaled_dot_product_attention(
            multi_q, multi_k, multi_v, mask=attn_mask
        )

        output = (
            output.view(self.config.num_heads, b_sz, q_len, self.config.dim_value)
            .permute(1, 2, 0, 3)
            .contiguous()
            .view(b_sz, q_len, -1)
        )

        # for residual connection added to the output of attention values
        output = self.layer_norm(
            self.dropout(self.dense_layer(output)) + attention_data.query
        )

        return output, attn


def gelu(x: torch.FloatTensor) -> torch.FloatTensor:
    """Implementation of the gelu activation function. For information: OpenAI
    GPT's gelu is slightly different (and gives slightly different results):

    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    cdf = (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )
    # return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x * cdf


class PositionwiseFeedForward(nn.Module):
    """Implements a two layer FF network with gelu activation and residual
    connections."""

    def __init__(
        self,
        dim_input: Optional[int] = 768,
        dim_ff: Optional[int] = 3072,
        dropout: Optional[float] = 0.0,
    ) -> None:
        """Construction of 2 linear layers + layer norm."""

        super(PositionwiseFeedForward, self).__init__()

        # layer weight matrices
        self.layer1 = nn.Linear(dim_input, dim_ff)
        self.layer2 = nn.Linear(dim_ff, dim_input)
        self.dropout = nn.Dropout(dropout)

        xavier_param_init(self)

        self.layer_norm = nn.LayerNorm(dim_input, eps=1e-12)

    def forward(self, *args: List[Any]) -> torch.FloatTensor:
        """
        Args:
            input_x (FloatTensor): ``(-1, -1, ... , -1, dim_input)``

        Returns:
            (FloatTensor):

            * output ``(-1, -1, ... , -1, dim_input)``
        """
        input_x = args[0]
        output = gelu(self.layer1(input_x))
        output = self.layer2(output)
        return self.layer_norm(self.dropout(output) + input_x)


class AttentionBlock(nn.Module):
    """Implements a full attention block in the transformers model."""

    def __init__(
        self,
        hidden_size: Optional[int] = 768,
        is_decoder: Optional[bool] = False,
        num_attention_heads: Optional[int] = 1,
        attention_probs_dropout_prob: Optional[float] = 0.0,
        intermediate_size: Optional[int] = 3072,
        hidden_dropout_prob: Optional[float] = 0.0,
    ) -> None:
        """Construction of the attention layer used in Albert Model.
          Args:

        hidden_size: (optional) int, size of hidden layer.

        is_decoder: boolean to select the type of decoder.
        num_attention_heads: int. Number of attention heads.
        attention_head_size: int. Size of attention head.
        attention_probs_dropout_prob: float. dropout probability for attention_layer
        intermediate_size: int. Size of intermediate hidden layer.
        hidden_dropout_prob: (optional) float. Dropout probability of the hidden
          layer.
          Returns:
            layer output
        """
        super(AttentionBlock, self).__init__()

        q_size = hidden_size // num_attention_heads
        self_attn_cfg = AttentionConfig(
            num_heads=num_attention_heads,
            dim_model=hidden_size,
            dim_query=q_size,
            dim_key=q_size,
            dim_value=q_size,
            mask_future=is_decoder,
            cross_attention=False,
        )
        self.self_attention = MultiHeadAttention(
            self_attn_cfg, dropout=attention_probs_dropout_prob
        )

        if is_decoder:
            cross_attn_cfg = AttentionConfig(
                num_heads=num_attention_heads,
                dim_model=hidden_size,
                dim_query=q_size,
                dim_key=q_size,
                dim_value=q_size,
                mask_future=False,
                cross_attention=True,
            )
            self.cross_attention = MultiHeadAttention(
                cross_attn_cfg, dropout=attention_probs_dropout_prob
            )

        self.ffd_layer = PositionwiseFeedForward(
            dim_input=hidden_size, dim_ff=intermediate_size, dropout=hidden_dropout_prob
        )

        self.is_decoder = is_decoder

    def forward(
        self,
        layer_input,
        attention_mask,
        encoder_input_mask=None,
        encoder_hidden_output=None,
    ) -> torch.FloatTensor:
        """
        layer_input: float Tensor of shape [batch_size, from_seq_length, from_width].
        attention_mask: float Tensor of shape [batch_size, from_seq_length].
        encoder_input_mask: input mask for the source sequence.
        encoder_hidden_output: output from the encoder over the source sequence.
        """
        output, _ = self.self_attention(
            AttentionData(
                query=layer_input,
                key=layer_input,
                value=layer_input,
                mask=attention_mask,
            )
        )
        if self.is_decoder:
            output, _ = self.cross_attention(
                AttentionData(
                    query=output,
                    key=encoder_hidden_output,
                    value=encoder_hidden_output,
                    mask=encoder_input_mask,
                )
            )

        return self.ffd_layer(output)


class TransformerModel(nn.Module):
    """Implements a complete encoder or decoder of a transformer model."""

    def __init__(
        self,
        hidden_size: Optional[int] = 768,
        is_decoder: Optional[bool] = False,
        num_hidden_layers: Optional[int] = 12,
        num_attention_heads: Optional[int] = 12,
        attention_probs_dropout_prob: Optional[float] = 0.1,
        intermediate_size: Optional[int] = 3072,
        hidden_dropout_prob: Optional[float] = 0.1,
    ) -> None:

        super(TransformerModel, self).__init__()
        # we use single block as we share weights.
        self.single_block = AttentionBlock(
            hidden_size=hidden_size,
            is_decoder=is_decoder,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        self.num_layers = num_hidden_layers

    def forward(
        self,
        layer_input,
        attention_mask,
        encoder_input_mask=None,
        encoder_hidden_output=None,
    ) -> torch.FloatTensor:
        """Apply the same weights multiple times."""
        for layer in range(self.num_layers):
            layer_input = self.single_block(
                layer_input, attention_mask, encoder_input_mask, encoder_hidden_output
            )

        return layer_input


class AlbertConfig(object):
    """Configuration for `AlbertModel`.

    The default settings match the configuration of model
    `albert_xxlarge`.
    """

    def __init__(
        self,
        vocab_size=30000,
        embedding_size=128,
        hidden_size=4096,
        num_hidden_layers=12,
        num_hidden_groups=1,
        num_attention_heads=64,
        intermediate_size=16384,
        inner_group_num=1,
        down_scale_factor=1,
        hidden_act="gelu",
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        source_max_position_embeddings=512,
        decoder_max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        go_symbol_id=2,
        is_decoder=False,
        word_pad_id=0,
        token_type_pad_id=0,
        use_position_embeddings=True,
        bos_token_id=2,
        eos_token_id=3,
    ):
        """Constructs AlbertConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `AlbertModel`.
          embedding_size: size of voc embeddings.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_hidden_groups: Number of group for the hidden layers, parameters in
            the same group are shared.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          inner_group_num: int, number of inner repetition of attention and ffn.
          down_scale_factor: float, the scale to apply
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `AlbertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
          is_decoder: will this model used as decoder or encoder. The default is False (encoder).
          go_symbol_id: the id for the go symbol token used as the first input to decoder transformer.
        """
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.inner_group_num = inner_group_num
        self.down_scale_factor = down_scale_factor
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_position_embeddings = use_position_embeddings
        self.source_max_position_embeddings = source_max_position_embeddings
        self.decoder_max_position_embeddings = decoder_max_position_embeddings
        self.token_type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.is_decoder = is_decoder
        self.go_symbol_id = go_symbol_id
        self.word_pad_id = word_pad_id
        self.token_type_pad_id = token_type_pad_id

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `AlbertConfig` from a Python dictionary of
        parameters."""
        config = AlbertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `AlbertConfig` from a json file of parameters."""
        with io.open(json_file, mode="r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class AlbertModel(nn.Module):
    """Complete Albert Model used as encoder or decoder."""

    def __init__(self, config):
        """Constructor for AlbertModel.

        config: `AlbertConfig` instance.
        """
        super(AlbertModel, self).__init__()
        config = copy.deepcopy(config)
        max_position = (
            config.decoder_max_position_embeddings
            if config.is_decoder
            else config.source_max_position_embeddings
        )
        self.embedding = AlbertEmbedding(
            vocab_size=config.vocab_size,
            word_pad_id=config.word_pad_id,
            token_type_pad_id=config.token_type_pad_id,
            embedding_size=config.embedding_size,
            token_type_vocab_size=config.token_type_vocab_size,
            use_position_embeddings=config.use_position_embeddings,
            max_position_embeddings=max_position,
            dropout=config.attention_probs_dropout_prob,
        )

        if config.embedding_size != config.hidden_size:
            self.embed_to_hidden = torch.nn.Linear(
                config.embedding_size, config.hidden_size, bias=True
            )

        self.main_block = TransformerModel(
            hidden_size=config.hidden_size,
            is_decoder=config.is_decoder,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )

        self.config = config

    def forward(
        self,
        input_ids,
        input_mask=None,
        token_type_ids=None,
        encoder_input_mask=None,
        encoder_hidden_output=None,
    ) -> torch.FloatTensor:
        """Create mask or type id if not given, also shift to right the decoder
        inputs."""
        batch_size, seq_length = input_ids.size()
        if input_mask is None:
            input_mask = torch.ones(
                (batch_size, seq_length),
                dtype=torch.long,
                device=input_ids.device,
            )

        if token_type_ids is None:
            token_type_ids = torch.ones(
                (batch_size, seq_length),
                dtype=torch.long,
                device=input_ids.device,
            )

        # For the decoder, shift right the input_ids, input_mask, and token_type_ids.
        if self.config.is_decoder:
            token_type_ids = torch.roll(token_type_ids, shifts=1, dims=1)
            token_type_ids[:, 0:1] = torch.zeros(
                (batch_size, 1), dtype=torch.long, device=input_ids.device
            )

            input_mask = torch.roll(input_mask, shifts=1, dims=1)
            input_mask[:, 0:1] = torch.ones(
                (batch_size, 1), dtype=torch.long, device=input_ids.device
            )

            input_ids = torch.roll(input_ids, shifts=1, dims=1)
            input_ids[:, 0:1] = (
                torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)
                * self.config.go_symbol_id
            )

        emb_output = self.embedding(input_ids, token_type_ids)

        if self.config.embedding_size != self.config.hidden_size:
            emb_output = self.embed_to_hidden(emb_output)

        return self.main_block(
            layer_input=emb_output,
            attention_mask=input_mask,
            encoder_hidden_output=encoder_hidden_output,
            encoder_input_mask=encoder_input_mask,
        )


class AlbertEncoderDecoder(nn.Module):
    """Complete Albert Model used as encoder and decoder."""

    def __init__(self, config):
        """Constructor for EncoderDecoder Model.

        config: `AlbertConfig` instance.
        """
        super(AlbertEncoderDecoder, self).__init__()
        config.is_decoder = False
        self.encoder = AlbertModel(config)

        config.is_decoder = True
        self.decoder = AlbertModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        target_ids,
        labels=None,
        input_mask=None,
        target_mask=None,
        token_type_ids=None,
        target_token_type_ids=None,
    ) -> torch.FloatTensor:
        """Overall computation in the encoder decoder model."""
        encoder_output = self.encoder(
            input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids
        )
        decoder_output = self.decoder(
            input_ids=target_ids,
            input_mask=target_mask,
            token_type_ids=target_token_type_ids,
            encoder_hidden_output=encoder_output,
            encoder_input_mask=input_mask,
        )
        ret = {}
        ret["hidden_outputs"] = decoder_output
        if labels is not None:
            ret["loss"] = self.cal_loss(labels, decoder_output)
        return ret

    def cal_loss(self, labels, decoder_output):
        logits = self.lm_head(decoder_output)
        logits = logits.permute(0, 2, 1)
        return self.loss_fn(logits, labels)


def list_parameters(model: nn.Module):
    parameters = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            parameters[name] = param

    return parameters


# Map the parameters of this model from the pretrained checkpoint.
param_mapper = {
    "encoder.embedding.pos_embedder": "embeddings.position_embeddings.weight",
    "encoder.embedding.word_embedder.token_embs.weight": "embeddings.word_embeddings.weight",
    "encoder.embedding.token_type_embedder.token_embs.weight": "embeddings.token_type_embeddings.weight",
    "encoder.embedding.layer_norm.weight": "embeddings.LayerNorm.weight",
    "encoder.embedding.layer_norm.bias": "embeddings.LayerNorm.bias",
    "encoder.embed_to_hidden.weight": "encoder.embedding_hidden_mapping_in.weight",
    "encoder.embed_to_hidden.bias": "encoder.embedding_hidden_mapping_in.bias",
    "encoder.main_block.single_block.self_attention.w_qs.weight": "encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight",
    "encoder.main_block.single_block.self_attention.w_qs.bias": "encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias",
    "encoder.main_block.single_block.self_attention.w_ks.weight": "encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight",
    "encoder.main_block.single_block.self_attention.w_ks.bias": "encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias",
    "encoder.main_block.single_block.self_attention.w_vs.weight": "encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight",
    "encoder.main_block.single_block.self_attention.w_vs.bias": "encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias",
    "encoder.main_block.single_block.self_attention.dense_layer.weight": "encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight",
    "encoder.main_block.single_block.self_attention.dense_layer.bias": "encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias",
    "encoder.main_block.single_block.self_attention.layer_norm.weight": "encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight",
    "encoder.main_block.single_block.self_attention.layer_norm.bias": "encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.bias",
    "encoder.main_block.single_block.ffd_layer.layer1.weight": "encoder.albert_layer_groups.0.albert_layers.0.ffn.weight",
    "encoder.main_block.single_block.ffd_layer.layer1.bias": "encoder.albert_layer_groups.0.albert_layers.0.ffn.bias",
    "encoder.main_block.single_block.ffd_layer.layer2.weight": "encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight",
    "encoder.main_block.single_block.ffd_layer.layer2.bias": "encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias",
    "encoder.main_block.single_block.ffd_layer.layer_norm.weight": "encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight",
    "encoder.main_block.single_block.ffd_layer.layer_norm.bias": "encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.bias",
    "decoder.embedding.pos_embedder": "embeddings.position_embeddings.weight",
    "decoder.embedding.word_embedder.token_embs.weight": "embeddings.word_embeddings.weight",
    "decoder.embedding.token_type_embedder.token_embs.weight": "embeddings.token_type_embeddings.weight",
    "decoder.embedding.layer_norm.weight": "embeddings.LayerNorm.weight",
    "decoder.embedding.layer_norm.bias": "embeddings.LayerNorm.bias",
    "decoder.embed_to_hidden.weight": "encoder.embedding_hidden_mapping_in.weight",
    "decoder.embed_to_hidden.bias": "encoder.embedding_hidden_mapping_in.bias",
    "decoder.main_block.single_block.self_attention.w_qs.weight": "encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight",
    "decoder.main_block.single_block.self_attention.w_qs.bias": "encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias",
    "decoder.main_block.single_block.self_attention.w_ks.weight": "encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight",
    "decoder.main_block.single_block.self_attention.w_ks.bias": "encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias",
    "decoder.main_block.single_block.self_attention.w_vs.weight": "encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight",
    "decoder.main_block.single_block.self_attention.w_vs.bias": "encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias",
    "decoder.main_block.single_block.self_attention.dense_layer.weight": "encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight",
    "decoder.main_block.single_block.self_attention.dense_layer.bias": "encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias",
    "decoder.main_block.single_block.self_attention.layer_norm.weight": "encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight",
    "decoder.main_block.single_block.self_attention.layer_norm.bias": "encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.bias",
    "decoder.main_block.single_block.ffd_layer.layer1.weight": "encoder.albert_layer_groups.0.albert_layers.0.ffn.weight",
    "decoder.main_block.single_block.ffd_layer.layer1.bias": "encoder.albert_layer_groups.0.albert_layers.0.ffn.bias",
    "decoder.main_block.single_block.ffd_layer.layer2.weight": "encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight",
    "decoder.main_block.single_block.ffd_layer.layer2.bias": "encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias",
    "decoder.main_block.single_block.ffd_layer.layer_norm.weight": "encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight",
    "decoder.main_block.single_block.ffd_layer.layer_norm.bias": "encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.bias",
}


def load_albert_encoder_decoder(mask_token_id, source_max_length, decoder_max_length):
    """Load the pretrained model into a encoder-decoder model."""
    config = AlbertConfig(
        num_hidden_layers=6,
        go_symbol_id=mask_token_id,
        source_max_position_embeddings=source_max_length,
        decoder_max_position_embeddings=decoder_max_length,
    )
    model = AlbertEncoderDecoder(config)

    pretrained_state_dict = torch.load(
        "./albert-xxlarge-v2-pytorch.model", map_location=lambda storage, loc: storage
    )["model_state_dict"]

    model_dict = model.state_dict()
    for key, _ in model_dict.items():
        if key in param_mapper:
            model_dict[key] = pretrained_state_dict[param_mapper[key]]

    model.load_state_dict(model_dict)

    return model


def load_pretrained_race(path, mask_token_id, source_max_length, decoder_max_length):
    """Load the pretrained model into a encoder-decoder model."""
    config = AlbertConfig(
        num_hidden_layers=6,
        go_symbol_id=mask_token_id,
        source_max_position_embeddings=source_max_length,
        decoder_max_position_embeddings=decoder_max_length,
    )
    model = AlbertEncoderDecoder(config)

    model_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_dict)

    return model
