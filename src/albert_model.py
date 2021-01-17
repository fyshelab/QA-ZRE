"""Implementation of the albert Module for Response Generation.

Some parts are from:
https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer
https://github.com/google-research/albert
"""
import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy
import numpy as np
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
        initializer_range: Optional[float] = 0.02,
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
                torch.Tensor(max_position_embeddings, embedding_size),
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
        length_indices = torch.LongTensor(list(range(s_len)))

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
        self.layer_norm = nn.LayerNorm(config.dim_model)

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
            future_mask = torch.FloatTensor(np.triu(future_mask, k=1))
            future_mask.to(attn.device)

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
        b_sz, s_len, _ = attention_data.query.size()

        attn_mask = attention_data.mask.repeat(1, s_len).view(-1, s_len, s_len)
        attn_mask = attn_mask.masked_fill_(
            attention_data.mask.unsqueeze(dim=2).to(torch.bool), 1
        )
        attn_mask = attn_mask.repeat(self.config.num_heads, 1, 1).float()

        multi_q = (
            self.w_qs(attention_data.query)
            .view(b_sz, s_len, self.config.num_heads, self.config.dim_query)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, s_len, self.config.dim_query)
        )
        multi_k = (
            self.w_ks(attention_data.key)
            .view(b_sz, s_len, self.config.num_heads, self.config.dim_key)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, s_len, self.config.dim_key)
        )
        multi_v = (
            self.w_vs(attention_data.value)
            .view(b_sz, s_len, self.config.num_heads, self.config.dim_value)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, s_len, self.config.dim_value)
        )

        output, attn = self.scaled_dot_product_attention(
            multi_q, multi_k, multi_v, mask=attn_mask
        )

        output = (
            output.view(self.config.num_heads, b_sz, s_len, self.config.dim_value)
            .permute(1, 2, 0, 3)
            .contiguous()
            .view(b_sz, s_len, -1)
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

        self.layer_norm = nn.LayerNorm(dim_input)

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
