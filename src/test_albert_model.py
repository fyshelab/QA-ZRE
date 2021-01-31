"""Tests for the albert model."""
import torch

from src.albert_model import (AlbertConfig, AlbertEmbedding,
                              AlbertEncoderDecoder, AlbertModel,
                              AlbertTokenEmbedding, AttentionBlock,
                              AttentionConfig, AttentionData,
                              MultiHeadAttention, PositionwiseFeedForward,
                              TransformerModel, list_parameters,
                              set_random_seed)

set_random_seed(len("dreamscape-qa"))


def test_embeddings():
    """Testing the TokenEmbedding class."""
    embedder = AlbertTokenEmbedding(pad_id=3, vocab_size=4, dim_embeddings=2)
    tokens = torch.LongTensor([[0, 1, 2, 3]])

    assert embedder.token_embs.weight.size() == (4, 2)
    assert embedder.token_embs.weight.tolist()[3] == [0.0, 0.0]

    output_vectors = embedder(tokens)
    assert output_vectors.size() == (1, 4, 2)

    # pads must get zero vector.
    assert output_vectors.tolist()[0][3] == [0.0, 0.0]

    albert_emb = AlbertEmbedding(
        vocab_size=4,
        word_pad_id=3,
        token_type_pad_id=9,
        embedding_size=2,
        token_type_vocab_size=10,
    )

    token_types = torch.LongTensor([[4, 5, 9, 9]])
    final_embeddings = albert_emb(input_ids=tokens, token_type_ids=token_types)

    assert final_embeddings.size() == (1, 4, 2)


def test_scaled_dot_attention():
    """Testing if the scaled dot attention works."""
    attn_cfg = AttentionConfig(
        num_heads=1, dim_model=2, dim_query=2, dim_key=2, dim_value=2
    )
    multi_head_atten = MultiHeadAttention(attn_cfg)
    matrix = [
        [[0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]],
    ]
    matrix = torch.FloatTensor(matrix)
    _, attn = multi_head_atten.scaled_dot_product_attention(matrix, matrix, matrix)
    assert attn.size() == (3, 4, 4)

    # test if future decoding works.
    attn_cfg = AttentionConfig(
        num_heads=1, dim_model=2, dim_query=2, dim_key=2, dim_value=2, mask_future=True
    )

    decoder_multi_head_atten = MultiHeadAttention(attn_cfg)
    _, decoder_attn = decoder_multi_head_atten.scaled_dot_product_attention(
        matrix, matrix, matrix
    )
    assert decoder_attn.tolist()[0][0][1:] == [0.0, 0.0, 0.0]
    assert decoder_attn.tolist()[0][1][2:] == [0.0, 0.0]
    assert decoder_attn.tolist()[0][1][3:] == [0.0]


def test_multi_head_attention():
    """Testing if the multi-head attention works."""
    attn_cfg = AttentionConfig(
        num_heads=2, dim_model=2, dim_query=2, dim_key=2, dim_value=2, mask_future=True
    )
    multi_head_atten = MultiHeadAttention(attn_cfg)
    matrix = [
        [[0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]],
    ]
    matrix = torch.FloatTensor(matrix)

    mask = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]]
    mask = torch.ByteTensor(mask)
    attn_data = AttentionData(query=matrix, key=matrix, value=matrix, mask=mask)
    _, attn = multi_head_atten(attn_data)

    """
    expected_attn_row = [
        [0.2406, 0.0000, 0.0000, 0.0000],
        [0.2406, 0.2406, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000],
    ]
    """

    assert attn.tolist()[0][0][1:4] == [0.0000, 0.0000, 0.0000]
    assert attn.tolist()[0][1][2:4] == [0.0000, 0.0000]
    assert attn.tolist()[0][2][3:4] == [0.0000]
    assert attn.tolist()[0][3] == [0.0000, 0.0000, 0.0000, 0.0000]
    assert attn.size() == (2 * 3, 4, 4)


def test_position_network():
    """Testing positionwise feedforward class."""
    model = PositionwiseFeedForward(2, 4, dropout=0.0)
    input_x = [
        [[0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]],
    ]
    input_x = torch.FloatTensor(input_x)
    output = model(input_x)
    output = output.tolist()

    # same vector in different steps should have a same output.
    assert output[0][0] == output[0][1]
    assert output[1][1] == output[1][2]


def test_attention_block():
    """Testing the attention block."""
    attn_block = AttentionBlock(
        hidden_size=2,
        is_decoder=False,
        num_attention_heads=1,
        attention_probs_dropout_prob=0.1,
        intermediate_size=4,
        hidden_dropout_prob=0.2,
    )

    matrix = [
        [[0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]],
    ]
    matrix = torch.FloatTensor(matrix)

    mask = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]]
    mask = torch.ByteTensor(mask)

    output = attn_block(layer_input=matrix, attention_mask=mask)
    assert output.size() == (3, 4, 2)
    assert output.tolist()[0][3] == [0, 0]

    attn_block = AttentionBlock(
        hidden_size=2,
        is_decoder=True,
        num_attention_heads=1,
        attention_probs_dropout_prob=0.1,
        intermediate_size=4,
        hidden_dropout_prob=0.2,
    )

    output = attn_block(
        layer_input=matrix,
        attention_mask=mask,
        encoder_hidden_output=matrix,
        encoder_input_mask=mask,
    )

    assert output.size() == (3, 4, 2)


def test_transformer_models():
    """Test the full block."""
    model = TransformerModel(
        hidden_size=2,
        is_decoder=True,
        num_attention_heads=1,
        attention_probs_dropout_prob=0.1,
        intermediate_size=4,
        hidden_dropout_prob=0.2,
        num_hidden_layers=12,
    )
    matrix = [
        [[0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]],
    ]
    matrix = torch.FloatTensor(matrix)
    mask = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]]
    mask = torch.ByteTensor(mask)
    output = model(
        layer_input=matrix,
        attention_mask=mask,
        encoder_hidden_output=matrix,
        encoder_input_mask=mask,
    )
    assert output.size() == (3, 4, 2)


def test_full_model():
    """Test the full albert model."""
    config = AlbertConfig(
        is_decoder=False,
        vocab_size=100,
        go_symbol_id=1,
        embedding_size=16,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
    )
    model = AlbertModel(config)

    input_ids = [[2, 3, 4, 5], [5, 4, 3, 2], [10, 11, 12, 13]]
    input_ids = torch.LongTensor(input_ids)

    mask = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]]
    mask = torch.ByteTensor(mask)

    output = model(input_ids=input_ids, input_mask=mask)
    assert output.size() == (3, 4, 32)

    decoder_config = AlbertConfig(
        is_decoder=True,
        vocab_size=100,
        go_symbol_id=1,
        embedding_size=16,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
    )

    decoder_model = AlbertModel(decoder_config)
    output = decoder_model(
        input_ids=input_ids,
        input_mask=mask,
        encoder_hidden_output=torch.ones((3, 4, 32), dtype=torch.float),
        encoder_input_mask=mask,
    )

    assert output.size() == (3, 4, 32)

    config = AlbertConfig(
        vocab_size=100,
        go_symbol_id=1,
        embedding_size=16,
        hidden_size=32,
        source_max_position_embeddings=64,
        decoder_max_position_embeddings=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
    )

    model = AlbertEncoderDecoder(config)
    output = model(
        input_ids=input_ids,
        input_mask=mask,
        target_mask=mask,
        target_ids=input_ids,
    )
    assert output["hidden_outputs"].size() == (3, 4, 32)
