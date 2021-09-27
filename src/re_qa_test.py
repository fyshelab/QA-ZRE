import torch

from src.re_qa_model import (REQA, HyperParameters, load_module,
                             prepare_response_module_input)


def test_prepare_response_module_input():
    """Test that if the input processing for the response module is correct."""

    # -100 is the pad token.
    labels = torch.tensor([[0, 1, 2, 3, -100, -100, -100], [0, 5, 6, 7, 8, -100, -100]])
    target_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]])
    sample_mask = torch.tensor([[1, 0], [1, 1]])
    input_ids = torch.tensor(
        [[0, 1, 2, 4, 4], [0, 1, 3, 4, 4], [0, 5, 6, 7, 4], [0, 5, 6, 6, 4]]
    )
    input_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]]
    )
    (
        answer_input_ids,
        answer_input_mask,
        target_mask,
        new_labels,
    ) = prepare_response_module_input(
        labels=labels,
        num_samples=2,
        target_mask=target_mask,
        sample_masks=sample_mask,
        answer_input_ids=input_ids,
        answer_input_mask=input_mask,
    )

    assert answer_input_ids.tolist() == [
        [0, 1, 2, 4, 4],
        [0, 0, 0, 0, 0],
        [0, 5, 6, 7, 4],
        [0, 5, 6, 6, 4],
    ]

    assert answer_input_mask.tolist() == [
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
    ]

    assert target_mask.tolist() == [
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
    ]

    # the second sample in the first row of the batch is dummy sample.
    assert new_labels.tolist() == [
        [0, 1, 2, 3, -100, -100, -100],
        [-100, -100, -100, -100, -100, -100, -100],
        [0, 5, 6, 7, 8, -100, -100],
        [0, 5, 6, 7, 8, -100, -100],
    ]
    assert new_labels.size() == (4, 7)


def test_model_prediction():
    """Test the main constructor of the model."""
    config = HyperParameters(batch_size=2, model_path="./", mode="train", num_beams=16)

    model = REQA(config, load_answer=False)

    batch = {
        "passages": ["This is Saeed", "This is a test"],
        "entity_relation_passage_input_ids": torch.tensor(
            [[0, 1, 2, 3, 4, 1, 0], [0, 5, 6, 7, 8, 1, 0]]
        ),
        "entity_relation_passage_attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0]]
        ),
        "second_entity_attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "second_entity_labels": torch.tensor([[10, 11, 1, 0], [12, 1, 0, 0]]),
        "input_ids": torch.tensor([[0, 1, 2, 3, 4, 1, 0], [0, 5, 6, 7, 8, 1, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0]]),
        "target_attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "labels": torch.tensor([[10, 11, 1, 0], [12, 1, 0, 0]]),
        "question_input_ids": torch.tensor(
            [[0, 1, 2, 3, 4, 1, 0], [0, 5, 6, 7, 8, 1, 0]]
        ),
        "question_attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0]]
        ),
        "question_target_attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "question_labels": torch.tensor([[10, 11, 1, 0], [12, 1, 0, 0]]),
    }
    output = list(model.predict(batch))

    answer_output = model.train(batch, phase="answer")
    assert round(answer_output["loss_value"], 4) == 13.3691

    question_output = model.train(batch, phase="question")
    assert round(question_output["loss_value"], 4) == 33.8681
