import torch

from src.re_qa_model import REQA, HyperParameters, load_module


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
