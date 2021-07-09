import argparse

from src.re_qa_model import HyperParameters
from src.t5_model import T5QA
from src.train import run_model
from src.zero_extraction_utils import create_zero_re_gold_qa_dataset


def run_re_gold_qa(args):
    """Run the relation-extraction qa models using the gold question."""
    if args.mode == "re_gold_qa_train":
        mode = "train"
    elif args.mode == "re_gold_qa_test":
        mode = "test"
    config = HyperParameters(
        model_path=args.model_path,
        batch_size=args.batch_size,
        source_max_length=256,
        decoder_max_length=32,
        gpu=args.gpu,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        mode=mode,
        prediction_file=args.prediction_file,
        checkpoint=args.checkpoint,
    )
    model = T5QA(config)

    (
        train_loaders,
        val_loaders,
        train_dataset,
        val_dataset,
    ) = create_zero_re_gold_qa_dataset(
        tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
        train_file=args.train,
        dev_file=args.dev,
    )

    run_model(
        model,
        config=config,
        train_dataloader=train_loaders,
        dev_dataloader=val_loaders,
        test_dataloader=None,
        save_always=True,
    )


def run_main(args):
    """Decides what to do in the code."""
    if args.mode in ["re_gold_qa_train", "re_gold_qa_test"]:
        run_re_gold_qa(args)


def argument_parser():
    """augments arguments for model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="re_gold_qa_train | re_gold_qa_test",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path for saving or loading models.",
    )

    # Train specific
    parser.add_argument("--train", type=str, help="file for train data.")

    parser.add_argument("--dev", type=str, help="file for validation data.")

    # Test specific
    parser.add_argument("--test", type=str, help="file for test data.")

    parser.add_argument(
        "--prediction_file", type=str, help="file for saving predictions"
    )

    parser.add_argument("--input_file_name", type=str, help="input file name")

    parser.add_argument("--learning_rate", type=float, default=0.0005)

    parser.add_argument("--batch_size", type=int, default=8, help="static batch size")

    parser.add_argument(
        "--max_epochs", type=int, default=25, help="max number of training iterations"
    )

    parser.add_argument("--seed", type=int, default=len("dream"), help="random seed")

    parser.add_argument(
        "--config_file", type=str, default="config.ini", help="config.ini file"
    )

    # GPU or CPU
    parser.add_argument(
        "--gpu", type=bool, default=True, help="on gpu or not? True or False"
    )

    parser.add_argument(
        "--checkpoint", type=str, help="checkpoint of the trained model."
    )

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    run_main(args)
