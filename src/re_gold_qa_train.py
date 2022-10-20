import argparse

from src.question_response_generation.t5_model import T5QA
from src.question_response_generation.train import run_model
from src.re_qa_model import REQA, HyperParameters, load_module, set_random_seed
from src.re_qa_train import iterative_run_model
from src.zero_extraction_utils import (create_fewrl_dataset,
    create_relation_qq_dataset, create_zero_re_qa_dataset,
    create_zero_re_qa_gold_dataset)


def run_relation_classification_qa(args):
    """Run the relation-extraction qa models using the given gold questions for
    the head entity and the relation."""
    config = HyperParameters(
        model_path=args.model_path,
        batch_size=args.batch_size,
        source_max_length=256,
        decoder_max_length=32,
        gpu=args.gpu,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        mode="test",
        prediction_file=args.prediction_file,
        checkpoint=args.checkpoint,
        training_steps=args.training_steps,
        seed=args.seed,
        predict_type=args.predict_type,
    )

    set_random_seed(config.seed)

    if args.concat == "True":
        concat_bool = True
    else:
        concat_bool = False
    model = T5QA(config)
    (val_loaders, val_dataset,) = create_zero_re_qa_gold_dataset(
        question_tokenizer=model.tokenizer,
        answer_tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
        file=args.dev,
        concat=concat_bool,
    )

    run_model(
        model,
        config=config,
        train_dataloader=None,
        test_dataloader=val_loaders,
        save_always=True,
    )


def run_re_gold_qa(args):
    """Run the relation-extraction qa models using the given gold questions for
    the head entity and the relation."""
    if args.mode == "re_gold_qa_train":
        mode = "train"
        for_evaluation = False
    elif args.mode == "re_gold_qa_test":
        mode = "test"
        for_evaluation = True
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
        training_steps=args.training_steps,
        seed=args.seed,
    )

    set_random_seed(config.seed)

    model = T5QA(config)

    if not for_evaluation:
        load_module(model.model.module, model.model_path, args.checkpoint)

    (
        train_loaders,
        val_loaders,
        train_dataset,
        val_dataset,
    ) = create_zero_re_qa_dataset(
        question_tokenizer=model.tokenizer,
        answer_tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
        train_file=args.train,
        dev_file=args.dev,
        ignore_unknowns=False,
        concat=False,
        gold_questions=True,
        for_evaluation=for_evaluation,
    )

    run_model(
        model,
        config=config,
        train_dataloader=train_loaders,
        test_dataloader=val_loaders,
        save_always=True,
    )


def run_re_concat_qa(args):
    """Run the relation-extraction qa models using the concat of head entity
    and the relation word."""
    if args.mode == "re_concat_qa_train":
        mode = "train"
        for_evaluation = False
    elif args.mode == "re_concat_qa_test":
        mode = "test"
        for_evaluation = True
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
        training_steps=args.training_steps,
        seed=args.seed,
    )

    set_random_seed(config.seed)
    model = T5QA(config)
    if not for_evaluation:
        load_module(model.model.module, model.model_path, args.checkpoint)

    (
        train_loaders,
        val_loaders,
        train_dataset,
        val_dataset,
    ) = create_zero_re_qa_dataset(
        question_tokenizer=model.tokenizer,
        answer_tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
        train_file=args.train,
        dev_file=args.dev,
        ignore_unknowns=False,
        concat=True,
        gold_questions=False,
        for_evaluation=for_evaluation,
    )

    run_model(
        model,
        config=config,
        train_dataloader=train_loaders,
        test_dataloader=val_loaders,
        save_always=True,
    )


def run_re_qa(args):
    """Run the relation-extraction qa models using the question generator and
    the response generator explored with some search algorithm."""
    if args.mode == "re_qa_train":
        mode = "train"
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
            training_steps=int(args.training_steps),
            answer_checkpoint=args.answer_checkpoint,
            question_checkpoint=args.question_checkpoint,
            num_search_samples=int(args.num_search_samples),
            seed=args.seed,
        )
        set_random_seed(config.seed)
        model = REQA(config)
        model = model.to("cuda:0")

        (
            train_loaders,
            val_loaders,
            train_dataset,
            val_dataset,
        ) = create_zero_re_qa_dataset(
            question_tokenizer=model.question_tokenizer,
            answer_tokenizer=model.answer_tokenizer,
            batch_size=config.batch_size,
            source_max_length=config.source_max_length,
            decoder_max_length=config.decoder_max_length,
            train_file=args.train,
            dev_file=args.dev,
            ignore_unknowns=False,
            concat=False,
            gold_questions=False,
        )
        iterative_run_model(
            model,
            config=config,
            train_dataloader=train_loaders,
            test_dataloader=val_loaders,
            save_always=True,
            current_device=0,
            train_method=args.train_method,
        )

    if args.mode == "re_qa_test":
        config = HyperParameters(
            model_path=args.model_path,
            batch_size=args.batch_size,
            source_max_length=256,
            decoder_max_length=32,
            gpu=args.gpu,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            mode="test",
            prediction_file=args.prediction_file,
            answer_checkpoint=args.answer_checkpoint,
            question_checkpoint=args.question_checkpoint,
            seed=args.seed,
        )
        set_random_seed(config.seed)
        model = REQA(config)
        model = model.to("cuda:0")

        (_, val_loaders, _, val_dataset) = create_zero_re_qa_dataset(
            question_tokenizer=model.question_tokenizer,
            answer_tokenizer=model.answer_tokenizer,
            batch_size=config.batch_size,
            source_max_length=config.source_max_length,
            decoder_max_length=config.decoder_max_length,
            dev_file=args.dev,
            ignore_unknowns=False,
            concat=False,
            gold_questions=False,
            for_evaluation=True,
        )

        iterative_run_model(
            model,
            config=config,
            test_dataloader=val_loaders,
            current_device=0,
        )


def run_fewrl(args):
    """Run the relation-extraction qa models using the question generator and
    the response generator explored with some search algorithm on the fewrel dataset."""
    if args.mode == "fewrl_train":
        mode = "train"
        for_fewrl = True
    elif args.mode == "fewrl_dev":
        mode = "test"
        for_fewrl = True
    elif args.mode == "multi_fewrl_dev":
        mode = "test"
        for_fewrl = True
    elif args.mode == "fewrl_test":
        mode = "test"
        for_fewrl = True
    elif args.mode == 'reqa_mml_eval':
        mode = "test"
        for_fewrl = False

    if args.mode == "multi_fewrl_dev":
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
            training_steps=int(args.training_steps),
            answer_checkpoint=args.answer_checkpoint, # will be ignored.
            question_checkpoint=args.question_checkpoint, # will be ignored.
            num_search_samples=int(args.num_search_samples),
            seed=args.seed,
            predict_type=args.predict_type,
        )
        set_random_seed(config.seed)
        model = REQA(config)
        model = model.to("cuda:0")

        (loader, dataset) = create_relation_qq_dataset(
                question_tokenizer=model.question_tokenizer,
                answer_tokenizer=model.answer_tokenizer,
                batch_size=config.batch_size,
                source_max_length=config.source_max_length,
                decoder_max_length=config.decoder_max_length,
                train_fewrel_path=args.dev,
                shuffle=False,
                for_fewrel_dataset=for_fewrl
        )
        for ep in range(args.start_epoch, args.end_epoch+1, 1):
            for step in range(args.start_step, args.end_step + args.step_up, args.step_up):
                prediction_file = args.model_path + "relation.supp_data.offmml-pgg.run.epoch.{}.dev.predictions.step.{}.csv".format(ep, step)
                answer_checkpoint="_{}_answer_step_{}".format(ep, step)
                question_checkpoint="_{}_question_step_{}".format(ep, step)
                config.prediction_file = prediction_file
                load_module(model.answer_model, model.model_path, answer_checkpoint)
                load_module(model.question_model, model.model_path, question_checkpoint)
                iterative_run_model(
                    model,
                    config=config,
                    test_dataloader=loader,
                    current_device=0,
                )
    else:
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
            training_steps=int(args.training_steps),
            answer_checkpoint=args.answer_checkpoint,
            question_checkpoint=args.question_checkpoint,
            num_search_samples=int(args.num_search_samples),
            seed=args.seed,
            predict_type=args.predict_type,
        )
        set_random_seed(config.seed)
        model = REQA(config)
        model = model.to("cuda:0")

        if args.mode == "fewrl_train":
            (loader, dataset) = create_relation_qq_dataset(
                question_tokenizer=model.question_tokenizer,
                answer_tokenizer=model.answer_tokenizer,
                batch_size=config.batch_size,
                source_max_length=config.source_max_length,
                decoder_max_length=config.decoder_max_length,
                train_fewrel_path=args.train,
                shuffle=True,
                for_fewrel_dataset=for_fewrl
            )

            iterative_run_model(
                model,
                config=config,
                train_dataloader=loader,
                test_dataloader=None,
                save_always=True,
                current_device=0,
                train_method=args.train_method,
            )

        if args.mode == "fewrl_dev":
            (loader, dataset) = create_relation_qq_dataset(
                question_tokenizer=model.question_tokenizer,
                answer_tokenizer=model.answer_tokenizer,
                batch_size=config.batch_size,
                source_max_length=config.source_max_length,
                decoder_max_length=config.decoder_max_length,
                train_fewrel_path=args.dev,
                shuffle=False,
                for_fewrel_dataset=for_fewrl
            )
            iterative_run_model(
                model,
                config=config,
                test_dataloader=loader,
                current_device=0,
            )

        if args.mode in ["fewrl_test", "reqa_mml_eval"]:
            (loader, dataset) = create_relation_qq_dataset(
                question_tokenizer=model.question_tokenizer,
                answer_tokenizer=model.answer_tokenizer,
                batch_size=config.batch_size,
                source_max_length=config.source_max_length,
                decoder_max_length=config.decoder_max_length,
                train_fewrel_path=args.test,
                shuffle=False,
                for_fewrel_dataset=for_fewrl
            )
            iterative_run_model(
                model,
                config=config,
                test_dataloader=loader,
                current_device=0,
            )

def run_multi_concat_fewrl_dev(args):
    """Run concat model on the fewrl dataset for multiple checkpoints."""
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
        seed=args.seed,
        predict_type=args.predict_type,
        model_name="t5-small",
        checkpoint=args.checkpoint, # to avoid error otherwise it is not being used in this multi eval.
    )
    set_random_seed(config.seed)
    model = T5QA(config)

    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_fewrl_dataset(
        question_tokenizer=model.tokenizer,
        answer_tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
        train_fewrel_path=args.train,
        dev_fewrel_path=args.dev,
        test_fewrel_path=args.test,
        concat=True
    )

    for ep in range(args.start_epoch, args.end_epoch+1, 1):
        for step in range(args.start_step, args.end_step + args.step_up, args.step_up):
            prediction_file = args.model_path + "relation.concat.run.epoch.{}.dev.predictions.step.{}.csv".format(ep, step)
            checkpoint="_{}_step_{}_model".format(ep, step)
            config.checkpoint = checkpoint
            config.prediction_file = prediction_file
            load_module(model.model, model.model_path, config.checkpoint)
            run_model(
                model,
                config=config,
                train_dataloader=train_loader,
                test_dataloader=val_loader,
                save_always=True,
            )


def run_concat_fewrl(args):
    """Run concat model on the fewrl dataset."""
    if args.mode == "concat_fewrl_train":
        mode = "train"
    elif args.mode == "concat_fewrl_dev":
        mode = "test"
    elif args.mode == "concat_fewrl_test":
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
        seed=args.seed,
        predict_type=args.predict_type,
        model_name="t5-small"
    )

    set_random_seed(config.seed)
    model = T5QA(config)
    if mode != "test":
        load_module(model.model.module, model.model_path, args.checkpoint)

    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_fewrl_dataset(
        question_tokenizer=model.tokenizer,
        answer_tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
        train_fewrel_path=args.train,
        dev_fewrel_path=args.dev,
        test_fewrel_path=args.test,
        concat=True
    )

    if args.mode == "concat_fewrl_train":
        run_model(
            model,
            config=config,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            save_always=True,
        )

    if args.mode == "concat_fewrl_dev":
        run_model(
            model,
            config=config,
            train_dataloader=train_loader,
            test_dataloader=val_loader,
            save_always=True,
        )

    if args.mode == "concat_fewrl_test":
        run_model(
            model,
            config=config,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            save_always=True,
        )


def run_main(args):
    """Decides what to do in the code."""
    if args.mode in ["re_gold_qa_train", "re_gold_qa_test"]:
        run_re_gold_qa(args)
    if args.mode in ["re_classification_qa_train"]:
        run_relation_classification_qa(args)
    if args.mode in ["re_concat_qa_train", "re_concat_qa_test"]:
        run_re_concat_qa(args)
    if args.mode in ["re_qa_train", "re_qa_test"]:
        run_re_qa(args)
    if args.mode in ["multi_fewrl_dev", "reqa_mml_eval", "fewrl_train", "fewrl_test", "fewrl_dev"]:
        run_fewrl(args)
    if args.mode in ["concat_fewrl_train", "concat_fewrl_test", "concat_fewrl_dev"]:
        run_concat_fewrl(args)
    if args.mode in ["multi_concat_fewrl_dev"]:
        run_multi_concat_fewrl_dev(args)


def argument_parser():
    """augments arguments for model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="re_gold_qa_train | re_gold_qa_test | re_concat_qa_train | re_concat_qa_test | re_qa_train | re_qa_test",
    )
    parser.add_argument(
        "--concat",
        type=str,
    )
    parser.add_argument(
        "--train_method",
        type=str,
        help="MML-PGG-Off-Sim",
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

    parser.add_argument("--start_epoch", type=int, default=0)

    parser.add_argument("--end_epoch", type=int, default=0)

    parser.add_argument("--start_step", type=int, default=0)

    parser.add_argument("--end_step", type=int, default=0)

    parser.add_argument("--step_up", type=int, default=0)

    parser.add_argument(
        "--num_unseen_relations",
        type=int,
        default=5,
        help="number of the unseen relations on the test or dev sets.",
    )

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
    parser.add_argument(
        "--num_search_samples", type=int, default=8, help="Number of search samples"
    )
    parser.add_argument(
        "--num_neg_samples",
        type=int,
        default=8,
        help="Number of the negative samples for nce loss.",
    )
    parser.add_argument(
        "--answer_checkpoint", type=str, help="checkpoint of the trained answer model."
    )
    parser.add_argument(
        "--question_checkpoint",
        type=str,
        help="checkpoint of the trained question model.",
    )
    parser.add_argument(
        "--training_steps",
        type=int,
        default=0,
        help="number of training steps over the train data.",
    )
    parser.add_argument(
        "--predict_type",
        type=str,
        help="What is the prediction type for the fewrel run.",
    )
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    run_main(args)
