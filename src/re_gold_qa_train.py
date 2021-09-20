import argparse
import os

import torch
import torch.distributed as dist
import torch.utils.data.distributed

from src.question_response_generation.t5_model import T5QA
from src.question_response_generation.train import run_model
from src.re_qa_model import REQA, HyperParameters, set_random_seed
from src.re_qa_train import iterative_run_model
from src.zero_extraction_utils import (create_question_generation_dataset,
                                       create_zero_re_qa_dataset)


def run_re_gold_qa(args):
    """Run the relation-extraction qa models using the gold questions for the
    head entity and the relation."""
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
        answer_training_steps=args.answer_training_steps,
        seed=args.seed,
    )

    set_random_seed(config.seed)
    model = T5QA(config)

    (
        train_loaders,
        val_loaders,
        train_dataset,
        val_dataset,
        train_sample,
    ) = create_zero_re_qa_dataset(
        question_tokenizer=model.tokenizer,
        answer_tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
        train_file=args.train,
        dev_file=args.dev,
        distributed=False,
        num_workers=1,
        ignore_unknowns=True,
        concat=False,
        gold_questions=True,
        for_evaluation=for_evaluation,
    )

    run_model(
        model,
        config=config,
        train_dataloader=train_loaders,
        dev_dataloader=val_loaders,
        test_dataloader=val_loaders,
        save_always=True,
    )


def run_re_concat_qa(args):
    """Run the relation-extraction qa models using the concat of head entity
    and the relation."""
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
        answer_training_steps=args.answer_training_steps,
        seed=args.seed,
    )

    set_random_seed(config.seed)
    model = T5QA(config)

    (
        train_loaders,
        val_loaders,
        train_dataset,
        val_dataset,
        train_sample,
    ) = create_zero_re_qa_dataset(
        question_tokenizer=model.tokenizer,
        answer_tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        source_max_length=config.source_max_length,
        decoder_max_length=config.decoder_max_length,
        train_file=args.train,
        dev_file=args.dev,
        distributed=False,
        num_workers=1,
        ignore_unknowns=True,
        concat=True,
        gold_questions=False,
        for_evaluation=for_evaluation,
    )

    run_model(
        model,
        config=config,
        train_dataloader=train_loaders,
        dev_dataloader=val_loaders,
        test_dataloader=val_loaders,
        save_always=True,
    )


def run_re_qa(args):
    """Run the relation-extraction qa models using the question generator and
    the response generator explored with some search algorithm."""
    if args.mode == "re_qa_train":
        mode = "train"
        '''
        ngpus_per_node = torch.cuda.device_count()

        """ This next line is the key to getting DistributedDataParallel working on SLURM:
            SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
            current process inside a node and is also 0 or 1 in this example."""

        local_rank = int(os.environ.get("SLURM_LOCALID"))

        rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank

        """ This next block parses CUDA_VISIBLE_DEVICES to find out which GPUs 
        have been allocated to the job, then sets torch.device to the GPU corresponding
        to the local rank (local rank 0 gets the first GPU, local rank 1 gets the second GPU etc) """

        available_gpus = list(os.environ.get("CUDA_VISIBLE_DEVICES").replace(",", ""))

        current_device = int(available_gpus[local_rank])
        torch.cuda.set_device(current_device)

        print("From Rank: {}, ==> Initializing Process Group...".format(rank))

        # init the process group
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.init_method,
            world_size=args.world_size,
            rank=rank,
        )
        print("process group ready!")

        print("From Rank: {}, ==> Making model..".format(rank))
        '''
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
            update_switch_steps=int(args.update_switch_steps),
            seed=args.seed,
        )
        set_random_seed(config.seed)
        model = REQA(config)
        model = model.to("cuda:0")

        # model = torch.nn.parallel.DistributedDataParallel(
        #    model, device_ids=[0]
        # )

        # print("From Rank: {}, ==> Preparing data..".format(rank))

        (
            train_loaders,
            val_loaders,
            train_dataset,
            val_dataset,
            train_sampler,
        ) = create_zero_re_qa_dataset(
            question_tokenizer=model.question_tokenizer,
            answer_tokenizer=model.answer_tokenizer,
            batch_size=config.batch_size,
            source_max_length=config.source_max_length,
            decoder_max_length=config.decoder_max_length,
            train_file=args.train,
            dev_file=args.dev,
            distributed=False,
            num_workers=args.num_workers,
            ignore_unknowns=True,
            concat=False,
            gold_questions=False,
        )
        # To train the question model, we do not use the negative data with unknown answers.
        (
            question_train_loaders,
            question_val_loaders,
            question_train_dataset,
            question_val_dataset,
            question_train_sampler,
        ) = create_zero_re_qa_dataset(
            question_tokenizer=model.question_tokenizer,
            answer_tokenizer=model.answer_tokenizer,
            batch_size=config.batch_size,
            source_max_length=config.source_max_length,
            decoder_max_length=config.decoder_max_length,
            train_file=args.train,
            dev_file=args.dev,
            distributed=False,
            num_workers=args.num_workers,
            ignore_unknowns=True,
            concat=False,
            gold_questions=False,
        )

        """
        (
            real_question_loader,
            real_question_dataset,
            real_question_sampler,
        ) = create_question_generation_dataset(
            question_tokenizer=model.question_tokenizer,
            batch_size=config.batch_size,
            source_max_length=config.source_max_length,
            decoder_max_length=config.decoder_max_length,
            distributed=False,
            num_workers=args.num_workers,
        )
        """
        iterative_run_model(
            model,
            config=config,
            train_dataloader=train_loaders,
            dev_dataloader=val_loaders,
            test_dataloader=val_loaders,
            question_train_dataloader=question_train_loaders,
            question_dev_dataloader=question_val_loaders,
            question_test_dataloader=question_val_loaders,
            save_always=True,
            rank=0,
            train_samplers=[train_sampler],
            question_train_samplers=[question_train_sampler],
            current_device=0,
            gold_eval_file=args.dev,
            real_question_dataloader=None,
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

        (_, val_loaders, _, val_dataset, _,) = create_zero_re_qa_dataset(
            question_tokenizer=model.question_tokenizer,
            answer_tokenizer=model.answer_tokenizer,
            batch_size=config.batch_size,
            source_max_length=config.source_max_length,
            decoder_max_length=config.decoder_max_length,
            dev_file=args.dev,
            distributed=False,
            num_workers=args.num_workers,
            ignore_unknowns=True,
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


def run_main(args):
    """Decides what to do in the code."""
    if args.mode in ["re_gold_qa_train", "re_gold_qa_test"]:
        run_re_gold_qa(args)
    if args.mode in ["re_concat_qa_train", "re_concat_qa_test"]:
        run_re_concat_qa(args)
    if args.mode in ["re_qa_train", "re_qa_test"]:
        run_re_qa(args)


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
        "--model_path",
        type=str,
        required=True,
        help="Path for saving or loading models.",
    )

    # Train specific
    parser.add_argument("--train", type=str, help="file for train data.")

    parser.add_argument("--dev", type=str, help="file for validation data.")

    parser.add_argument("--concat_questions", type=bool, default=False)

    parser.add_argument("--ignore_unknowns", type=bool, default=False)

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
    parser.add_argument("--num_beams", type=int, default=32, help="Number of beam size")
    parser.add_argument(
        "--num_search_samples", type=int, default=8, help="Number of search samples"
    )
    parser.add_argument(
        "--update_switch_steps",
        type=int,
        default=10,
        help="Number of steps to train each question or response module before switching to train the other one!",
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=4,
        help="Number of beam groups for diverse beam.",
    )
    parser.add_argument(
        "--beam_diversity_penalty",
        type=float,
        default=0.5,
        help="Diversity penalty in diverse beam.",
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
        "--partition_checkpoint",
        type=str,
        help="checkpoint of the trained partition model.",
    )
    parser.add_argument(
        "--answer_training_steps",
        type=int,
        help="number of training steps over the train data for the answer model.",
    )
    parser.add_argument(
        "--training_steps",
        type=int,
        help="number of training steps over the train data.",
    )
    parser.add_argument(
        "--question_training_steps",
        type=int,
        help="number of training steps over the train data for the question model.",
    )
    parser.add_argument(
        "--init_method",
        default="tcp://127.0.0.1:3456",
        type=str,
        help="I guess the address of the master",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="")
    parser.add_argument("--world_size", default=1, type=int, help="")
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="number of sub processes per main process of gpu to load data",
    )
    parser.add_argument("--distributed", action="store_true", help="")

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    run_main(args)
