"""This is the main module to launch the pre-training steps for the question
and response generation."""

import csv
import io
import os
from typing import Any, Callable, Iterator, Optional, Tuple

import numpy as np
import torch
from absl import app, flags
from torch.utils.tensorboard import SummaryWriter

from src.metrics import compute_response_f1
from src.models import QAT5, MyBaseT5
from src.question_utils import create_question_pretrain_dataset
from src.response_utils import create_response_dataset

FLAGS = flags.FLAGS
flags.DEFINE_integer("max_epochs", 10, "The maximum number of epochs for training.")
flags.DEFINE_integer(
    "training_steps", 100, "The number of training steps for each epoch."
)
flags.DEFINE_integer(
    "steps_per_checkpoint",
    100,
    "keep checkpoint of the model every this number of steps",
)
flags.DEFINE_string(
    "prediction_file",
    "/tmp/predictions.csv",
    "the path/name for saving the predictions.",
)
flags.DEFINE_string("dev_file", "/tmp/dev.csv", "the path/name of the dev file.")
flags.DEFINE_string(
    "task_name", "question_pretrain", "the name of the downstream nlp task."
)
flags.DEFINE_string("train_file", "/tmp/train.csv", "the path/name of the train file.")
flags.DEFINE_integer("batch_size", 16, "The batch size used for training or inference.")
flags.DEFINE_integer(
    "source_max_length",
    128,
    "The maximum number of tokens consider in the input sequence.",
)
flags.DEFINE_integer(
    "decoder_max_length",
    128,
    "The maximum number of tokens consider in the output sequence.",
)


def start_training(
    model: MyBaseT5, dataloader: torch.utils.data.DataLoader
) -> Iterator[Tuple[int, float]]:
    """Pick a batch from the dataloader, and train the model for one step."""
    step = 0
    for batch in dataloader:
        loss_values = model.train(batch)
        step += 1
        yield step, loss_values["loss_value"]


def start_predicting(
    model: MyBaseT5, dataloader: torch.utils.data.DataLoader, prediction_file: str
) -> None:
    """Read batches from the dataloader and predict the outputs from the model
    for the correct experiment and save the results in the prediction_file as
    csv format row by row."""
    with io.open(prediction_file, mode="w", encoding="utf-8") as out_fp:
        writer = csv.writer(out_fp, quotechar='"', quoting=csv.QUOTE_ALL)
        header_written = False
        for batch in dataloader:
            for ret_row in model.predict(batch):
                if not header_written:
                    headers = ret_row.keys()
                    writer.writerow(headers)
                    header_written = True
                writer.writerow(list(ret_row.values()))
    return


def run_model(
    model: MyBaseT5,
    train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
    metric: Optional[Callable[[str, str, str], float]] = None,
) -> None:
    """Run the model on input data; for training or inference."""
    if FLAGS.mode == "train":
        writer = SummaryWriter(FLAGS.model_path)
        epoch = 0
        global_step = 0
        total_loss = []
        best_score = float("-inf")
        eval_file = os.path.join(FLAGS.model_path, "temp_eval.csv")
        while epoch < FLAGS.max_epochs:
            print("\nEpoch:{0}\n".format(epoch))
            epoch_loss = []
            for step, loss in start_training(model, train_dataloader):
                global_step += 1
                total_loss.append(loss)
                epoch_loss.append(loss)
                mean_total_loss = np.mean(total_loss)
                mean_epoch_loss = np.mean(epoch_loss)
                print(
                    "\rEpoch:{0} | Batch:{1} | Mean Loss:{2} | Epoch Loss:{3} | Loss:{4}\n".format(
                        epoch, step, mean_total_loss, mean_epoch_loss, loss
                    )
                )
                if global_step % FLAGS.steps_per_checkpoint == 0:
                    if eval_dataloader is not None:
                        start_predicting(model, eval_dataloader, eval_file)
                        score = metric(FLAGS.dev_file, eval_file, FLAGS.task_name)  # type: ignore
                        writer.add_scalar("Score/dev", score, global_step)
                        if score > best_score:
                            best_score = score
                            model.save("best_step")

                writer.add_scalar("Mean_Total_Loss/train", mean_total_loss, global_step)
                writer.add_scalar("Mean_Epoch_Loss/train", mean_epoch_loss, global_step)
                writer.flush()
                if global_step == FLAGS.training_steps:
                    # stop training in this epoch.
                    break

            if eval_dataloader is not None:
                # do final evaluation on the dev data at the end of epoch.
                start_predicting(model, eval_dataloader, eval_file)
                score = metric(FLAGS.dev_file, eval_file, FLAGS.task_name)  # type: ignore
                writer.add_scalar("Score/dev", score, global_step)
                if score > best_score:
                    best_score = score
                    model.save("best_step")
            else:
                model.save(f"{global_step}_step")
            epoch += 1

        writer.close()

        if eval_dataloader is not None:
            # delete the eval_file
            os.remove(eval_file)

    if FLAGS.mode in ["test", "inference", "eval", "no_finetune_test"]:
        print("Predicting...")
        start_predicting(model, eval_dataloader, FLAGS.prediction_file)


def launch_qa_pretrain() -> None:
    """launch the pre-training phase for question and response generation."""

    FLAGS.mode = "train"

    model = QAT5()
    if FLAGS.task_name == "question_pretrain":
        # Run the T5 to do the pretraining of the question module.
        train_dataloader = create_question_pretrain_dataset(
            question_tokenizer=model.tokenizer,
            batch_size=FLAGS.batch_size,
            source_max_length=FLAGS.source_max_length,
            decoder_max_length=FLAGS.decoder_max_length,
            shuffle=True,
        )
        run_model(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=None,
            metric=None,
        )
    elif FLAGS.task_name == "response_pretrain":
        # Run the T5 on multiple qa datasets to pre-train the response generator.
        # Test the T5 for response generation on the squad v2 dev data as you train.
        train_loader, _, _ = create_response_dataset(
            tokenizer=model.tokenizer,
            batch_size=FLAGS.batch_size,
            source_max_length=FLAGS.source_max_length,
            decoder_max_length=FLAGS.decoder_max_length,
            dataset="all",
        )

        _, sq_val_loader, _ = create_response_dataset(
            tokenizer=model.tokenizer,
            batch_size=FLAGS.batch_size,
            source_max_length=FLAGS.source_max_length,
            decoder_max_length=FLAGS.decoder_max_length,
            dataset="squad_v2",
        )

        run_model(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=sq_val_loader,
            metric=compute_response_f1,
        )


def main(argv: Any) -> None:
    """Main function to switch over the t5 experiment type and launch the
    correct train script."""
    if FLAGS.t5_exp_type == "qa":
        launch_qa_pretrain()


if __name__ == "__main__":
    app.run(main)
