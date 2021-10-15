import csv
import io
import math
import os
import time
from configparser import ConfigParser
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data.distributed

from src.re_qa_model import HyperParameters, save


def white_space_fix(text):
    return " ".join(text.split())


def run_predict(model, dev_dataloader, prediction_file: str, current_device) -> None:
    """Read the 'dev_dataset' and predict results with the model, and save the
    results in the prediction_file."""
    writerparams = {"quotechar": '"', "quoting": csv.QUOTE_ALL}
    with io.open(prediction_file, mode="w", encoding="utf-8") as out_fp:
        writer = csv.writer(out_fp, **writerparams)
        header_written = False
        for batch in dev_dataloader:
            for ret_row in model.predict_step(batch, current_device):
                if not header_written:
                    headers = ret_row.keys()
                    writer.writerow(headers)
                    header_written = True
                writer.writerow(list(ret_row.values()))


def save_config(config: HyperParameters, path: str) -> None:
    """Saving config dataclass."""

    config_dict = vars(config)
    parser = ConfigParser()
    parser.add_section("train-parameters")
    for key, value in config_dict.items():
        parser.set("train-parameters", str(key), str(value))
    # save to a file
    with io.open(
        os.path.join(path, "config.ini"), mode="w", encoding="utf-8"
    ) as configfile:
        parser.write(configfile)


def iterative_run_model(
    model,
    config,
    train_dataloader=None,
    dev_dataloader=None,
    test_dataloader=None,
    save_always: Optional[bool] = False,
    rank=0,
    train_samplers=None,
    current_device=0,
    gold_eval_file=None,
    train_method="MML-MML",
) -> None:
    """Run the model on input data (for training or testing)"""

    model_path = config.model_path
    max_epochs = config.max_epochs
    mode = config.mode
    if mode == "train":
        print("\nRank: {0} | INFO: ML training\n".format(rank))
        first_start = time.time()
        epoch = 0
        while epoch < max_epochs:
            # let all processes sync up before starting with a new epoch of training
            # dist.barrier()

            # make sure we get different orderings.
            # for sampler in train_samplers:
            #    sampler.set_epoch(epoch)

            # print("\nRank: {0} | Epoch:{1}\n".format(rank, epoch))
            start = time.time()

            data_iter = iter(train_dataloader)
            step = 0
            total_loss = []
            mean_loss = 0.0
            while step < config.training_steps:
                for inner_step in range(config.update_switch_steps):
                    data_batch = next(data_iter)
                    loss = model.iterative_train(
                        data_batch,
                        current_device,
                        phase=train_method,
                        sample_p=0.95,
                    )
                    if math.isnan(loss):
                        print("NAN")
                        exit()
                    if loss and not math.isinf(loss):
                        total_loss.append(loss)

                    if total_loss:
                        mean_loss = np.mean(total_loss)

                    print(
                        "\rRank:{0} | Batch:{1} | Loss:{2} | Mean Loss:{3} | GPU Usage:{4}\n".format(
                            rank,
                            step + inner_step + 1,
                            loss,
                            mean_loss,
                            torch.cuda.memory_allocated(device=current_device),
                        )
                    )

                step += config.update_switch_steps
                if rank == 0 and save_always and step > 0 and (step % 100 == 0):
                    save(
                        model.question_model,
                        model.model_path,
                        str(epoch) + "_question_step_" + str(step),
                    )
                    save(
                        model.answer_model,
                        model.model_path,
                        str(epoch) + "_answer_step_" + str(step),
                    )

                # if save_always and step > 0 and (step % 100 == 0):
                #    dist.barrier()

            if rank == 0 and save_always:
                save(
                    model.question_model,
                    model.model_path,
                    str(epoch) + "_question_full",
                )
                save(
                    model.answer_model,
                    model.model_path,
                    str(epoch) + "_answer_full",
                )

            msg = "\nRank: {0} | Epoch training time: {1} seconds\n".format(
                rank, time.time() - start
            )
            print(msg)
            epoch += 1

        if rank == 0:
            save_config(config, model_path)

        msg = "\nRank: {0} | Total training time: {1} seconds\n".format(
            rank, time.time() - first_start
        )
        print(msg)

    elif mode == "test":
        print("Predicting...")
        start = time.time()
        run_predict(model, test_dataloader, config.prediction_file, current_device)
        msg = "\nTotal prediction time:{} seconds\n".format(time.time() - start)
        print(msg)
