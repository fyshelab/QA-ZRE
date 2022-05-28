import csv
import io
import math
import os
import time
from configparser import ConfigParser
from typing import Optional

import numpy as np
import torch

from src.re_qa_model import HyperParameters, save


def run_predict(
    model, dev_dataloader, prediction_file: str, current_device, predict_type="entity"
):
    """Read the 'dev_dataset' and predict results with the model, and save the
    results in the prediction_file."""
    writerparams = {"quotechar": '"', "quoting": csv.QUOTE_ALL}
    with io.open(prediction_file, mode="w", encoding="utf-8") as out_fp:
        writer = csv.writer(out_fp, **writerparams)
        header_written = False
        for batch in dev_dataloader:
            if predict_type == "entity":
                for ret_row in model.predict_step(batch, current_device):
                    if not header_written:
                        headers = ret_row.keys()
                        writer.writerow(headers)
                        header_written = True
                    writer.writerow(list(ret_row.values()))
            elif predict_type == "relation":
                for ret_row in model.relation_classifier(batch, current_device):
                    if not header_written:
                        headers = ret_row.keys()
                        writer.writerow(headers)
                        header_written = True
                    writer.writerow(list(ret_row.values()))


def save_config(config: HyperParameters, path: str):
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
    test_dataloader=None,
    save_always: Optional[bool] = False,
    current_device=0,
    train_method="MML-MML-On-Sim",
) -> None:
    """Run the model on input data (for training or testing)"""

    model_path = config.model_path
    max_epochs = config.max_epochs
    mode = config.mode
    if mode == "train":
        print("\nINFO: ML training\n")
        first_start = time.time()
        epoch = 0
        while epoch < max_epochs:
            start = time.time()
            data_iter = iter(train_dataloader)
            step = 0
            total_loss = []
            mean_loss = 0.0
            while step < config.training_steps:
                data_batch = next(data_iter)
                loss = model.train_objectives(
                    data_batch,
                    current_device,
                    objective_type=train_method,
                    sample_p=0.95,
                )

                if type(loss) == tuple:
                    # This not very important it is for showing the average of MML + PGG loss during training.
                    avg_loss = loss[0] + loss[1] / 2.0

                else:
                    avg_loss = loss

                if avg_loss and not math.isinf(avg_loss):
                    total_loss.append(avg_loss)

                if total_loss:
                    mean_loss = np.mean(total_loss)

                print(
                    "\rBatch:{0} | Loss:{1} | Mean Loss:{2} | GPU Usage:{3}\n".format(
                        step + 1,
                        loss,
                        mean_loss,
                        torch.cuda.memory_allocated(device=current_device),
                    )
                )

                step += 1
                if save_always and step > 0 and (step % 100 == 0):
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

            if save_always:
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

            msg = "\nEpoch training time: {0} seconds\n".format(time.time() - start)
            print(msg)
            epoch += 1

        save_config(config, model_path)

        msg = "\nTotal training time: {0} seconds\n".format(time.time() - first_start)
        print(msg)

    elif mode == "test":
        print("Predicting...")
        start = time.time()
        run_predict(
            model,
            test_dataloader,
            config.prediction_file,
            current_device,
            config.predict_type,
        )
        msg = "\nTotal prediction time:{0} seconds\n".format(time.time() - start)
        print(msg)
