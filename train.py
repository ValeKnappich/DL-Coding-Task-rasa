import click
import rasa
import os
from os.path import isfile, join, isdir
import logging
import warnings

from utils import validate_config, validate_data_path, pipe_warnings_to_file

pipe_warnings_to_file("warnings.log")


def add_evaluation_to_config(config: rasa.nlu.config.RasaNLUModelConfig, eval_frac: int):
    to_add = {
        "evaluate_every_number_of_epochs": 1,
        "evaluate_on_number_of_examples": eval_frac,
        "tensorboard_log_directory": "tensorboard",
        "tensorboard_log_level": "epoch"
    }
    config.pipeline[-1].update(to_add)
    return config


@click.command()
@click.option("-c", "--config", default=None, help="Name of the chosen config")
@click.option("-d", "--data", default="data", help="File or directory containing the training data")
@click.option("-f", "--train-frac", default=0.8, help="Fraction of data to use for training, rest for validation")
def main(config, data, train_frac):
    data = validate_data_path(data)
    nlu_data = rasa.shared.nlu.training_data.loading.load_data(data)
    
    config = rasa.nlu.config.load(validate_config(config))
    config = add_evaluation_to_config(config, int((1 - train_frac) * len(nlu_data.nlu_examples)))

    trainer = rasa.nlu.model.Trainer(config)
    trainer.train(nlu_data)
    model_path = trainer.persist("model")


if __name__ == "__main__":
    main()
