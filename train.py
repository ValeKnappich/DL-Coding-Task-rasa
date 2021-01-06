import click
import rasa
import os
from os.path import isfile, join, isdir
import logging
import warnings

from utils import validate_config, validate_data_path, pipe_warnings_to_file

pipe_warnings_to_file("warnings.log")

@click.command()
@click.option("-c", "--config", default=None, help="Name of the chosen config")
@click.option("-d", "--data", default="data", help="File or directory containing the training data")
def main(config, data):
    config = validate_config(config)
    data = validate_data_path(data)

    nlu_data = rasa.shared.nlu.training_data.loading.load_data(data)
    trainer = rasa.nlu.model.Trainer(rasa.nlu.config.load(config))
    trainer.train(nlu_data)
    model_path = trainer.persist("model")


if __name__ == "__main__":
    main()
