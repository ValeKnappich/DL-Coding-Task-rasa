import os
from os.path import join
import click
import rasa
import json

from utils import validate_data_path, pipe_warnings_to_file, read_metrics

pipe_warnings_to_file("warnings.log")


@click.command()
@click.option("-f", "--folds", type=int, default=3, help="Number of folds")
@click.option("-d", "--data", default="data", help="File or directory containing the training data")
def main(folds, data):
    """
    Perform a k-fold cross-validation on one or multiple configurations. The results are averaged across runs and saved to disk.
    """
    data = validate_data_path(data)

    configs = list(os.listdir("config"))
    metrics = {}
    nlu_data = rasa.shared.nlu.training_data.loading.load_data(data)
    for config in configs:
        print(f"Starting to train {config}")
        results = rasa.nlu.cross_validate(nlu_data, n_folds=folds, nlu_config=join("config", config), disable_plotting=True)
        metrics[config] = read_metrics(results)

    json.dump(metrics, open("CV-Results.json", "w"), indent=4)

        

    
if __name__ == "__main__":
    main()