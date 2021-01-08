from utils import validate_data_path, read_metrics, pipe_warnings_to_file

from hyperopt import hp, fmin, tpe, space_eval
import rasa
import click
import json
import os
from os.path import isdir, join, dirname
from contextlib import redirect_stdout



# pipe_warnings_to_file("hp.log")

search_space = {
    "max_char_ngram": hp.quniform("max_char_ngram", 2, 4, 1),
    # "epochs": hp.quniform("epochs", 8, 50, 1),
    "epochs": hp.quniform("epochs", 1, 1, 1),
    "ff_layers": hp.choice("ff_layers", [
        "[]", "[128]", "[256]", "[128, 256]", "[256, 128]"
    ]),
    "num_transformer_layers": hp.quniform("tr_layers", 2, 4, 1),
    "embedding_dim": hp.quniform("emb_dim", 20, 40, 1),
    "transformer_size": hp.choice("tr_size", [128, 256, 512])
}

types = { # default: int
    "ff_layers": str
}

config_template = None
config_out = None
config_tmp = "config/current-hp-conf.yml"
nlu_data = None
folds = None
cv_results = {}


def dict_to_tuple(d):
    # transform dict with literal values to tuple
    return tuple((key, value) for key, value in d.items())


def export_config(args, file=config_tmp):
    args = {key: types.get(key, int)(value) for key, value in args.items()}
    with open(file, "w") as fp:
        fp.write(config_template.format(**args))


def target(args):
    global nlu_data, folds, config_tmp, cv_results
    export_config(args)
    data = rasa.shared.nlu.training_data.loading.load_data(nlu_data)
    with open("hp.log", "w") as fp:
        with redirect_stdout(fp):
            results = rasa.nlu.cross_validate(data, n_folds=folds, nlu_config=config_tmp, disable_plotting=True)
    metrics = read_metrics(results)
    cv_results[dict_to_tuple(args)] = metrics
    combined_score = (metrics["test"]["intent"]["Accuracy"] + metrics["test"]["entity"]["F1-score"]) / 2
    return 1 - combined_score


def setup(config, data_path, n_folds):
    global config_template, nlu_data, folds
    with open(config, "r") as fp:
        config_template = fp.read()
    nlu_data = validate_data_path(data_path)
    folds = n_folds


def cleanup():
    to_delete = [config_tmp] #, "data/tmp*"]
    for file in to_delete:
        os.system(f"rm -f {file}")


@click.command()
@click.option("-c", "--config", default="config/hp-template.yml")
@click.option("-o", "--out-config", default="config/config-best-hp.yml")
@click.option("-d", "--data", default="data")
@click.option("-f", "--folds", default=3, type=int)
@click.option("-e", "--evals", default=20, type=int)
def main(config, out_config, data, folds, evals):
    global cv_results
    setup(config, data, folds)
    best = fmin(target, search_space, algo=tpe.suggest, max_evals=evals)
    export_config(space_eval(search_space, best), out_config)
    cleanup()
    json.dump(cv_results, open("HP-Results.json", "w"))
    

if __name__ == "__main__":
    main()