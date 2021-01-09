from utils import validate_data_path, read_metrics, pipe_warnings_to_file, import_rasa_functions

from hyperopt import hp, fmin, tpe, space_eval, Trials
import rasa
import click
import json
import pickle
import git
import os
from os.path import isdir, join, dirname, isfile

run_prediction, eval_intents, eval_entities = import_rasa_functions()

pipe_warnings_to_file("hp.log")

search_space = {
    "max_char_ngram": hp.quniform("max_char_ngram", 2, 4, 1),
    "epochs": hp.quniform("epochs", 8, 50, 1),
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
test_results = {}
repo = git.Repo(".")


def git_push(file, message):
    global repo
    repo.git.add(file)
    repo.index.commit(message)
    repo.remote(name="origin").push()


def export_and_push_results():
    global test_results
    if isfile("HP-Results.json"):
        old_results = json.load(open("HP-Results.json", "r"))
    else:
        old_results = {}
    json.dump(dict(old_results, **test_results), open("HP-Results.json", "w"), indent=4)
    git_push("HP-Results.json", "Update HP Results")


def export_config(args, file=config_tmp):
    args = {key: types.get(key, int)(value) for key, value in args.items()}
    with open(file, "w") as fp:
        fp.write(config_template.format(**args))


def target(args):
    global nlu_data, config_tmp, test_results
    export_config(args)
    data = rasa.shared.nlu.training_data.loading.load_data(nlu_data)
    # data, _ = data.train_test_split(train_frac=0.1)
    train, test = data.train_test_split(train_frac=0.7)
    trainer = rasa.nlu.model.Trainer(rasa.nlu.config.load(config_tmp))
    interpreter = trainer.train(train)
    i_results, _, e_results = run_prediction(interpreter, test)
    i_eval = eval_intents(i_results, None, False, False, True, True)
    e_eval = eval_entities(e_results, {"DIETClassifier"}, None, False, False, True, True)
    import pdb; pdb.set_trace()
    metrics = {
        "intent": {
            metric: i_eval[metric] for metric in ["precision", "f1_score", "accuracy"]
        }, "entity": {
            metric: e_eval["DIETClassifier"][metric] for metric in ["precision", "f1_score", "accuracy"]
        }
    }
    test_results[str(args)] = metrics
    export_and_push_results()
    combined_score = (metrics["intent"]["accuracy"] + metrics["entity"]["f1_score"]) / 2
    return 1 - combined_score


def setup(config, data_path):
    global config_template, nlu_data
    with open(config, "r") as fp:
        config_template = fp.read()
    nlu_data = validate_data_path(data_path)


def cleanup():
    to_delete = [config_tmp] #, "data/tmp*"]
    for file in to_delete:
        os.system(f"rm -f {file}")


@click.command()
@click.option("-c", "--config", default="config/hp-template.yml")
@click.option("-o", "--out-config", default="config/config-best-hp.yml")
@click.option("-d", "--data", default="data")
@click.option("-e", "--evals", default=20, type=int)
def main(config, out_config, data,evals):
    setup(config, data)
    if isfile("hp-trials.pkl"):
        trials = pickle.load(open("hp-trials.pkl", "rb"))
    else:
        trials = Trials()
    best = fmin(target, search_space, trials=trials, algo=tpe.suggest, max_evals=len(trials.trials) + evals, show_progressbar=False)
    export_config(space_eval(search_space, best), out_config)
    pickle.dump(trials, open("hp-trials.pkl", "wb"))
    git_push("hp-trials.pkl", "Update HP Results")
    cleanup()
    
    
if __name__ == "__main__":
    main()