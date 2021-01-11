import os
from os.path import isfile, isdir, join
import warnings
import click
from datetime import datetime


def validate_config(config):
    if not config:
        if "RASA_CONFIG" in os.environ:
            return validate_config(os.environ["RASA_CONFIG"])
        else:
            return validate_config(os.listdir("config")[0])
    else:
        if isfile(config):
            return config
        elif isfile(join("config", config)):
            return join("config", config)
        else:
            raise ValueError(
                f"Invalid config: no such file: {join(os.getcwd(), config)} \
                                   or {join(os.getcwd(), 'config', config)}"
            )


def validate_data_path(path):
    if isdir(path):
        if isfile(join(path, "train.yml")):
            return join(path, "train.yml")
        elif isfile(join(path, "nlu.yml")):
            return join(path, "nlu.yml")
        else:
            raise ValueError(f"Could not find file train.yml or nlu.yml in {join(os.getcwd(), path)}")
    elif isfile(path):
        return path
    else:
        raise ValueError(f"No such file or directory: {join(os.getcwd(), path)}")


def validate_model_path(model):
    if not model:
        models = os.listdir("model")
        dates = [datetime.strptime(model[4:], "%Y%m%d-%H%M%S") for model in models]
        return join("model", models[dates.index(max(dates))])
    elif not isdir(model):
        if isdir(join("model", model)):
            return join("model", model)
        else:
            raise ValueError("Model does not exist")
    else:
        return model
        

def pipe_warnings_to_file(file):
    logs_fp = open(file, "w")
    def customwarn(message, category, filename, lineno, file=None, line=None):
        logs_fp.write(warnings.formatwarning(message, category, filename, lineno))
    warnings.showwarning = customwarn


def read_metrics(results):
    return {
        mode: {
            "intent": {
                metric: sum(getattr(results[0], mode)[metric]) / len(getattr(results[0], mode)[metric])
                for metric in getattr(results[0], mode).keys()
            },
            "entity": {
                metric: sum(getattr(results[1], mode)["DIETClassifier"][metric]) / len(getattr(results[1], mode)["DIETClassifier"][metric])
                for metric in getattr(results[1], mode)["DIETClassifier"].keys()
            }
        } for mode in ("train", "test")
    }


def import_rasa_functions():
    """
    Ugly helper function, that imports functions, that were not exposed by rasa
    """
    from os.path import join, dirname
    import rasa
    from importlib.util import spec_from_file_location, module_from_spec

    path = join(dirname(rasa.nlu.__file__), "test.py")
    spec = spec_from_file_location("rasa.nlu.test", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_eval_data, module.evaluate_intents, module.evaluate_entities


@click.command()
def main():
    """
    Helper functions shared across scripts.
    """
    pass

if __name__ == "__main__":
    main()