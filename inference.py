import click
import rasa
import json
import os
from os.path import isdir, join
from datetime import datetime
from tqdm import tqdm

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


@click.command()
@click.option("-d", "--data", "path", default="data/dev.json", help="Path to dev.json")
@click.option("-o", "--out", "out_path", default="data/dev_labelled.json", help="Path of the output json")
@click.option("-m", "--model", default=None, help="Path to the trained model, defaults to the latest model")
def main(path, model, out_path):
    data = json.load(open(path, "r"))
    model_path = validate_model_path(model)

    interpreter = rasa.nlu.model.Interpreter.load(model_path)
    for index, instance in tqdm(data.items()):
        pred = interpreter.parse(instance["text"])
        data[index]["intent"] = pred["intent"]["name"]
        data[index]["positions"] = {entity["entity"]: [entity["start"], entity["end"]] for entity in pred["entities"]}
        data[index]["slots"] = {entity["entity"]: entity["value"] for entity in pred["entities"]}

    json.dump(data, open(out_path, "w"), indent=4)

if __name__ == "__main__":
    main()