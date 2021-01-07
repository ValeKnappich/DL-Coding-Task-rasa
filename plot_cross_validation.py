import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import click
import pandas as pd
import json
from os.path import isfile

@click.command()
def plot():
    """Plot the CV results"""
    if not isfile("CV-Results.json"):
        raise FileNotFoundError(f"{join(os.getcwd(), 'CV-Results.json')} does not exist, run cross-validation first.")

    data = json.load(open("CV-Results.json", "r"))
    data_list = []
    for config in data.keys():
        for mode in data[config].keys():
            for task in data[config][mode].keys():
                for metric, value in zip(["Accuracy", "F1-score", "Precision"], data[config][mode][task].values()):
                    data_list.append((config, mode, task, metric, value))
    
    df = pd.DataFrame(data_list, columns=["Config", "Mode", "Task", "Metric", "Value"])
    fig = px.bar(
        df, x="Config", y="Value", color="Metric", barmode="group", 
        facet_col="Task", facet_row="Mode", log_y=True
    )
    fig.write_image("CV-Results.png")

if __name__ == "__main__":
    plot()