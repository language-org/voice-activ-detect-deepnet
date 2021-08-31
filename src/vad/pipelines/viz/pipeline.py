from kedro.pipeline import Pipeline, node
from vad.pipelines.train.nodes import train_and_log


def run(**kwargs):
    return Pipeline([])
