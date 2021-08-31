from kedro.pipeline import Pipeline, node
from vad.pipelines.train.nodes import train_and_log


def run(**kwargs):
    return Pipeline(
        [
            node(
                func=train_and_log,
                inputs=[
                    "train_audio",
                    "train_label",
                    "test_audio",
                    "test_label",
                    "params:TRAIN",
                    "params:DATA_ENG",
                ],
                outputs="model",
                name="training",
            ),
        ]
    )
