# author: steeve LAQUITAINE
# purpose:
#   module that contains the training pipeline that is run by pipeline_registry.py
#   when you call kedro run --pipeline ...
#
# usage:
#
#   from vad.pipelines.train import pipeline


from kedro.pipeline import Pipeline, node
from vad.pipelines.train.nodes import train_and_log


def run(**kwargs):
    """Pipeline run after a data engineering to run training

    Returns:
        (Pipeline): an training pipeline graph
    """
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
