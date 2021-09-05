# author: steeve LAQUITAINE
# purpose:
#   module that contains the inference pipeline that is run by pipeline_registry.py
#   when you call kedro run --pipeline ...
#
# usage:
#
#   from vad.pipelines.inference import pipeline


from kedro.pipeline import Pipeline, node
from vad.pipelines.inference.nodes import predict


def run(**kwargs):
    """Pipeline run after a data engineering and training pipeline to run inference

    Returns:
        (Pipeline): an inference pipeline graph
    """
    return Pipeline(
        [
            node(
                func=predict,
                inputs=["model", "prod_audio"],
                outputs="prediction",
                name="predict-production-audio",
            )
        ]
    )
