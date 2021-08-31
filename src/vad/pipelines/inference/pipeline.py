from kedro.pipeline import Pipeline, node
from vad.pipelines.inference.nodes import predict


def run(**kwargs):
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
