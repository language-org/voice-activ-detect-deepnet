from kedro.pipeline import Pipeline, node
from vad.pipelines.evaluate.nodes import Validation


def run(**kwargs):
    return Pipeline(
        [
            node(
                func=Validation.reshape_label,
                inputs="prod_audio",
                outputs="label_for_eval",
                name="reshape-label",
            ),
            node(
                func=Validation.reshape_prediction,
                inputs="prediction",
                outputs="prediction_for_eval",
                name="reshape-prediction",
            ),
            node(
                func=Validation.evaluate,
                inputs=["prediction_for_eval", "label_for_eval"],
                outputs="metrics",
                name="evaluate-predictions",
            ),
            node(
                func=Validation.get_confusion,
                inputs=["prediction_for_eval", "label_for_eval"],
                outputs="confusion",
                name="get-confusion",
            ),
        ]
    )
