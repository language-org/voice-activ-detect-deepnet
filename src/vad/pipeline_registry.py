# author: steeve LAQUITAINE

"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from vad.pipelines.data_eng import pipeline as data_eng
from vad.pipelines.train import pipeline as train
from vad.pipelines.inference import pipeline as inference
from vad.pipelines.evaluate import pipeline as evaluate


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {
        "data_eng": Pipeline([data_eng.run_for_train()]),
        "train": Pipeline([data_eng.run_for_train(), train.run()]),
        "predict": Pipeline([data_eng.run_for_inference(), inference.run()]),
        "predict_and_eval": Pipeline(
            [data_eng.run_for_inference(), inference.run(), evaluate.run()]
        ),
    }
