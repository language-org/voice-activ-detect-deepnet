# author: steeve LAQUITAINE
# purpose:
#   define and tag all the existing pipelines for running
#
# usage:
#   ```bash
#   kedro run --pipeline train --env train
#   ```
# or
#   ```bash
#   kedro run --pipeline predict --env predict
#   ````

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
        "data_eng": Pipeline([data_eng.run_for_train()]), # data engineering pipeline
        "train": Pipeline([data_eng.run_for_train(), train.run()]), # training pipeline
        "predict": Pipeline([data_eng.run_for_inference(), inference.run()]), # prediction pipeline
        "predict_and_eval": Pipeline(
            [data_eng.run_for_inference(), inference.run(), evaluate.run()] # prediction and evaluation pipeline
        ),
    }
