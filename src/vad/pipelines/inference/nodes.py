# author: steeve LAQUITAINE
# purpose:
#   module that contains functions to run inference
# usage:
#
#   from vad.pipelines.inference.nodes import predict

import numpy as np
import pandas as pd
from typing import Any, Dict


def predict(model, prod_audio: Dict[str, Any]) -> Dict[str, Any]:
    """Use "model" to make predictions on "prod_audio"

    Args:
        model ([type]): trained model
        prod_audio (Dict[str, Any]): audio that must be labelled

    Returns:
        [type]: [description]
    """
    prediction_proba = model.predict(prod_audio["audio"]["data"])

    # case prediction probabilities are shaped as
    # (s samples, t timesteps, 2 OHE classes)
    if prediction_proba.ndim == 3:
        prediction_proba = prediction_proba[:, -1, :]
    prediction = np.argmax(prediction_proba, axis=1).astype(int)
    timestamp = prod_audio["audio"]["metadata"]["timestamp"]
    return pd.DataFrame(data={"timestamp": timestamp, "prediction": prediction})
