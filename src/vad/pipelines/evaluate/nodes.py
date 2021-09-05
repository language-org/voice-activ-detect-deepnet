# author: steeve LAQUITAINE
# purpose:
#   module that contains functions to evaluate inference
# usage:
#
#   from vad.pipelines.evaluate.nodes import Validation


from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import pandas as pd
from typing import Any, Dict
import numpy as np


class Validation:
    """Model performance validation class"""

    @staticmethod
    def plot_predictions(preds, Y_test, n_sample):

        # plot predictions
        f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
        ax1.plot(preds[:n_sample], "r-.")

        # plot ground truth
        ax2.plot(Y_test[:n_sample], "b-.")
        plt.show()

    @staticmethod
    def reshape_label(prod_audio: Dict[str, Any]) -> np.ndarray:
        """Reshape the labels

        Args:
            prod_audio (Dict[str, Any]): audio that must be labelled

        Returns:
            np.ndarray: [description]
        """
        # case labels are shaped as (s samples, 2 OHE classes)
        if prod_audio["label"].ndim == 2:
            return prod_audio["label"][:, 1]
        # case label are shaped as (s samples, t timesteps, 2 OHE classes)
        elif prod_audio["label"].ndim == 3:
            return prod_audio["label"][:, -1, 1]

    @staticmethod
    def reshape_prediction(prediction: pd.DataFrame):
        return prediction["prediction"].values

    @staticmethod
    def evaluate(prediction, label):
        accuracy = accuracy_score(label, prediction)
        precision = precision_score(label, prediction, average="binary", pos_label=1)
        recall = recall_score(label, prediction, average="binary", pos_label=1)
        f1 = f1_score(label, prediction, average="binary", pos_label=1)
        FRR = 1 - recall

        return pd.DataFrame(
            {
                "accuracy": [accuracy],
                "precision": [precision],
                "recall": [recall],
                "f1": [f1],
                "false_rejection_rate": [FRR],
            }
        )

    @staticmethod
    def get_confusion(prediction, label):
        LABELS = [False, True]
        confusion = confusion_matrix(label, prediction, labels=LABELS)
        return pd.DataFrame(
            confusion, index=["no speech", "speech"], columns=["no speech", "speech"]
        )
