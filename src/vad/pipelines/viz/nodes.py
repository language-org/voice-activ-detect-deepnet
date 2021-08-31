from ipdb import set_trace
import os
import scipy.io.wavfile
import pandas as pd
import numpy as np

# from tensorflow import keras
from matplotlib import pyplot as plt


class Viz:
    @staticmethod
    def plot_labelled_audio(timestamp, audio, labels, n_sample):

        # plot audio
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(timestamp, audio[:n_sample], "r-")

        # plot label
        ax2.plot(timestamp, labels[:n_sample], "b-")
        ax2.set_ylim([-0.1, 1.1])

        plt.show()

    @staticmethod
    def plot_predictions(preds, Y_test, n_sample):

        # plot predictions
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(preds[:n_sample], "r-.")

        # plot ground truth
        ax2.plot(Y_test[:n_sample], "b-.")
        plt.show()
