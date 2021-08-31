#  author: steeve laquitaine
# purpose:
#   quickly build a minimal working prototype that can be trained within 1 hour and make prediction on test data
# usage:
#
#   ```bash
#   python notebooks/toy.py
#   ````
#
# [TODO]: check label imbalance and make training robust to imbalance

import os
import scipy.io.wavfile
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime

# from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from tensorboard.plugins.hparams import api as hp

# [TODO]: add to config .yml file
AUDIO_FILE = os.path.abspath("data/01_raw/vad_data/19-198-0003.wav")
LABEL_FILE = os.path.abspath("data/01_raw/vad_data/19-198-0003.json")
RUN_DIR = "tbruns/"

# ensure reproducbikity
np.random.seed(0)


class Etl:
    @staticmethod
    def read_X():
        """Load audio, convert to time series and calculate metadata (length..)

        [TODO]: return audio descriptive metadata

        returns sampling rate:
        resolution ? in bits
        nb of channels ?

        Args:
            flag ([type]): [description]
                audio: 2D N x 1 array
        """
        sample_rate, audio = scipy.io.wavfile.read(
            AUDIO_FILE
        )  # File assumed to be in the same directory
        sample_size = len(audio)
        time_unit = 1 / sample_rate
        duration_in_sec = time_unit * len(audio)

        return {
            "data": audio.reshape(sample_size, 1),
            "metadata": {
                "sample_rate": sample_rate,
                "sample_size": sample_size,
                "time_unit": time_unit,
                "duration_in_sec": duration_in_sec,
            },
        }

    @staticmethod
    def load_Y():
        """Load Labels use create time series based on audio metadata
        use sampling rate
        calculate resolution in bits from labels start_time and end_time data

        Args:
            flag ([type]): [description]
        """
        return pd.read_json(LABEL_FILE)

    @staticmethod
    def sync_audio_and_labels(audio, label):

        # get data
        data = audio["data"]
        sample_rate = audio["metadata"]["sample_rate"]
        time_unit = audio["metadata"]["time_unit"]
        sample_size = audio["metadata"]["sample_size"]

        # create timestamps
        audio["metadata"]["timestamp"] = np.arange(0, len(data), 1) * time_unit

        # initialize synchronized label vector
        synced_label = np.zeros((sample_size, 1))

        # synchronize labels by flagging
        # timestamps labelled as speech
        array = label.values
        for ix in range(array.shape[0]):
            interval = array[ix][0]
            speech_start = interval["start_time"]
            speech_end = interval["end_time"]
            span = np.where(
                np.logical_and(
                    audio["metadata"]["timestamp"] >= speech_start,
                    audio["metadata"]["timestamp"] <= speech_end,
                )
            )
            synced_label[span] = 1
        return audio, synced_label


class DataEng:
    @staticmethod
    def split(X, Y, frac):

        # [TODO]: add test that X_train.shape + X_test.shape = X.shape
        # [TODO]: same for Y

        n_train = int(np.round(X.shape[0] * frac))
        X_train = X[:n_train, :]
        X_test = X[n_train:, :]
        Y_train = Y[:n_train]
        Y_test = Y[n_train:]
        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def reshape_inputs(X, Y, timesteps):
        """[summary]

        The LSTM network expects the input data (X) to be provided with a specific
        array structure in the form of: [samples, time steps, features].

        # [TODO]: add unit testing X and Y must be same length

        Args:
            X ([type]): [description]
            look_back (int, optional): [description]. Defaults to 1.
                rows: time
                columns: features
                values: audio signal

        Returns:
            [type]:
                3D tensorflow arrays
                    Dim 1: number of samples
                    Dim 2: timesteps in the past
                    Dim 3: feature: predictors
        """

        dataX, dataY = [], []
        for i in range(len(X) - timesteps - 1):
            dataX.append(X[i : (i + timesteps), :])
        X = np.array(dataX)

        # drop Y's first n_look_back
        # based on the n_look_back past X
        Y = Y[timesteps + 1 :]
        return X, Y


class Validation:
    @staticmethod
    def plot_predictions(preds, Y_test, n_sample):

        # plot predictions
        f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
        ax1.plot(preds[:n_sample], "r-.")

        # plot ground truth
        ax2.plot(Y_test[:n_sample], "b-.")
        plt.show()

    @staticmethod
    def eval_model(Y_test, pred_test):

        LABELS = [False, True]

        accuracy = accuracy_score(Y_test, pred_test)
        precision = precision_score(Y_test, pred_test, average="binary", pos_label=1)
        recall = recall_score(Y_test, pred_test, average="binary", pos_label=1)
        f1 = f1_score(Y_test, pred_test, average="binary", pos_label=1)
        confusion = confusion_matrix(Y_test, pred_test, labels=LABELS)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion": confusion,
        }


if __name__ == "__main__":

    tic = time.time()

    # PARAMETERS --------------------------------------------
    # preprocessing
    TIMESTEPS = 1
    N_CLASSES = 2
    REDUCE_TRAIN = True

    # training undersampling
    TR_SAMPLE = 0.2

    # model
    VAL_FRAC = 0.6
    SPLIT_FRAC = 0.5
    MODEL_SEED = 0
    OUT_ACTIVATION = "sigmoid"
    LOSS = "categorical_crossentropy"
    OPTIM = "adam"
    EPOCH = 3
    BATCH_SIZE = 1
    VERBOSE = 2
    METRICS = ["Accuracy", "Precision", "Recall"]

    # hyperparameter search
    HP_N_GRU = hp.HParam("n_gru", hp.Discrete([1, 2]))

    # ETL --------------------------------------------

    # read and sync audio and labels
    audio = Etl.read_X()
    label = Etl.load_Y()
    audio, label = Etl.sync_audio_and_labels(audio, label)
    ts = audio["data"]
    Y = label

    # PREPROCESSING -----------------------------------

    """
    ------ Encode as 32 bits floats ------
    Prerequisite for tensorflow model training
    """
    ts = ts.astype("float32")
    Y = Y.astype("float32")

    """
    ------ MinMax Scale normalisation ------
    e.g., LSTM are sensitive to the scale
    particularly for sigmoid activ (defailt) or tanh
    """
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # dataset = scaler.fit_transform(dataset)

    """
    ------ Reshaping ------ 
    # reshape raw time series to net inputs
    # 3D input (batch, timestep, feature ?)
    # Works for LSTM, GRU and BiLSTM
    """
    X, Y = DataEng.reshape_inputs(ts, Y, timesteps=TIMESTEPS)

    # format labels
    Y = tf.keras.utils.to_categorical(Y, N_CLASSES)

    """
    ------ Split train/test ------     
    """
    X_train, X_test, Y_train, Y_test = DataEng.split(X, Y, frac=SPLIT_FRAC)

    """
    ------ REDUCE TRAIN SIZE FOR QUICK HYPERPARAMETER SEARCH ------     
    We only use a fraction of the training dataset to quickly demonstrate how the loss
    behave for different hyperparameter sets. Ultimately we need as much training data as possible.
    If we can speed up the pipeline, we can discard this reduction step.
    """
    if REDUCE_TRAIN:
        X_train = X_train[: int(np.ceil(TR_SAMPLE * X_train.shape[0])), :, :]
        y_train = Y_train[: int(np.ceil(TR_SAMPLE * Y_train.shape[0])), :]

    # MODEL TRAIN ------------------------------------

    # set seed for reproducibility
    tf.random.set_seed(MODEL_SEED)

    # test hyperparameter sets
    for n_gru in HP_N_GRU.domain.values:

        # print run config.
        hparams = {HP_N_GRU: n_gru}
        date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_run_name = f"vad-{n_gru}-gru-{date_time}"
        print("--- Starting trial: %s" % tb_run_name)
        print({h.name: hparams[h] for h in hparams})

        # config log directories
        scalars_log_dir = f"{RUN_DIR}{tb_run_name}/train"
        hparams_log_dir = f"{RUN_DIR}{tb_run_name}/hparams"
        test_log_dir = f"{RUN_DIR}{tb_run_name}/test"

        # log hyperparams
        writer = tf.summary.create_file_writer(hparams_log_dir)
        with writer.as_default():
            hp.hparams(hparams)
            hparams_callback = hp.KerasCallback(hparams_log_dir, hparams)

        # log training metrics
        writer = tf.summary.create_file_writer(scalars_log_dir)
        with writer.as_default():
            scalars_callback = TensorBoard(log_dir=scalars_log_dir, histogram_freq=1)

        # create the model architecture
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.GRU(4))
        model.add(tf.keras.layers.Dense(N_CLASSES, activation=OUT_ACTIVATION))

        # compile and train the model
        model.compile(loss=LOSS, optimizer=OPTIM, metrics=METRICS)
        model.fit(
            X_train,
            Y_train,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
            validation_split=VAL_FRAC,
            callbacks=[
                scalars_callback,  # log metrics
                hparams_callback,  # log hyperparams
            ],
        )

        # describe model
        model.summary()

        # INFERENCE ------------------------------------

        # calculate label probabilities
        predictions_proba = model.predict(X_test)

        # make label predictions
        preds = np.argmax(predictions_proba, axis=1).astype(int)

        # VALIDATION  ------------------------------------

        # plot predictions
        # Validation.plot_predictions(preds, Y_test[:,1], n_sample=100)

        # eval model
        perfs = Validation.eval_model(Y_test[:, 1], preds)

        # log metrics on test in tensorboard
        test_writer = tf.summary.create_file_writer(test_log_dir)
        with test_writer.as_default():
            tf.summary.scalar("test_accuracy", perfs["accuracy"], step=n_gru)
            tf.summary.scalar("test_precision", perfs["precision"], step=n_gru)
            tf.summary.scalar("test_recall", perfs["recall"], step=n_gru)
            tf.summary.scalar("test_f1_score", perfs["f1"], step=n_gru)

        # [TODO] log in tensorboard or mlflow
        # save model architecture
        tf.keras.utils.plot_model(
            model,
            to_file="report/model_architecture.png",
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
            layer_range=None,
        )

    # print duration
    print(time.time() - tic, "secs")
