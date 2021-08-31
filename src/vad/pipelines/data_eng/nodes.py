import os
import scipy.io.wavfile
import pandas as pd
import numpy as np
import tensorflow as tf

# from tensorflow import keras
from kedro.config import ConfigLoader
import yaml
from kedro.framework.session import get_current_session
import random

# get run config
with open("config.yml") as conf:
    config = yaml.load(conf)
pipeline = config["run"]["env"]
conf_loader = ConfigLoader("conf/" + pipeline)
globals = conf_loader.get("globals*", "globals*/**")
catalog = conf_loader.get("catalog*", "catalog*/**")
params = conf_loader.get("parameters*", "parameters*/**")

# get data path
AUDIO_FILE = os.path.abspath(globals["data_path"])
if params["DATA_ENG"]["LABEL"]:
    LABEL_FILE = os.path.abspath(globals["label_path"])


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
        sample_rate, data = scipy.io.wavfile.read(
            AUDIO_FILE
        )  # File assumed to be in the same directory
        sample_size = len(data)
        time_unit = 1 / sample_rate
        duration_in_sec = time_unit * len(data)
        timestamp = np.arange(0, len(data), 1) * time_unit

        return {
            "audio": {
                "data": data.reshape(sample_size, 1),
                "metadata": {
                    "sample_rate": sample_rate,
                    "sample_size": sample_size,
                    "time_unit": time_unit,
                    "duration_in_sec": duration_in_sec,
                    "timestamp": timestamp,
                },
            }
        }

    @staticmethod
    def load_Y(params):
        """Load Labels use create time series based on audio metadata
        use sampling rate
        calculate resolution in bits from labels start_time and end_time data

        Args:
            flag ([type]): [description]
        """
        if params["LABEL"]:
            return pd.read_json(LABEL_FILE)
        else:
            return []

    @staticmethod
    def sync_audio_and_labels(audio, label):
        # get data
        data = audio["audio"]["data"]
        time_unit = audio["audio"]["metadata"]["time_unit"]
        sample_size = audio["audio"]["metadata"]["sample_size"]

        # create timestamps
        audio["audio"]["metadata"]["timestamp"] = np.arange(0, len(data), 1) * time_unit

        # case label exists
        # synchronize labels with audio
        if len(label) is not 0:
            synced_label = np.zeros((sample_size, 1))
            array = label.values
            for ix in range(array.shape[0]):
                interval = array[ix][0]
                speech_start = interval["start_time"]
                speech_end = interval["end_time"]
                span = np.where(
                    np.logical_and(
                        audio["audio"]["metadata"]["timestamp"] >= speech_start,
                        audio["audio"]["metadata"]["timestamp"] <= speech_end,
                    )
                )
                synced_label[span] = 1
            audio["label"] = synced_label
        return audio

    @staticmethod
    def test_on_label(audio, params):
        if "label" in audio:
            sample_size = len(audio["label"])
            if params["SHUFFLE_LABEL"]:
                random.seed(0)
                random.shuffle(audio["label"])
            if params["ALL_SPEECH"]:
                audio["label"] = np.ones((sample_size, 1))
            if params["NO_SPEECH"]:
                audio["label"] = np.zeros((sample_size, 1))
        return audio


class DataEng:
    @staticmethod
    def split_train_test(data, params):

        # [TODO]: test that X_train.shape + X_test.shape = X.shape
        # [TODO]: test same for Y

        # get parameters
        FRAC = params["SPLIT_FRAC"]

        # get data
        audio = data["audio"]["data"]
        label = data["label"]
        n_train = int(audio.shape[0] * FRAC)

        # split
        train_audio = audio[:n_train, :]
        test_audio = audio[n_train:, :]
        train_label = []
        test_label = []
        if "label" in data:
            train_label = label[:n_train, :]
            test_label = label[n_train:, :]
        return train_audio, test_audio, train_label, test_label

    @staticmethod
    def set_resolution(synced, params):
        # get params
        RESOLUTION = params["RESOLUTION"]
        synced["audio"]["data"] = synced["audio"]["data"].astype("float32")
        if "label" in synced:
            synced["label"] = synced["label"].astype(RESOLUTION)
        return synced

    @staticmethod
    def _reshape_audio_for_net(data, params: dict):
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
        # get data
        audio = data["audio"]["data"]
        timestamp = data["audio"]["metadata"]["timestamp"]

        # get params
        TIMESTEPS = params["TIMESTEPS"]

        # reshape X
        dataX = []
        for i in range(len(audio) - TIMESTEPS - 1):
            dataX.append(audio[i : (i + TIMESTEPS), :])
        data["audio"]["data"] = np.array(dataX)
        data["audio"]["metadata"]["timestamp"] = timestamp[TIMESTEPS + 1 :]
        return data

    @staticmethod
    def _reshape_label_for_net(data, params: dict):

        # get data
        label = data["label"]

        # get params
        TIMESTEPS = params["TIMESTEPS"]
        N_CLASSES = params["N_CLASSES"]

        # drop initial n labels whic corresponds to the
        # initial audio data used for prediction
        data["label"] = label[TIMESTEPS + 1 :]
        data["label"] = tf.keras.utils.to_categorical(data["label"], N_CLASSES)
        return data

    @staticmethod
    def reshape_input_for_net(data, params: dict):
        data = DataEng._reshape_audio_for_net(data, params)
        if "label" in data:
            data = DataEng._reshape_label_for_net(data, params)
        return data

    @staticmethod
    def reduce_train(train_audio, train_label, params: dict):

        # get parameters
        REDUCE = params["REDUCE_TRAIN"]
        TR_SAMPLE = params["TR_SAMPLE"]

        if REDUCE:
            train_audio = train_audio[
                : int(np.ceil(TR_SAMPLE * train_audio.shape[0])), :, :
            ]
            train_label = train_label[
                : int(np.ceil(TR_SAMPLE * train_label.shape[0])), :
            ]
        return train_audio, train_label
