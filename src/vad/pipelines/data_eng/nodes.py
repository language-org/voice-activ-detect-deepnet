# author: steeve LAQUITAINE
# purpose:
#   module that contains functions to load and engineer the datasets for modeling and inference
# usage:
#
#   from vad.pipelines.data_eng.nodes import Etl

import os
import scipy.io.wavfile
import pandas as pd
import numpy as np
import tensorflow as tf
from kedro.config import ConfigLoader
import yaml
import random
from typing import Dict, Any

# load the run's configuration
with open("config.yml") as conf:
    config = yaml.load(conf)
pipeline = config["run"]["env"]
conf_loader = ConfigLoader("conf/" + pipeline)
globals = conf_loader.get("globals*", "globals*/**")
catalog = conf_loader.get("catalog*", "catalog*/**")
params = conf_loader.get("parameters*", "parameters*/**")

# get the audio and label file's path from config. for loading
AUDIO_FILE = os.path.abspath(globals["data_path"])
if params["DATA_ENG"]["LABEL"]:
    LABEL_FILE = os.path.abspath(globals["label_path"])


class Etl:
    """Loading pipeline"""

    @staticmethod
    def read_X() -> Dict[str, Any]:
        """Load a .wav audio, convert it to time series and
        calculate some audio properties (length..)

        Returns:
            (dict): loaded audio with main keys:
                "audio":
                    "data": audio time series
                    "metadata": audio properties such as "sample rate",..
        """
        # read audio file
        sample_rate, data = scipy.io.wavfile.read(AUDIO_FILE)

        # calculate some metadata
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
    def load_Y(params: Dict[str, Any]) -> pd.DataFrame:
        """Load labels from a .json file

        Args:
            params (Dict): dictionary of parameters containing
                the key "LABEL" associated with a boolean value
                    True:  load label
                    False: do not load label

        Returns:
            (pd.DataFrame): a dataframe of label interval start and end times
        """
        # case labels are required in downstream pipelines
        # read label file
        if params["LABEL"]:
            return pd.read_json(LABEL_FILE)
        else:
            return []

    @staticmethod
    def sync_audio_and_labels(
        audio: Dict[str, Any], label: pd.DataFrame
    ) -> Dict[str, Any]:
        """Synchronise an audio and its labels by creating two
        same length time series with a value per timestamp

        Args:
            data (Dict[str, Any]): contains to main keys:
                "audio":
                    "data": audio time series
                    "metadata": audio properties such as "sample rate",..
            label ([type]): [description]

        Returns:
            (Dict[str, Any]): data dictionary updated with a "label”
                containing the audio labels converted to a time series
        """
        # get audio data and its metadata
        data = audio["audio"]["data"]
        time_unit = audio["audio"]["metadata"]["time_unit"]
        sample_size = audio["audio"]["metadata"]["sample_size"]

        # create audio timestamps
        audio["audio"]["metadata"]["timestamp"] = np.arange(0, len(data), 1) * time_unit

        # case label exists
        # synchronize labels and audio
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
    def test_on_label(audio: Dict[str, Any], params: Dict[str, bool]) -> Dict[str, Any]:
        """Apply some sanity checks on the training and inference pipeline
        Args:
            params (dict): parameters with the following keys:
                "SHUFFLE_LABEL": labelled mapping to audio data is randomized
                "ALL_SPEECH": all data points are labelled as speech
                "NO_SPEECH": all data points are labelled as no speech
        Returns:
            (dict): modified audio labels
        """

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
    """Data engineering pipeline"""

    @staticmethod
    def split_train_test(
        data: Dict[str, Any],
        params: Dict[str, Any],
    ) -> np.ndarray:
        """Split the dataset in to train and test set for a simple cross-validation

        Args:
            data (Dict[str, Any]): contains to main keys:
                "audio":
                    "data": audio time series
                "label": time series of labels (0 and 1)
            params (Dict[str, Any]): [description]

        Returns:
            np.ndarray: [description]
        """
        # get parameters
        FRAC = params["SPLIT_FRAC"]

        # get data
        audio = data["audio"]["data"]
        label = data["label"]
        n_train = int(audio.shape[0] * FRAC)

        # split the datat set into a training and a test set
        train_audio = audio[:n_train, :]
        test_audio = audio[n_train:, :]
        train_label = []
        test_label = []
        if "label" in data:
            train_label = label[:n_train, :]
            test_label = label[n_train:, :]
        return train_audio, test_audio, train_label, test_label

    @staticmethod
    def set_resolution(
        synced: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Set the data resolution
        Args:
            synced
            params
        Returns:
            (Dict[str, Any]): audio with the set resolution
        """

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
    def _reshape_label_for_net(
        data, params_dataeng: dict, params_train: dict
    ) -> Dict[str, Any]:

        # get data
        label = data["label"]

        # get parameters
        TIMESTEPS = params_dataeng["TIMESTEPS"]
        N_CLASSES = params_dataeng["N_CLASSES"]

        # reshape as audio (s samples, t timesteps, 1 feature)
        # elif params_train["NAME"] == "MIN_SPEECH":
        labelY = []
        for i in range(len(label) - TIMESTEPS - 1):
            labelY.append(label[i : (i + TIMESTEPS), :])
        data["label"] = np.array(labelY)

        # case basic model
        # drop N first labels to match reshaped audio
        if params_train["NAME"] == "BASIC":
            data["label"] = label[TIMESTEPS + 1 :]

        # one hot encode
        data["label"] = tf.keras.utils.to_categorical(data["label"], N_CLASSES)
        return data

    @staticmethod
    def reshape_input_for_net(data, params_dataeng: dict, params_train: dict):
        data = DataEng._reshape_audio_for_net(data, params_dataeng)
        if "label" in data:
            data = DataEng._reshape_label_for_net(data, params_dataeng, params_train)
        return data

    @staticmethod
    def reduce_train(
        train_audio: np.ndarray, train_label: np.ndarray, params: dict
    ) -> np.ndarray:
        """Reduce the size of the training dataset

        Args:
            train_audio (np.ndarray): preprocessed audio
            train_label (np.ndarray): loaded labels
            params (dict): data engineering parameters (see parameters.yml)

        Returns:
            (np.ndarray): training audio and labels reduced in size
        """
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
