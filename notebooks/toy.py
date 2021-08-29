
#  author: steeve laquitaine
# purpose: 
#   quickly build a minimal working prototype that can be trained within 1 hour and make prediction on test data
# usage:
#
#   ```bash
#   python notebooks/toy.py
#   ````

import os
import scipy.io.wavfile
import pandas as pd
import numpy as np
import tensorflow as tf

# [TODO]: add to config
AUDIO_FILE = os.path.abspath("data/01_raw/vad_data/19-198-0003.wav")
LABEL_FILE = os.path.abspath("data/01_raw/vad_data/19-198-0003.json")

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
        sample_rate, audio = scipy.io.wavfile.read(AUDIO_FILE)  # File assumed to be in the same directory
        sample_size = len(audio)
        time_unit = 1/sample_rate
        duration_in_sec = time_unit * len(audio)      

        return {
            "data": audio.reshape(sample_size, 1), 
            "metadata": {
                "sample_rate": sample_rate,
                "sample_size": sample_size,
                "time_unit": time_unit,
                "duration_in_sec": duration_in_sec,                
                }
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
        sample_size =  audio["metadata"]["sample_size"]
        
        # create timestamps
        audio["metadata"]["timestamp"] = np.arange(0, len(data), 1)*time_unit

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
                    audio["metadata"]["timestamp"]>=speech_start, 
                    audio["metadata"]["timestamp"]<=speech_end)
            )
            synced_label[span] = 1
        return audio, synced_label

class DataEng:  

    @staticmethod      
    def split(X, Y, frac):

        # [TODO]: add test that X_train.shape + X_test.shape = X.shape
        # [TODO]: same for Y

        n_train = int(np.round(X.shape[0]*frac))
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
        for i in range(len(X)-timesteps-1):
            dataX.append(X[i:(i+timesteps), :])
        X = np.array(dataX)

        # drop Y's first n_look_back
        # the n_look_back Y is used for the first prediction
        # based on the n_look_back past X
        Y = Y[timesteps+1:]
        return X, Y


if __name__ == "__main__":

    # PARAMETERS --------------------------------------------
    # preprocessing
    TIMESTEPS = 1
    N_CLASSES = 2

    # mdoel
    SPLIT_FRAC = 0.5
    MODEL_SEED = 0
    OUT_ACTIVATION = "sigmoid"
    LOSS = "categorical_crossentropy" 
    OPTIM = "adam"
    EPOCH = 5
    BATCH_SIZE = 1
    VERBOSE = 2

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
    ts = ts.astype('float32')
    Y = Y.astype('float32')

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


    # MODEL TRAIN ------------------------------------

    # set seed before instantiation
    # for reproducibility
    tf.random.set_seed(MODEL_SEED)

    # design model architecture
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(4))
    model.add(tf.keras.layers.Dense(N_CLASSES, activation=OUT_ACTIVATION))
    
    # compile and train train
    model.compile(loss=LOSS, optimizer=OPTIM)
    model.fit(X_train, Y_train, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=VERBOSE)


    # INFERENCE ------------------------------------

    # calculate each label proba
    predictions_proba = model.predict(X_test)
    
    # make predictions
    preds = np.argmax(predictions_proba, axis=1).astype(int)
    print(predictions_proba)
    print(preds)
