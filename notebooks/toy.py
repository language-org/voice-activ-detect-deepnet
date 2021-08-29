
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



if __name__ == "__main__":

    # RAW --------------------------------------------

    # read and sync audio and labels
    audio = Etl.read_X()
    label = Etl.load_Y()
    audio, label = Etl.sync_audio_and_labels(audio, label)
    ts = audio["data"]
    Y = label
