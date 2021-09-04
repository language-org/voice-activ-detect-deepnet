# author: steeve LAQUITAINE
# purpose:
#   module that contains functions to train models
# usage:
#
#   from vad.pipelines.train.nodes import train_and_log

import time
from datetime import datetime
import numpy as np
import pandas as pd

# config
from kedro.config import ConfigLoader
from yaml.events import StreamEndEvent

# tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.ops.gen_data_flow_ops import BarrierReadySize
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import TimeDistributed, GRU, Dense
import tensorflow.keras.callbacks as C

# my custom package
from vad.pipelines.evaluate.nodes import Validation
import yaml

# tracking
import mlflow

# get run config
with open("config.yml") as conf:
    config = yaml.load(conf)
pipeline = config["run"]["env"]
conf_loader = ConfigLoader("conf/" + pipeline)
conf_tensorboard = conf_loader.get("globals*", "globals*/**")

# get tensorboard config
TB_DIR = conf_tensorboard["LOG_DIR"]

# ensure reproducibility
np.random.seed(0)


def train_and_log(
    train_audio,
    train_label,
    test_audio,
    test_label,
    params_train: dict,
    params_data_eng: dict,
):

    """Train the model and log the run's parameters to tensorboard and mlflow
    The model is trained via cross-validation and tested on a left-out test set
    of the audio data.

    Returns:
        [type]: the best trained model
    """
    tic = time.time()

    # get parameters
    MODEL = params_train["NAME"]
    MODEL_SEED = params_train["MODEL_SEED"]
    N_GRU = params_train["N_GRU"]
    OUT_ACTIVATION = params_train["OUT_ACTIVATION"]
    METRICS = params_train["METRICS"]
    LOSS = params_train["LOSS"]
    OPTIM = params_train["OPTIM"]
    EPOCH = params_train["EPOCH"]
    BATCH_SIZE = params_train["BATCH_SIZE"]
    VERBOSE = params_train["VERBOSE"]
    VAL_FRAC = params_train["VAL_FRAC"]
    BIAS = params_train["BIAS"]
    N_CLASSES = params_data_eng["N_CLASSES"]

    # set seed for reproducibility
    tf.random.set_seed(MODEL_SEED)

    # create hyperparameter sets
    HP_N_GRU = hp.HParam("n_gru", hp.Discrete(N_GRU))

    # loop over number of gru layers sets
    models_f1 = []
    models = []
    for ix, n_gru in enumerate(HP_N_GRU.domain.values):

        # track time
        t0 = time.time()

        # print config and log hyperparams & metrics
        tb_run, hparams = print_config(HP_N_GRU, n_gru)
        hp_clb = log_hparams_in_tb(log_dir=f"{TB_DIR}{tb_run}/hparams", hparams=hparams)
        scalar_clb = log_train_metrics_in_tb(log_dir=f"{TB_DIR}{tb_run}/train", freq=1)

        # calculate initial bias needed to compensate label imbalance
        bias = get_bias(train_label, BIAS)

        # create the model architecture
        model = init_model(
            MODEL, N_CLASSES=N_CLASSES, OUT_ACTIVATION=OUT_ACTIVATION, bias=bias
        )

        # compile
        model.compile(loss=eval(f"{LOSS}()"), optimizer=OPTIM, metrics=METRICS)

        # train the model
        model.fit(
            train_audio,
            train_label,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
            validation_split=VAL_FRAC,
            callbacks=[
                # checkpoint,  # monitor checkpoint
                scalar_clb,  # log metrics
                hp_clb,  # log hyperparams
            ],
        )

        # print model's description
        model.summary()

        # INFERENCE ------------------------------------
        test_predictions = test(model, test_audio)

        # evaluate the model on the left-out test set
        # case test_label is shaped as (s samples, t timesteps, 2 OHE labels)
        if test_label.ndim == 2:
            test_label = test_label[:, 1]
        elif test_label.ndim == 3:
            test_label = test_label[:, -1, 1]
        perfs = Validation.evaluate(test_predictions, test_label)

        # record models and their f1-score for comparison
        f1 = perfs["f1"].item()
        models_f1.append(f1)
        models.append(model)

        # log metrics on test in tensorboard
        log_dir = f"{TB_DIR}{tb_run}/test"
        log_metrics_in_tb(perfs, log_dir=log_dir, step=n_gru)

        # save model's architecture
        plot_model_architecture(model)

        # log pipeline's params in mlflow
        mlflow.log_param(key=f"tb_set_{ix}", value=f"{TB_DIR}{tb_run}")
        mlflow.log_param(key=f"duration_set_{ix}", value=time.time() - t0)

    # get best model
    ix_max = np.argmax(models_f1)
    best_model = models[ix_max]

    # print duration
    print(time.time() - tic, "secs")
    return best_model


def get_bias(train_label: np.ndarray, BIAS: bool) -> float:
    """Calculate the prior bias that must be applied to the classification layer's softmax output
     to compensate the class imbalance

    Args:
        train_label (np.ndarray):
            (s samples, t timesteps, 1 feature) size training label time series
            or (s samples, 1 feature) size training label time series
        BIAS (bool): [description]

    Returns:
        float: [description]
    """
    # initialize the bias
    bias = 0
    if BIAS:
        # case train_label is 2 dimensions: (s samples, 1 label)
        if train_label.ndim == 2:
            noise, speech = np.bincount(train_label[:, 1].astype(int))
            bias = np.log([speech / noise])
        # case train_label is 3 dimensions: (s samples, t timesteps, l=2 (OHE labels))
        if train_label.ndim == 3:
            noise, speech = np.bincount(train_label[:, -1, 1].astype(int))
            bias = np.log([speech / noise])
    return tf.keras.initializers.Constant(bias)


def test(model, test_audio: np.ndarray) -> np.ndarray:
    """Compute model predictions

    Args:
        model ([type]): a trained model
        test_audio (np.ndarray): audio signal to label

    Returns:
        (np.nd.array): predicted labels
    """
    # calculate label probabilities
    # make label predictions
    predictions_proba = model.predict(test_audio)

    # case prediction are shaped as (s samples, 2 OHE classes)
    if predictions_proba.ndim == 2:
        return np.argmax(predictions_proba, axis=1).astype(int)

    # case prediction are shaped as (s samples, timesteps, 2 OHE classes)
    elif predictions_proba.ndim == 3:
        return np.argmax(predictions_proba[:, -1, :], axis=1).astype(int)


def log_hparams_in_tb(log_dir: StreamEndEvent, hparams):
    """Log hyperparameters in tensorboard

    Args:
        log_dir (str): local logging path
        hparams ([type]): hyperparameters

    Returns:
        [type]: [description]
    """
    # write parameters locally to log_dir
    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
        hp_clb = hp.KerasCallback(log_dir, hparams)
    return hp_clb


def log_train_metrics_in_tb(log_dir, freq):
    """Log training and validation metrics in tensorboard

    Args:
        log_dir (str): local logging path
        freq ([type]): [description]

    Returns:
        [type]: [description]
    """
    # write training metrics locally to log_dir
    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
        scalar_clb = TensorBoard(log_dir=log_dir, histogram_freq=freq)
    return scalar_clb


def log_metrics_in_tb(perfs: pd.DataFrame, log_dir: str, step):
    """Log performance metrics in tensorboard

    Args:
        perfs (pd.DataFrame): [description]
        log_dir (str): [description]
        step ([type]): [description]
    """
    # convert to dict
    perfs = perfs.T.to_dict()[0]

    # log metrics on test in tensorboard
    test_writer = tf.summary.create_file_writer(log_dir)
    with test_writer.as_default():
        tf.summary.scalar("test_accuracy", perfs["accuracy"], step=step)
        tf.summary.scalar("test_precision", perfs["precision"], step=step)
        tf.summary.scalar("test_recall", perfs["recall"], step=step)
        tf.summary.scalar("test_f1_score", perfs["f1"], step=step)
        tf.summary.scalar("test_FRR", perfs["false_rejection_rate"], step=step)


def print_config(HP_N_GRU, n_gru):

    # print run configuration
    hparams = {HP_N_GRU: n_gru}
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_run = f"vad-{n_gru}-gru-{date_time}"
    print("--- Starting trial: %s" % tb_run)
    print({h.name: hparams[h] for h in hparams})
    return tb_run, hparams


def plot_model_architecture(model):
    """Save model architecture as .png file in report/

    Args:
        model (model): the trained model to plot and save
    """

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


def init_model(name: str, N_CLASSES: int, OUT_ACTIVATION: str, bias: float):
    """Instantiate the model to train

    Args:
        name (str): the model name:
            "BASIC" : basic model without VAD modification based on loss constrains
            "MIN_SPEECH": vad model with a loss penalized based on the number of
                minimum speech violations
        N_CLASSES (int): number of labels to predict (1:"speech" and 0:"non-speech")
        OUT_ACTIVATION (str): output layer's activation function, e.g.,  "softmax"
        bias (float32): bias to apply to the output layer to compensate class label imbalance

    Returns:
        [type]: [description]
    """
    # case the simplest model is selected
    if name == "BASIC":
        return VadNetModel(N_CLASSES, OUT_ACTIVATION, bias)

    # case the model constrained by minimum speech is selected
    if name == "MIN_SPEECH":
        return MinSpeechVadNetModel(N_CLASSES, OUT_ACTIVATION, bias)


def _check_exists_interval(y_pred):

    # get predictions
    y_pred = tf.cast(tf.greater(y_pred[0, :-1, 1], 0.5), tf.float32)

    # get starts and ends
    diff1 = tf.subtract(y_pred[:-1], y_pred[1:])
    starts = tf.where(diff1 == 1)
    ends = tf.where(diff1 == -1)

    # check if there is any interval
    is_exist_start = tf.equal(tf.size(starts), 0)
    is_exist_end = tf.equal(tf.size(ends), 0)
    return is_exist_start, is_exist_end


def count_violations(y_pred):

    # set minimum speech time threshold
    MIN_SPEECH = tf.constant(8, tf.int64)

    # get model predictions
    y_pred = tf.cast(tf.greater(y_pred[0, :-1, 1], 0.5), tf.float32)

    # get speech interval starts & ends
    diff1 = tf.subtract(y_pred[:-1], y_pred[1:])
    start_times = tf.where(diff1 == 1) + tf.constant(1, dtype=tf.int64)
    end_times = tf.where(diff1 == -1)
    speech_time = end_times - start_times

    # equate the loss penalty to the number of violations
    count_violations = tf.cast(tf.math.less(speech_time, MIN_SPEECH), tf.float32)
    penalty = tf.math.reduce_sum(count_violations)
    return tf.cast(penalty, tf.float32)


def get_penalty(y_pred):
    """Calculate the penalty for occurring minimum speech time violations.
    + 1 is added to the loss for every violation

    Args:
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """

    # check if speech interval exists
    is_exist_start, is_exist_end = _check_exists_interval(y_pred)

    # case it exists calculate penalty if violation
    penalty = tf.cond(
        tf.logical_and(is_exist_start, is_exist_end),
        lambda: count_violations(y_pred),
        lambda: tf.convert_to_tensor(0, tf.float32),
    )
    return penalty


class MinSpeechLoss(tf.keras.losses.Loss):
    """Loss penalized to enforce a minimum speech time"""

    def __init__(self):
        """Instantiate loss constrained to enforce a minimum speech time"""
        super().__init__()

    def call(self, y_true, y_pred):
        # replicate y_pred over timestep axis and calculate loss
        cce = CategoricalCrossentropy()
        cce_loss = cce(y_true, y_pred)
        cce_loss = tf.convert_to_tensor(cce_loss)

        # penalize minimum speech violations
        minspeech_loss = get_penalty(y_pred)
        return cce_loss + minspeech_loss

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class VadNetModel(tf.keras.Model):
    """Basic Voice activity detection model"""

    def __init__(self, N_CLASSES, OUT_ACTIVATION, bias):
        """Instantiate basic Voice activity detection model

        Args:
             N_CLASSES (int): number of labels to predict (1:"speech" and 0:"non-speech")
             OUT_ACTIVATION (str): output layer's activation function, e.g.,  "softmax"
             bias (float32): bias to apply to the output layer to compensate class label imbalance
        """
        super(VadNetModel, self).__init__()

        # instantiate 4 units GRU layer
        self.layer_1 = GRU(units=4)

        # instantiate classifier
        self.classifier = Dense(
            N_CLASSES, activation=OUT_ACTIVATION, bias_initializer=bias
        )

    def call(self, inputs):
        """Perform forward pass

        Args:
            inputs (..): model's input

        Returns:
            (..): model's output
        """
        x = self.layer_1(inputs)
        return self.classifier(x)


class MinSpeechVadNetModel(tf.keras.Model):
    """Voice activity detection model with a loss penalty each time minimum speech is violated"""

    def __init__(self, N_CLASSES, OUT_ACTIVATION, bias):
        """Instantiate Voice activity detection model constrained to enforce minimum speech
        time

        Args:
            N_CLASSES (int): number of labels to predict (1:"speech" and 0:"non-speech")
            OUT_ACTIVATION (str): output layer's activation function, e.g.,  "softmax"
            bias (float32): bias to apply to the output layer to compensate class label imbalance
        """
        super(MinSpeechVadNetModel, self).__init__()

        # instantiate 4 units GRU layer
        self.layer_1 = GRU(units=4, return_sequences=True)

        # instantiate classifier
        self.classifier = TimeDistributed(
            Dense(N_CLASSES, activation=OUT_ACTIVATION, bias_initializer=bias)
        )

    def call(self, inputs):
        """Perform forward pass

        Args:
            inputs (..): model's input

        Returns:
            (..): model's output
        """
        x = self.layer_1(inputs)
        return self.classifier(x)
