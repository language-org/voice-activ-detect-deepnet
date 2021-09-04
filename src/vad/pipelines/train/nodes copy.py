# author: steeve LAQUITAINE

import numpy as np
import tensorflow as tf
from datetime import datetime
import pandas as pd

# from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import time

from tensorboard.plugins.hparams import api as hp
from tensorflow.python.ops.gen_data_flow_ops import BarrierReadySize
from vad.pipelines.evaluate.nodes import Validation
import yaml
from kedro.config import ConfigLoader
import mlflow
import keras

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import TimeDistributed, GRU, Dense

# Setup debugging
# tf.debugging.experimental.enable_dump_debug_info(
#     "tbruns/debug/", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1
# )

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

    models_f1 = []
    models = []

    # test hyperparameter sets
    for ix, n_gru in enumerate(HP_N_GRU.domain.values):

        t0 = time.time()

        # print run configuration
        hparams = {HP_N_GRU: n_gru}
        date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_run = f"vad-{n_gru}-gru-{date_time}"
        print("--- Starting trial: %s" % tb_run)
        print({h.name: hparams[h] for h in hparams})

        # log hyperparams & metrics
        hp_clb = log_hparams_in_tb(log_dir=f"{TB_DIR}{tb_run}/hparams", hparams=hparams)
        scalar_clb = log_train_metrics_in_tb(log_dir=f"{TB_DIR}{tb_run}/train", freq=1)

        # calculate initial bias needed to compensate label imbalance
        bias = get_bias(train_label, BIAS)

        # create the model architecture
        model = init_model(
            MODEL, n_classes=N_CLASSES, activation=OUT_ACTIVATION, bias=bias
        )

        output = model.predict(train_audio)
        print(output.shape)

        # compile
        model.compile(loss=eval(f"{LOSS}()"), optimizer=OPTIM, metrics=METRICS)

        from ipdb import set_trace

        set_trace()
        # train the model
        model.fit(
            train_audio,
            train_label,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
            validation_split=VAL_FRAC,
            callbacks=[
                scalar_clb,  # log metrics
                hp_clb,  # log hyperparams
            ],
        )

        # describe the model
        model.summary()

        # INFERENCE ------------------------------------
        test_predictions = test(model, test_audio)

        # eval model
        perfs = Validation.evaluate(test_predictions, test_label[:, 1])

        # record models and their f1-score for comparison
        f1 = perfs["f1"].item()
        models_f1.append(f1)
        models.append(model)

        # log metrics on test in tensorboard
        log_dir = f"{TB_DIR}{tb_run}/test"
        log_metrics_in_tb(perfs, log_dir=log_dir, step=n_gru)

        # [TODO] log in tensorboard or mlflow
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

        # log pipeline's params in mlflow
        mlflow.log_param(key=f"tb_set_{ix}", value=f"{TB_DIR}{tb_run}")
        mlflow.log_param(key=f"duration_set_{ix}", value=time.time() - t0)

    ix_max = np.argmax(models_f1)
    best_model = models[ix_max]
    print(time.time() - tic, "secs")
    return best_model


def get_bias(train_label, BIAS):
    bias = 0
    if BIAS:
        noise, speech = np.bincount(train_label[:, 1].astype(int))
        bias = np.log([speech / noise])
    return tf.keras.initializers.Constant(bias)


def test(model, test_audio):

    # calculate label probabilities
    # make label predictions
    predictions_proba = model.predict(test_audio)
    return np.argmax(predictions_proba, axis=1).astype(int)


def log_hparams_in_tb(log_dir, hparams):
    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
        hp_clb = hp.KerasCallback(log_dir, hparams)
    return hp_clb


def log_train_metrics_in_tb(log_dir, freq):
    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
        scalar_clb = TensorBoard(log_dir=log_dir, histogram_freq=freq)
    return scalar_clb


def log_metrics_in_tb(perfs: pd.DataFrame, log_dir, step):

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


def init_model(name, n_classes, activation, bias):
    if name == "BASIC":
        return VadNetModel(n_classes, activation, bias)
    if name == "MIN_SPEECH":
        return MinSpeechVadNetModel(n_classes, activation, bias)


def repeat(y_true, y_pred):
    return np.repeat(y_true[:, np.newaxis, :], y_pred.shape[1], axis=1)


def get_penalty(y_pred):
    first_deriv = np.diff(y_pred)
    start_time = np.where(first_deriv == 1)[0] + 1
    end_time = np.where(first_deriv == -1)[0]
    speech_time = end_time - start_time
    penalty = sum(speech_time < 3)
    return penalty


class MinSpeechLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        cce = CategoricalCrossentropy()

        # replicate y_pred over timestep axis and calculate loss
        y_true = tf.numpy_function(repeat, [y_true, y_pred], tf.float32)
        loss = cce(y_true, y_pred)

        # penalize min speech violations
        penalty = tf.numpy_function(get_penalty, [y_pred], tf.float32)
        return loss + penalty


class VadNetModel(tf.keras.Model):
    def __init__(self, N_CLASSES, OUT_ACTIVATION, bias):
        super(VadNetModel, self).__init__()
        self.layer_1 = GRU(units=4)
        self.classifier = Dense(
            N_CLASSES, activation=OUT_ACTIVATION, bias_initializer=bias
        )

    def call(self, inputs):
        x = self.layer_1(inputs)
        return self.classifier(x)


class MinSpeechVadNetModel(tf.keras.Model):
    def __init__(self, N_CLASSES, OUT_ACTIVATION, bias):
        super(MinSpeechVadNetModel, self).__init__()
        self.layer_1 = GRU(units=4, return_sequences=True)
        # self.classifier = TimeDistributed(
        #     Dense(N_CLASSES, activation=OUT_ACTIVATION, bias_initializer=bias)
        # )
        self.classifier = Dense(
            N_CLASSES, activation=OUT_ACTIVATION, bias_initializer=bias
        )

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.classifier(x)
        return x
