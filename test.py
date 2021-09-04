from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow as tf
import numpy as np

# create sequence (sample, timestep, feature)
length = 30  # audio sequence
seq = array([i / float(length) for i in range(length)])

n_sample = 3  # n sample in batch
n_timesteps = 10
n_feat = 1

# reshape X
X = seq.reshape(n_sample, n_timesteps, 1)

# simulate y (sample, timestep, binary)
y = np.random.random((n_sample, n_timesteps, 1)) > 0.5
y = tf.keras.utils.to_categorical(y, 2)

n_neurons = length
n_batch = 1
n_epoch = 10
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(n_timesteps, 1), return_sequences=True))
model.add(TimeDistributed(Dense(2, activation="softmax")))

output = model.predict(X)

model.compile(loss="categorical_crossentropy", optimizer="adam")
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
result = model.predict(X, batch_size=n_batch, verbose=0)



# print(tf.slice(deriv, begin=[1], size=[3]))
#     [
#         [[0, 1], [0, 0], [1, 0]],
#         [[1, 1], [1, 1], [1, 0]],
#         [[1, 1], [1, 1], [1, 0]],
#         [[0, 1], [0, 0], [1, 0]],
#     ],
#     dtype=tf.float16,
# )

from ipdb import set_trace

set_trace()

# def myloss():
#     cce = CategoricalCrossentropy()

#     # replicate y_pred over timestep axis and calculate loss
#     y_true = tf.numpy_function(repeat, [y_true, y_pred], tf.float32)
#     loss = cce(y_true, y_pred)

#     # penalize min speech violations
#     penalty = tf.numpy_function(get_penalty, [y_pred], tf.float32)
