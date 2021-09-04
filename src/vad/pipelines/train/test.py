from tensorflow.keras.layers import Sequential, GRU, TimeDistributedDense, Activation

print("Build model...")
model = Sequential()
model.add(GRU(512, return_sequences=True))
model.add(GRU(512, return_sequences=True))
model.add(TimeDistributedDense(1))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
model.fit(x_pad, y_pad, batch_size=128, nb_epoch=2)
