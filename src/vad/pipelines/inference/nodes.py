import numpy as np
import pandas as pd


def predict(model, prod_audio):
    prediction_proba = model.predict(prod_audio["audio"]["data"])
    prediction = np.argmax(prediction_proba, axis=1).astype(int)
    timestamp = prod_audio["audio"]["metadata"]["timestamp"]
    return pd.DataFrame(data={"timestamp": timestamp, "prediction": prediction})
