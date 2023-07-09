from os.path import isfile

from keras.layers import Dense, LSTM
from keras.models import Sequential, load_model
import numpy as np
from pandas import DataFrame

TRAIN_SIZE = 0.8


def split_dataset(dataset, train_size=TRAIN_SIZE):
    index = int(len(dataset) * train_size)
    return dataset[:index], dataset[index:]


# return type: tuple(ndarray, ndarray)
def normalize_dataset(dataset: DataFrame, scaler):
    final_dataset = dataset.values
    train_data, _ = split_dataset(final_dataset)

    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data = [], []
    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i - 60 : i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
    x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    return x_train_data, y_train_data


# return type: Sequential
def train_lstm_model(x_train_data, y_train_data):
    # model_filename = "lstm_model.h5"

    # if isfile(model_filename):
        # return load_model(model_filename)

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    lstm_model.compile(loss="mean_squared_error", optimizer="adam")
    lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose="2")

    # lstm_model.save(model_filename)
    return lstm_model


def get_sample_dataset(dataset: DataFrame, scaler):
    _, valid_data = split_dataset(dataset.values)
    inputs = dataset[len(dataset) - len(valid_data) - 60 :].values
    inputs = inputs.reshape(-1, 1)
    return scaler.transform(inputs)


def predict_close_price(model, scaler, inputs):
    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60 : i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    return closing_price
