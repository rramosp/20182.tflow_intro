import os
import time
import warnings
import numpy as np
from numpy import newaxis
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras import Sequential
from tensorflow.keras.backend import clear_session
from bokeh.plotting import figure, output_file, show
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def load_data(filename, seq_len, normalise_window, train_pct=.9):
#    f = open(filename, 'rb').read()
#    data = f.decode().split('\n')

    data = pd.read_csv(filename, names=["c1"])
    data = data.values[:,0]

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(train_pct * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers, dropout=.2):
    clear_session()
    
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        units=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(
        units=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_from_point(model, init_point, seq_len, nb_points):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = init_point
    predicted = []
    for i in range(nb_points):
        predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [seq_len-1], predicted[-1], axis=0)
    return np.r_[predicted]

def predict_sequences_every_n_points(model, data, seq_len, every_n_points):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/every_n_points)):
        curr_frame = data[i*every_n_points]
        predicted = []
        for j in range(every_n_points):
            predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [seq_len-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return np.r_[prediction_seqs]


class bplot:
    
    def __init__(self, width=800, height=400,x_axis_label="ticks", y_axis_label="signal",
                 title=""):
        self.p = figure(title=title, x_axis_label=x_axis_label, y_axis_label=y_axis_label, 
                        width=width, height=height)
    
    def line(self, x,y, **kwargs):
        x = x or range(len(y))
        self.p.line(x,y,**kwargs)
        return self

    def circle(self, x,y, **kwargs):
        x = x or range(len(y))
        self.p.circle(x,y,**kwargs)
        return self
    
    
    def show(self):
        show(self.p)