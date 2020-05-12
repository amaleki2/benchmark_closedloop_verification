import os


import tqdm
import pickle
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, RNN, SimpleRNN
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np
from util import ControllerDoublePendulum, ControllerTriplePendulum
from gym.envs.registration import make


# DO NOT USE GPU
NO_GPU = True
if NO_GPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

def read_data(ndata=1000, nviz=0):
    X_data = np.zeros((ndata, 51, 12))
    Y_data = np.zeros((ndata, 50, 6))
    for n in range(0, ndata):
        try:
            state = np.load("data_airplane/short_state_%d.npz" %(n+1))
            control = np.load("data_airplane/short_control_%d.npz" % (n+1))
        except:
            continue
        X_data[n, :, :] = state
        Y_data[n, :, :] = control

    if nviz > 0:
        x_goal = np.zeros(12); x_goal[0] = 25; x_goal[2] = 50
        env = make("AirPlane-v0", g=1., dt=0.1)
        for n in range(nviz):
            idx = np.random.randint(0, ndata)
            print("data %d" %idx)
            x_0 = X_data[idx, 0, :]
            env.x_0 = x_0
            env.env.x_0 = x_0
            env.reset()
            u = Y_data[idx, :, :]
            for k in range(u.shape[0]):
                env.step(u[k, :])
            env.render(x_goals=[x_goal])

    return X_data, Y_data


def model_train_ff(X_data, Y_data, savepath="trained_model.h5"):
    data_size, epoch_size, batch_size = None, 150, 32
    layer_dims, output_size = [100, 100, 20], Y_data.shape[1]

    model = Sequential()
    for dim in layer_dims:
        model.add(Dense(dim, activation='relu'))
    model.add(Dense(output_size, activation=None))

    checkpoint = ModelCheckpoint(savepath, verbose=1, save_best_only=True)  # , mode = 'max', monitor = 'val_acc')
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    callbacks = [checkpoint, ReduceLROnPlateau(patience=5, factor=0.5, verbose=True, min_lr=1E-4), PlotLosses()]

    model.fit(x=X_data, y=Y_data,
              batch_size=batch_size,
              epochs=epoch_size,
              verbose=1,
              callbacks=callbacks,
              validation_split=0.1, shuffle=True)

X_data, Y_data = read_data(ndata=1000, nviz=25)
print(X_data[-1, :])
print(X_data.shape)