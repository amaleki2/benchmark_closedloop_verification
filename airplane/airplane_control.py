import os
import tqdm
import json
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Flatten, RNN, SimpleRNN, LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import adam

# DO NOT USE GPU
NO_GPU = True
if NO_GPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        #clear_output(wait=True)
        if epoch % 2 == 0:
            plt.plot(self.x, self.losses, label="loss", color="red")
            plt.plot(self.x, self.val_losses, label="val_loss", color="blue")
            plt.gca().set_yscale('log')
            if epoch == 0:
                plt.legend()
            plt.draw()
            plt.pause(0.01)


def trailing_lqr(last_x, dt, last_action):
    env = make("AirPlane-v0", dt=dt, x_0=last_x, g=1.)
    env.reset()
    env.env.reset()
    control = ControllerAirPlane(env)
    lqr_actions = control.run_lqr(Q, R, x_goal, n_step_lqr, last_action, goal_action=u_goal)
    return np.array(lqr_actions)


def trailing_ilqr(last_x, dt):
    env = make("AirPlane-v0", dt=dt, x_0=last_x, g=1.)
    env.reset()
    env.env.reset()
    control = ControllerAirPlane(env)
    ilqr_actions = control.run_ilqr(Q, R, Qf, last_x, 25)
    return np.array(ilqr_actions)


def read_trajopt_data():
    itr = 0
    for i in tqdm.trange(10000):
        if os.path.exists("data_airplane/julia2/state_%d.npz" %i):
            state = np.load("data_airplane/julia2/state_%d.npz" %i)
            control = np.load("data_airplane/julia2/control_%d.npz" %i)
            div_by = 2
        elif os.path.exists("data_airplane/julia2/long_state_%d.npz" %i):
            state = np.load("data_airplane/julia2/long_state_%d.npz" % i)
            control = np.load("data_airplane/julia2/long_control_%d.npz" % i)
            div_by = 1
        elif os.path.exists("data_airplane/julia2/short_state_%d.npz" %i):
            state = np.load("data_airplane/julia2/short_state_%d.npz" % i)
            control = np.load("data_airplane/julia2/short_control_%d.npz" % i)
            div_by = 4
        else:
            continue

        # last_x = state[-1, :]
        # last_action = control[-1, :]
        # lqr_actions = trailing_lqr(last_x, 0.1, last_action)
        # #ilqr_actions = trailing_ilqr(last_x, 0.1)
        # t_interp = np.linspace(0, 99, 100)
        # t = np.linspace(0, 98, control.shape[0])
        # control_interp = np.zeros((100, 6))
        # for k in range(6):
        #     control_interp[:, k] = np.interp(t_interp, t, control[:, k])

        env = make("AirPlane-v0", dt=0.05, x_0=state[0, :], g=1.)
        env.reset()
        for j in range(100):
            env.step(control[j // div_by, :])
            if j < 100:
                env.write_message("trajopt")
            else:
                env.write_message("lqr")

        env.save_history(name="data_airplane/python_data/history_%d.pkl" %itr)
        if i % 2000 == 0:
            env.render(x_goals=[x_goal])
        itr += 1


def read_history_data():
    itr = 0
    x = np.zeros((10000, 101, 12))
    T = np.zeros((10000, 101, 6))
    for k in range(9066):
        with open("data_airplane/python_data/history_%d.pkl" % k, "rb") as f:
            data_vec = pickle.load(f)
        x_last, T_last = data_vec[-1]["x"], data_vec[-1]["a"]
        if np.linalg.norm(x_last - x_goal) > 0.05:
            print("step %d did not approve" %k)
            continue
        else:
            itr += 1

        for i, data in enumerate(data_vec):
            x[k, i, :] = data["x"]
            T[k, i, :] = data["a"]
    x = x[:itr, :, :]
    T = T[:itr, :, :]
    return x, T


def nn_model(rnn=False, layer_rnn_dim=20, layer_dims=[100, 100, 20], output_size=6, input_shape=None, stateful=False):

    model = Sequential()
    if rnn:
        if input_shape is None:
            model.add(SimpleRNN(layer_rnn_dim, activation='tanh', stateful=stateful,
                                return_sequences=True))
            # model.add(SimpleRNN(layer_rnn_dim*3, activation='tanh', stateful=False,
            #                     return_sequences=True))
        else:
            model.add(SimpleRNN(layer_rnn_dim, activation='tanh', stateful=stateful,
                                return_sequences=True, input_shape=input_shape))
            # model.add(SimpleRNN(layer_rnn_dim*3, activation='tanh', stateful=False,
            #                     return_sequences=True, input_shape=input_shape))
    for dim in layer_dims:
        model.add(Dense(dim, activation='relu'))
    model.add(Dense(output_size, activation=None))
    return model

def train_model_ff(savepath="trained_model.h5"):

    x, T = read_history_data()
    X_data = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
    Y_data = T.reshape(T.shape[0]*T.shape[1], T.shape[2])

    data_size, epoch_size, batch_size = None, 50, 16
    layer_dims, output_size = [100, 100, 20], Y_data.shape[-1]

    model = Sequential()
    # model.add(SimpleRNN(10, activation='tanh', stateful=False,
    #                     return_sequences=True))
    for dim in layer_dims:
        model.add(Dense(dim, activation='relu'))
    model.add(Dense(output_size, activation=None))

    checkpoint = ModelCheckpoint(savepath, verbose=1, save_best_only=True)  # , mode = 'max', monitor = 'val_acc')

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d--%H-%M-")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    callbacks = [checkpoint,
                 ReduceLROnPlateau(patience=5, factor=0.5, verbose=True, min_lr=1E-4),
                 PlotLosses(),
                 tensorboard_callback, ]



    model.fit(x=X_data, y=Y_data,
              batch_size=batch_size,
              epochs=epoch_size,
              verbose=1,
              callbacks=callbacks,
              validation_split=0.1, shuffle=True)

def train_model_rnn(savepath="trained_model_rnn.h5"):
    os.system("rm -r logs")
    X_data, Y_data = read_history_data()
    print(X_data[0,0,:])
    print(Y_data[0,0,:])
    # time dimension = 101. problem with test time. what is x at future time
    # Y_data = Y_data.reshape(Y_data.shape[0], Y_data.shape[1]*Y_data.shape[2])

    # time dimension = 1
    #X_data = X_data.reshape(X_data.shape[0] * X_data.shape[1], 1, X_data.shape[2])
    #Y_data = Y_data.reshape(Y_data.shape[0] * Y_data.shape[1],    Y_data.shape[2])

    epoch_size, batch_size = 100, 50
    output_size = Y_data.shape[-1]
    # model = Sequential()
    #
    # # model.add(SimpleRNN(layer_rnn_dim, activation='relu', stateful=False,
    # #                     return_sequences=True, return_state=True))
    # model.add(SimpleRNN(layer_rnn_dim, activation='tanh', stateful=False,
    #                     return_sequences=True, return_state=True))
    # for dim in layer_dims:
    #     model.add(Dense(dim, activation='relu'))
    # model.add(Dense(output_size, activation=None))

    model = nn_model(rnn=True, layer_rnn_dim=rnn_dim, layer_dims=ff_dims, output_size=output_size, input_shape=None)

    opt = adam(clipnorm=5.0)

    checkpoint = ModelCheckpoint(savepath, verbose=1, save_best_only=True)  # , mode = 'max', monitor = 'val_acc')

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    callbacks = [checkpoint,
                 ReduceLROnPlateau(patience=5, factor=0.5, verbose=True, min_lr=1E-4),
                 PlotLosses(),
                 tensorboard_callback,]

    model.fit(x=X_data, y=Y_data,
              batch_size=batch_size,
              epochs=epoch_size,
              verbose=1,
              callbacks=callbacks,
              validation_split=0.1, shuffle=True)

def run_model():
    x_0 = [0., 0., 0., 0., 0., 5., .5, .5, 0., 0., 0., 0.]
    env = make("AirPlane-v0", dt=0.05, x_0=x_0, g=1.)
    env.reset()

    model = load_model('trained_model.h5')
    print(env.x)
    for i in range(100):
        nn_in = np.atleast_2d(env.x)
        nn_out = model.predict(nn_in)
        actions = nn_out[0]
        env.step(actions)

    print(env.x)
    env.render(x_goals=[x_goal])
    env.animate(x_goals=[x_goal], file_name="airplane_ff.gif")

def run_model_rnn(method):
    x_0 = np.array([0., 0., 0., 2., 0., 2., 0.5, 0.5, 0., 0., 0., 0.])
    env = make("AirPlane-v0", dt=0.05, x_0=x_0, g=1.)
    env.reset()
    layer_rnn_dim = 100
    model_trained = load_model('trained_model_rnn.h5')

    if method == 1:
        # model_test = Sequential()
        # model_test.add(SimpleRNN(layer_rnn_dim, activation='relu', stateful=False,
        #                          return_sequences=True, input_shape=(1, 12)))
        # for dim in [150, 100, 30]:
        #     model_test.add(Dense(dim, activation='relu'))
        # model_test.add(Dense(6, activation=None))
        #
        # assert len(model_trained.layers) == len(model_test.layers)
        # for j in range(len(model_trained.layers)):
        #     model_test.layers[j].set_weights(model_trained.layers[j].get_weights())
        trained_model = load_model('trained_model_rnn.h5')
        trained_model_json = trained_model.to_json()
        test_model_dict = json.loads(trained_model_json)
        test_model_dict['config']['layers'][0]['config']['batch_input_shape'] = [1, 1, 12]

        for l in test_model_dict['config']['layers']:
            if 'stateful' in l['config']:
                l['config']['stateful'] = True


        model_test = model_from_json(json.dumps(test_model_dict))
        model_test.set_weights(trained_model.get_weights())
        # model_test = nn_model(rnn=True, layer_rnn_dim=rnn_dim, layer_dims=ff_dims, output_size=6,
        #                  input_shape=(1, 12), stateful=True)

        for i in range(100):
            nn_in = env.x.reshape(1, 1, 12)
            nn_out = model_test.predict(nn_in)
            actions = nn_out[0][-1]
            print(i, "  ", actions)
            env.step(actions)

    else:
        for i in range(100):
            model_test = Sequential()
            model_test.add(SimpleRNN(layer_rnn_dim, activation='relu', stateful=False,
                                     return_sequences=True, input_shape=(i+1, 12)))
            for dim in [150, 100, 30]:
                model_test.add(Dense(dim, activation='relu'))
            model_test.add(Dense(6, activation=None))

            assert len(model_trained.layers) == len(model_test.layers)
            for j in range(len(model_trained.layers)):
                model_test.layers[j].set_weights(model_trained.layers[j].get_weights())
            nn_in = np.array([h["x"] for h in env.history]).reshape(1, i+1, 12)
            nn_out = model_test.predict(nn_in)
            actions = nn_out[0][-1]
            print(i, "  ", actions)
            env.step(actions)

    env.render(x_goals=[x_goal])
    env.animate(x_goals=[x_goal], file_name="airplane_rnn_%d.gif" %method, dpi=90)

if __name__ == "__main__":
    n_step_lqr = 50
    Q = np.eye(12, 12) * 10
    Qf = np.eye(12, 12) * 1000
    R = np.eye(6, 6) * 10
    x_goal = [25., 0., 50., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    u_goal = [0., 0., -1., 0., 0., 0.]
    rnn_dim = 100
    ff_dims = [150, 50, 12]
    #read_trajopt_data()
    # train_model_ff()
    # run_model()
    # train_model_rnn()
    run_model_rnn(1)
# ilqr_actions_1 = control.run_ilqr(Q, R, Qf, x_med_1, n_step_ilqr)
# #lqr_actions_1 = control.run_lqr(Q, R, x_med_1, n_step_lqr, ilqr_actions_1[-1])
#
# # control.env.env.x_0 = control.env.x
# # control.env.env.reset()
# # control.env.reset()
# #ilqr_actions_2 = control.run_ilqr(Q, R, Qf, x_goal, n_step_ilqr)
# lqr_actions_3 = control.run_lqr(Q, R, x_goal, n_step_lqr, ilqr_actions_1[-1])
#
# #control.env.env.x_0 = control.env.x
# #lqr_actions = control.run_lqr(Q, R, x_goal, n_step_lqr, ilqr_actions_2[-1])
#
# control.env.env.x_0 = x_0
# control.env.reset()
# control.env.env.reset()
# for action in ilqr_actions_1  + lqr_actions_3 :
#     control.env.step(action)
# env.render(skip=3, x_goals=[x_med_1, x_goal])
