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
from util import ControllerDoublePendulum, ControllerTriplePendulum, ControllerSinglePendulum
from gym.envs.registration import make


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

class BehaviorClonning:
    def __init__(self, env, controller_class, n_data, folder_name="data2/", **kwargs):
        self.env = env
        self.controller = controller_class(self.env)
        self.n_data = n_data
        self.folder_name = folder_name

        # default values:
        n_step_ilqr = 150
        n_step_lqr = 250
        Q = np.eye(self.env.n_state, self.env.n_state)
        Qf = np.eye(self.env.n_state, self.env.n_state) * 1000
        R = np.eye(self.env.n_state // 2, self.env.n_state // 2)
        x_range = [(0, 2 * np.pi)] * (self.env.n_state // 2) + [(-4 * np.pi, 4 * np.pi)] * (self.env.n_state // 2)
        x_goal = [0.] * self.env.n_state

        self.n_step_ilqr = n_step_ilqr
        self.n_step_lqr = n_step_lqr
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.x_range = x_range
        self.x_goal = x_goal

        # unpacking kwargs
        for (name, value) in kwargs.items():
            setattr(self, name, value)

    def generate_data(self):
        for i in range(self.n_data):
            x_0 = [0.] *len(self.x_range)
            for k in range(len(self.x_range)):
                x_k_min, x_k_max = self.x_range[k]
                x_0[k] = np.random.random()*(x_k_max - x_k_min) + x_k_min

            self.env.env.x_0 = x_0
            self.env.reset()

            ilqr_actions = self.controller.run_ilqr(self.Q, self.R, self.Qf, self.x_goal, self.n_step_ilqr)
            _            = self.controller.run_lqr(self.Q, self.R, self.x_goal, self.n_step_lqr, ilqr_actions[-1])

            if i < 3:
                self.env.render(skip=1)
            self.env.save_history(name=self.folder_name + "history_%d.pkl" %i)

    def make_data_lqr(self):
        for k in range(self.n_data):
            with open(self.folder_name + "history_%d.pkl" % k, "rb") as f:
                data_vec = pickle.load(f)
            x = np.zeros((self.n_step_lqr, self.Q.shape[0]))
            T = np.zeros((self.n_step_lqr, self.R.shape[0]))
            for i, data in enumerate(data_vec):
                if i > self.n_step_ilqr:
                    x[i - self.n_step_ilqr - 1, :] = data["x"]
                    T[i - self.n_step_ilqr - 1, :] = data["a"]

            if np.linalg.norm(x[-1, :] - self.x_goal) + np.linalg.norm(T[-1, :]) > 0.05:
                continue

            if 'X_data' not in locals():
                X_data = x.copy()
                Y_data = T.copy()
            else:
                X_data = np.concatenate((X_data, x))
                Y_data = np.concatenate((Y_data, T))
        return X_data, Y_data

    def make_data_ilqr(self):
        for k in range(self.n_data):
            with open(self.folder_name + "history_%d.pkl" % k, "rb") as f:
                data_vec = pickle.load(f)
            x = np.zeros((self.n_step_ilqr+1, self.Q.shape[0]))
            T = np.zeros((self.n_step_ilqr+1, self.R.shape[0]))
            for i, data in enumerate(data_vec):
                if i <= self.n_step_ilqr:
                    x[i, :] = data["x"]
                    T[i, :] = data["a"]

            x_last, T_last = data_vec[-1]["x"], data_vec[-1]["a"]
            if np.linalg.norm(x_last - self.x_goal) + np.linalg.norm(T_last) > 0.05:
                continue

            if 'X_data' not in locals():
                X_data = x.copy()
                Y_data = T.copy()
            else:
                X_data = np.concatenate((X_data, x))
                Y_data = np.concatenate((Y_data, T))
        return X_data, Y_data

    def make_data(self, plot_data=False):
        X_data1, Y_data1 = self.make_data_ilqr()
        X_data2, Y_data2 = self.make_data_lqr()
        X_data = np.concatenate((X_data1, X_data2))
        Y_data = np.concatenate((Y_data1, Y_data2))

        if plot_data:
            plt.figure(figsize=(10, 8))
            plt.plot(X_data[:, 0], X_data[:, 1], marker=".", linestyle="none")
            plt.xlabel("x_0")
            plt.ylabel("y_0")
            plt.show()

            plt.figure(figsize=(10, 8))
            plt.plot(Y_data[:, 0], Y_data[:, 1], marker=".", linestyle="none")
            plt.xlabel("T_1")
            plt.xlabel("T_2")
            plt.show()

        return X_data, Y_data

    def model_train_ff(self, X_data, Y_data, savepath="trained_model.h5"):
        data_size, epoch_size, batch_size = None, 150, 32
        #layer_dims, output_size = [100, 100, 20], Y_data.shape[1]
        layer_dims, output_size = [50, 10], Y_data.shape[1]
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

    def run_model(self, x_0, n_max):
        env.env.x_0 = x_0
        env.reset()

        model_ilqr = load_model('model_ilqr.h5')
        model_lqr = load_model('model_lqr.h5')
        model = model_ilqr
        print(self.env.x)
        msg = "using ilqr model"
        for i in range(n_max):
            state_norms = np.linalg.norm(self.env.x)
            if state_norms < 0.025:
                msg = "stabilized"
                break
            elif (model == model_ilqr) & (state_norms < 1.0):
                msg = "using lqr model"
                model = model_lqr
            elif (model == model_lqr) & (state_norms > 5.0):
                msg = "using ilqr"
                model = model_ilqr
            nn_in = np.atleast_2d(self.env.x)
            nn_out = model.predict(nn_in)
            torques = nn_out[0]
            self.env.step(torques)
            self.env.write_message(msg)

        self.env.render()


class BehaviorClonning3(BehaviorClonning):
    def generate_data(self):
        for i in range(self.n_data):
            if os.path.isfile(self.folder_name + "history_%d.pkl" % i):
                continue


            x_0 = [0.] * len(self.x_range)
            for k in range(len(self.x_range)):
                x_k_min, x_k_max = self.x_range[k]
                x_0[k] = np.random.random()*(x_k_max - x_k_min) + x_k_min

            x_0 = np.round(x_0, decimals=3)
            print(x_0)
            self.env.env.x_0 = x_0
            self.env.reset()
            self.env.env.reset()

            x_mid_1 = [1.5, 1., 1.5, 1., 1., 1.]
            x_mid_2 = [0.5, 0.5, 0.5, 0.0, 0.0, -0.]

            assert self.n_step_ilqr%3 == 0
            n_step_ilqr_div_3 = self.n_step_ilqr // 3

            u_init = np.zeros((3, n_step_ilqr_div_3))
            u_init[0, :n_step_ilqr_div_3] = np.linspace(1, 0, n_step_ilqr_div_3)

            ilqr_actions = []
            #if get_norm(self.env.x[:3], x_mid_2[:3]) < get_norm(self.env.x[:3], self.x_goal[:3]):

            ilqr_actions_1 = self.controller.run_ilqr(self.Q, self.R, self.Qf, x_mid_1, n_step_ilqr_div_3, u_init=u_init)
            ilqr_actions += ilqr_actions_1

            #if get_norm(self.env.x, x_mid_2) < get_norm(self.env.x, self.x_goal):
            self.env.env.x_0 = self.env.env.x
            self.env.reset()
            self.env.env.reset()
            ilqr_actions_2 = self.controller.run_ilqr(self.Q, self.R, self.Qf, x_mid_2, n_step_ilqr_div_3)
            ilqr_actions += ilqr_actions_2

            self.env.env.x_0 = self.env.env.x
            self.env.reset()
            self.env.env.reset()
            ilqr_actions_3 = self.controller.run_ilqr(self.Q, self.R, self.Qf, self.x_goal, n_step_ilqr_div_3)
            ilqr_actions += ilqr_actions_3

            lqr_actions = self.controller.run_lqr(self.Q, self.R, self.x_goal, self.n_step_lqr, ilqr_actions[-1])
            self.env.env.x_0 = x_0
            self.env.reset()
            self.env.env.reset()
            all_actions = ilqr_actions + lqr_actions
            for jj in range(len(all_actions)):
                self.env.x, _, _, _ = self.env.step(all_actions[jj])

            #self.env.render(skip=1)
            self.env.save_history(name=self.folder_name + "history_%d.pkl" % i)

    def run_model(self, x_0, n_max):
        env.env.x_0 = x_0
        env.reset()

        model_ilqr = load_model('model_ilqr_pend3.h5')
        model_lqr = load_model('model_lqr_pend3.h5')
        model = model_ilqr
        print(self.env.x)
        msg = "using ilqr model"
        for i in range(n_max):
            state_norms = np.linalg.norm(self.env.x)
            if state_norms < 0.025:
                msg = "stabilized"
                #break
            elif (model == model_ilqr) & (state_norms < 0.5):
                msg = "using lqr model"
                model = model_lqr
                print(i, msg)
            elif (model == model_lqr) & (state_norms > 2.0):
                msg = "using ilqr"
                model = model_ilqr
                print(i, msg)
            nn_in = np.atleast_2d(self.env.x)
            nn_out = model.predict(nn_in)
            torques = nn_out[0]
            self.env.step(torques)
            self.env.write_message(msg)


        self.env.render()

    def run_model_2(self, x_0, n_max=1000, x_mid_1 = [1.5, 1., 1.5, 1., 1., 1.], x_mid_2 = [0.5, 0.5, 0.5, 0.0, 0.0, -0.]):
        env.env.x_0 = x_0
        env.reset()

        model_ilqr_0 = load_model('model_ilqr_0_pend3.h5')
        model_ilqr_1 = load_model('model_ilqr_1_pend3.h5')
        model_ilqr_2 = load_model('model_ilqr_2_pend3.h5')
        model_lqr = load_model('model_lqr_pend3.h5')
        model = model_ilqr_0
        print(self.env.x)
        msg = "using ilqr_0 model "
        for i in range(n_max):
            dist_to_goal = np.linalg.norm(self.env.x)
            dist_to_mid_1 = np.linalg.norm(self.env.x - x_mid_1)
            dist_to_mid_2 = np.linalg.norm(self.env.x - x_mid_2)
            if dist_to_goal < 0.025:
                msg = "stabilized"
                print(i, msg)
                break
            elif (model in [model_ilqr_0, model_ilqr_1, model_ilqr_2]) & (dist_to_goal < 0.5):
                msg = "using lqr model"
                model = model_lqr

            if (dist_to_mid_1 < dist_to_mid_2) & (i > 50):
                msg = "using ilqr_1 model"
                model = model_ilqr_1

            if (dist_to_mid_1 > dist_to_mid_2) & (i > 50):
                msg = "using ilqr_2 model"
                model = model_ilqr_2

            nn_in = np.atleast_2d(self.env.x)
            nn_out = model.predict(nn_in)
            torques = nn_out[0]
            self.env.step(torques)
            self.env.write_message(msg)

        self.env.render()

if __name__ == '__main__':

    # # make the environment
    # env = make("Pendulum1-v0", dt=0.1)
    # env.reset()
    #
    # # call the class
    # bh = BehaviorClonning(env, ControllerSinglePendulum, 200, folder_name="data_single_pend_g1_c02_m1_L1/",
    #                       n_step_ilqr=25, n_step_lqr=50)
    #
    # generate_data, train = False, False
    # if generate_data:
    #     # generate data.
    #     bh.generate_data()
    #
    # if train:
    #     # setup training data for ilqr model
    #     X_data_ilqr, Y_data_ilqr = bh.make_data_ilqr()
    #     # train ilqr network
    #     bh.model_train_ff(X_data_ilqr, Y_data_ilqr, "model_ilqr.h5")
    #
    #     #setup training data for lqr model
    #     X_data_lqr, Y_data_lqr = bh.make_data_lqr()
    #     # train lqr network
    #     bh.model_train_ff(X_data_lqr, Y_data_lqr, "model_lqr.h5")
    #
    # # choose a random initial state
    # ns = bh.env.n_state
    # x_0 = [1., 0.5]
    # # x_0 = np.random.random(ns)
    # # x_0[0:ns//2] *= 2 * np.pi
    # # x_0[ns//2:] *= 2 * np.pi
    # # x_0[ns//2:] -= 2 * np.pi
    # #
    # print(x_0)
    #
    # # run the model.
    # bh.run_model(x_0, 1500)

    # # make the environment
    env = make("Pendulum2-v0", dt=0.1)
    env.reset()

    # call the class
    bh = BehaviorClonning(env, ControllerDoublePendulum, 200, folder_name="data2/",
                          n_step_ilqr=150, n_step_lqr=250)

    generate_data, train = False, False
    if generate_data:
        # generate data.
        bh.generate_data()

    if train:
        # setup training data for ilqr model
        X_data_ilqr, Y_data_ilqr = bh.make_data_ilqr()
        # train ilqr network
        bh.model_train_ff(X_data_ilqr, Y_data_ilqr, "model_ilqr.h5")

        #setup training data for lqr model
        X_data_lqr, Y_data_lqr = bh.make_data_lqr()
        # train lqr network
        bh.model_train_ff(X_data_lqr, Y_data_lqr, "model_lqr.h5")

    # choose a random initial state
    ns = bh.env.n_state
    x_0 = [1.0, 0.5, -0.5, 0.5]
    # x_0 = np.random.random(ns)
    # x_0[0:ns//2] *= 2 * np.pi
    # x_0[ns//2:] *= 2 * np.pi
    # x_0[ns//2:] -= 2 * np.pi

    print(x_0)

    # run the model.
    bh.run_model(x_0, 1500)

    # make the environment
    # env = make("Pendulum3-v0", dt=0.02)
    # env.reset()
    #
    # # call the class
    # x_range = [(0, 2 * np.pi)] * 3 + [(-9, 9)] * 3
    # Q = np.eye(6, 6) * 10
    # bh3 = BehaviorClonning3(env, ControllerTriplePendulum, 1425, folder_name="data3/", Q=Q, n_step_lqr=200, x_range=x_range)
    #
    # generate_data, train = False, False
    # if generate_data:
    #     # generate data.
    #     for _ in range(100):
    #         try:
    #             bh3.generate_data()
    #         except:
    #             env = make("Pendulum3-v0", dt=0.02)
    #             env.reset()
    #             bh3 = BehaviorClonning3(env, ControllerTriplePendulum, 2000, folder_name="data3/", Q=Q, n_step_lqr=200, x_range=x_range)
    #
    # if train:
    #     # setup training data for ilqr model
    #     X_data_ilqr, Y_data_ilqr = bh3.make_data_ilqr()
    #     # train ilqr network
    #     n_model_ilqr = 3
    #     n_step_per_ilqr_model = bh3.n_step_ilqr // n_model_ilqr
    #
    #     slc1 = [x for x in range(X_data_ilqr.shape[0]) if x % 150 <= 50]
    #     slc2 = [x for x in range(X_data_ilqr.shape[0]) if x % 150 > 50 and x % 150 <= 100]
    #     slc3 = [x for x in range(X_data_ilqr.shape[0]) if x % 150 > 100]
    #     bh3.model_train_ff(X_data_ilqr[slc1, :], Y_data_ilqr[slc1, :], "model_ilqr_0_pend3.h5")
    #     bh3.model_train_ff(X_data_ilqr[slc2, :], Y_data_ilqr[slc2, :], "model_ilqr_1_pend3.h5")
    #     bh3.model_train_ff(X_data_ilqr[slc3, :], Y_data_ilqr[slc3, :], "model_ilqr_2_pend3.h5")
    #
    #     # setup training data for lqr model
    #     X_data_lqr, Y_data_lqr = bh3.make_data_lqr()
    #     # train lqr network
    #     bh3.model_train_ff(X_data_lqr, Y_data_lqr, "model_lqr_pend3.h5")
    #
    # # choose a random initial state
    # x_0 = np.random.random(6)
    # x_0[0:3] *= 0.25 * np.pi
    # x_0[3:] *= 2
    # x_0[3:] -= 1
    #
    # print(x_0)
    #
    # # run the model.
    # bh3.run_model_2(x_0)