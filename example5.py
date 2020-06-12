# this example runs a behavior generates data and then train a network
# to control the pendulum

from pendulum import *
data_folder = ""
generate_data, train = False, False

n_step_ilqr, n_step_lqr, n_data, dt, lr = 100, 100, 1500, 0.05, 0.001
env = Pendulum2Env(dt=dt)
controller = ControllerDoublePendulum
folder_name = data_folder
x_range = [(0, 2*np.pi)] * 2 + [(-2.0 * np.pi, 2.0 * np.pi)] * 2

# call the class
bh = BehaviorClonning(env, controller, n_data, folder_name=folder_name,
                      n_step_ilqr=n_step_ilqr, n_step_lqr=n_step_lqr,
                      x_range=x_range
                      )


if generate_data:
    bh.generate_data()
if train:
    # training ilqr model
    X_data_ilqr, Y_data_ilqr = bh.make_data_ilqr()
    np.clip(Y_data_ilqr, -3.0, 3.0)
    bh.model_train_ff(X_data_ilqr, Y_data_ilqr, "model_ilqr.h5", layer_dims=[25, 25],
                      epoch_size=70, lr=lr, batch_size=10)

    # training lqr model
    X_data_lqr, Y_data_lqr = bh.make_data_lqr()
    bh.model_train_ff(X_data_lqr, Y_data_lqr, "model_lqr.h5", layer_dims=[25, 25],
                      epoch_size=50, lr=lr, batch_size=10)

# test
ns = bh.env.n_state
x_0 = np.random.random(ns)
x_0 *= np.pi/2

print(x_0)
bh.run_model(x_0, 1500)
