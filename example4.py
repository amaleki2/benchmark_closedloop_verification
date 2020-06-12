# This example runs the pendulum and airplane models with neural network controllers.

import numpy as np
from keras.models import load_model
from airplane import AirplaneEnv
from pendulum import Pendulum1Env, Pendulum2Env, Pendulum3Env

# running a single pendulum
x_0 = np.random.random(2)
x_0 *= 0.2
x_0[0] += 1.0
p1 = Pendulum1Env(x_0=x_0, dt =0.05)
p1.reset()
model = load_model("controller_files/controller_single_pendulum.h5")
for i in range(150):
    nn_in = np.atleast_2d(p1.x)
    nn_out = model.predict(nn_in)
    actions = nn_out[0]
    p1.step(actions)
p1.render()

# running a double pendulum
x_0 = np.random.random(4)
x_0 *= 0.3
x_0 += 1.
p2 = Pendulum2Env(x_0=x_0, dt =0.02)
p2.reset()
model = load_model("controller_files/controller_double_pendulum_less_robust.h5")
#model = load_model("controller_files/controller_double_pendulum_more_robust.h5")


for i in range(250):
    nn_in = np.atleast_2d(p2.x)
    nn_out = model.predict(nn_in)
    actions = nn_out[0]
    p2.step(actions)
p2.render()


# running a triple pendulum
x_0 = np.random.random(6)
x_0 *= 0.2
x_0[:3] += 0.5
x_0[3:] -= 0.5
p3 = Pendulum3Env(x_0=x_0, dt =0.02)
p3.reset()
model = load_model("controller_files/controller_triple_pendulum.h5")
for i in range(250):
    nn_in = np.atleast_2d(p3.x)
    nn_out = model.predict(nn_in)
    actions = nn_out[0]
    p3.step(actions)
p3.render()


# running airplane model
x_0 = [0., 0., -3-5*np.random.random(), 0., np.random.random(), 1., np.random.random(), .5, np.random.random(), 0., 0., 0.]
env = AirplaneEnv(g=1., dt=0.1, x_0=x_0)
env.reset()
print(env.x)

model = load_model('controller_files/controller_airplane.h5')

for i in range(50):
    nn_in = np.atleast_2d(env.x)
    nn_out = model.predict(nn_in)
    actions = nn_out[0]
    env.step(actions)

print(env.x)
x_goal = [25., 0., 50., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
env.render(x_goals=[x_goal])


