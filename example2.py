# This example runs the airplane model.

from airplane import AirplaneEnv

# running airplane model
env = AirplaneEnv(g=2., dt=0.05, x_0=[0.,  2., 0.,  -5., 0., -3., 0., 0., 0., 0., 0., 0.])
env.reset()

actions = [5., 0., 0., 0., 0., 0.]

for i in range(200):
    env.x, _, _, _ = env.step(actions)
env.render()