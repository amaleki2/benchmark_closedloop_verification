from pendulum import Pendulum1Env, Pendulum2Env, Pendulum3Env

# running a single pendulum
p1 = Pendulum1Env(x_0=[2., 0.], dt =0.1)
p1.reset()
for _ in range(200):
    p1.step([0.])
p1.render()


# running a double pendulum
p2 = Pendulum2Env(x_0=[3., 1., 3., 1.], dt=0.01)
p2.reset()
for _ in range(200):
    p2.step([0., 0.])
p2.render()

# running a triple pendulum
p3 = Pendulum3Env(x_0=[3., 1., 3., 1., 0., 1.], dt=0.01)
p3.reset()
for _ in range(200):
    p3.step([0., 0., 0.])
p3.render()