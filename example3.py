# this example runs a controller for pendulum environements.
# the controller is fairly robust for single and double pendulum, but
# not quite so for triple pendulum.
from pendulum import *

program = "pend2"
if program == "pend1":
    env = Pendulum1Env(x_0=[np.pi * 0.9, -3.], dt=0.1)
    env.reset()
    control = ControllerSinglePendulum(env)
    n_step_lqr, n_step_ilqr = 50, 25
    Q = np.eye(2, 2)
    Q[0, 0] = 5
    Qf = np.eye(2, 2) * 1000
    R = np.eye(1, 1)
    x_goal = [0., 0.]
    ilqr_actions = control.run_ilqr(Q, R, Qf, x_goal, n_step_ilqr)
    lqr_actions = control.run_lqr(Q, R, x_goal, n_step_lqr, [0.])
    print(env.x)
    env.render()
elif program == "pend2":
    env = Pendulum2Env(x_0=[np.pi * 0.5, np.pi * 0.1, 1., 0.], dt=0.025)
    env.reset()
    control = ControllerDoublePendulum(env)
    n_step_lqr, n_step_ilqr = 50, 50
    Q = np.eye(4, 4)
    Q[1, 1] = 0
    Q[2, 2] = 0
    Qf = np.eye(4, 4) * 1000
    R = np.eye(2, 2)
    x_goal = [0., 0., 0., 0.]
    ilqr_actions = control.run_ilqr(Q, R, Qf, x_goal, n_step_ilqr)
    lqr_actions = control.run_lqr(Q, R, x_goal, n_step_lqr, ilqr_actions[-1])
    print(env.x)
    env.render()
elif program == "pend3":
    x_0 = [1, 2, 2., 0, 0, 0]
    env = Pendulum3Env(dt=0.02, x_0=x_0)
    env.reset()

    n_step_lqr, n_step_ilqr = 200, 50
    Q = np.eye(6, 6) *10
    Qf = np.eye(6, 6) * 1000
    R = np.eye(3, 3)
    x_goal = [0., 0., 0., 0., 0., 0.]

    x_mid = [1.5, 1., 1.5, 1., 1., 1.]
    x_mid2 = [0.5, 0.5, 0.5, 0.0, 0.0, -0.]

    u_init = np.zeros((3, n_step_ilqr))
    u_init[0, :n_step_ilqr] = np.linspace(1, 0, n_step_ilqr)
    control = ControllerTriplePendulum(env, use_sympy=False)
    ilqr_actions_1 = control.run_ilqr(Q, R, Qf, x_mid, n_step_ilqr, u_init=u_init)

    env.x_0 = env.x
    ilqr_actions_2 = control.run_ilqr(Q, R, Qf, x_mid2, n_step_ilqr)#, u_init=u_init*0.2)

    env.x_0 = env.x
    ilqr_actions_3 = control.run_ilqr(Q, R, Qf, x_goal, n_step_ilqr)  # , u_init=u_init*0.2)

    #try:
    lqr_actions = control.run_lqr(Q, R, x_goal, n_step_lqr, ilqr_actions_3[-1], goal_action=[0]*3)
    # except:
    #     lqr_actions = [np.zeros(3)]*n_step_lqr

    env.x_0 = x_0
    env.reset()

    all_actions = ilqr_actions_1 + ilqr_actions_2 + ilqr_actions_3 + lqr_actions
    for i in range(n_step_ilqr + n_step_ilqr + n_step_ilqr + n_step_lqr):
        env.x, _, _, _ = env.step(all_actions[i])
    env.render()
    env.animate(file_name="pend3_upright.gif", dpi=90)