# from scipy.optimize import minimize
# import time
# import numpy as np
#
# ndim, N = 2, 30
# tbegin, tend = 0, 20
# xB = np.zeros((1, ndim)) + np.pi
# xF = np.zeros((1, ndim))
# vB = np.zeros((1, ndim))
# vF = np.zeros((1, ndim))
# dt = (tend - tbegin) / N
#
# m, L, g, c = 0.5, 0.5, 9.8, 0.1
#
# def obj(x):
#     if np.ndim(x) == 1:
#         x = x.reshape(ndim, -1)
#
#     x = make_x_2d(x)
#
#     u = x[:, -(N + 1):]
#     obj_val = 0
#     for i in range(ndim):
#         obj_val += 2 * np.sum(u[i, 1:-1] ** 2) + u[i, 0] ** 2 + u[i, -1] ** 2
#     return obj_val
#
# # def jac_obj(all_vec):
# #     jac = np.zeros(3*N+3)
# #     u = x[-(N+1):]
# #     jac[-N-1]  = u[-N-1]*2
# #     jac[-N:-1] = u[-N:-1]*4
# #     jac[-1]    = u[-1]*2
# #     return jac
# #
# # def hess_obj(x):
# #     h1 = np.diag([4.]*(N+1))
# #     h1[0, 0] = 2.
# #     h1[-1, -1] = 2.
# #
# #     h = np.zeros((3*N+3, 3*N+3))
# #     h[-(N+1):, -(N+1):] = h1
# #     return h
#
# def make_x_2d(x):
#
#     if np.ndim(x) == 1:
#         x = x.reshape(ndim, -1)
#     elif np.ndim(x) > 2:
#         raise (IOError("input vec should be 2d."))
#
#     if x.shape[1] != 3 * N + 3:
#         raise (IOError("input vec should of size [?,%d]" % 3 * N + 3))
#
#     return x
#
#
# def func_x_i(i, k):
#     def f1(x):
#         x = make_x_2d(x)
#         dxdt = (x[k, i + 1 + 0 * N] - x[k, i + 0 + 0 * N]) / dt
#         v = 1 / 2 * (x[k, i + 2 + 1 * N] + x[k, i + 1 + 1 * N])
#         return dxdt - v
# #        return x[k, i + 1 + 0 * N] - x[k, i + 0 + 0 * N] - dt / 2 * (x[k, i + 2 + 1 * N] + x[k, i + 1 + 1 * N])
#     def f2(x):
#         x = make_x_2d(x)
#         dvdt1 = (x[0, i + 2 + 1 * N] - x[0, i + 1 + 1 * N]) / dt
#         dvdt2 = (x[1, i + 2 + 1 * N] - x[1, i + 1 + 1 * N]) / dt
#         x1 = 1 / 2 * (x[0, i + 1 + 0 * N] - x[0, i + 0 + 0 * N])
#         x2 = 1 / 2 * (x[1, i + 1 + 0 * N] - x[1, i + 0 + 0 * N])
#         v1 = 1 / 2 * (x[0, i + 2 + 1 * N] - x[0, i + 1 + 1 * N])
#         v2 = 1 / 2 * (x[1, i + 2 + 1 * N] - x[1, i + 1 + 1 * N])
#         T1 = 1 / 2 * (x[0, i + 3 + 2 * N] + x[0, i + 2 + 2 * N])
#         T2 = 1 / 2 * (x[1, i + 3 + 2 * N] + x[1, i + 2 + 2 * N])
#
#         if k == 0:
#             return (2*c*v1 - 2*T1 + T2*np.cos(x1 - x2) - c*v2*np.cos(x1 - x2) - 4*L*g*m*np.sin(x1) + 2*L**2*m*v2**2*np.sin(x1 - x2) +
#                 L*g*m*np. cos(x1 - x2)*np.sin(x2) + L**2*m*v1**2*np.cos(x1 - x2)*np.sin(x1 - x2))/(L**2*m*(np.cos(x1 - x2)**2 - 4)) - dvdt1
#         elif k == 1:
#             return -(2*T2 - 2*c*v2 - T1*np.cos(x1 - x2) + c*v1*np.cos(x1 - x2) + 2*L*g*m*np.sin(x2) + 2*L**2*m*v1**2*np.sin(x1 - x2) -
#                      2*L*g*m*np.cos(x1 - x2)*np.sin(x1) + L**2*m*v2**2*np.cos(x1 - x2)*np.sin(x1 - x2))/(L**2*m*(np.cos(x1 - x2)**2 - 4)) - dvdt2
#         else:
#             raise("error")
#     return f1, f2
#
# def set_xB():
#     def f_xB(x):
#         x = make_x_2d(x)
#         return x[k, 0] - xB
#     return f_xB
#
# def set_xF():
#     def f_xF(x):
#         x = make_x_2d(x)
#         return x[k, N] - xF
#     return f_xF
#
# def set_vB():
#     def f_xB(x):
#         x = make_x_2d(x)
#         return x[k, 1 + N] - vB
#     return f_xB
#
# def set_vF():
#     def f_vF(x):
#         x = make_x_2d(x)
#         return x[k, 1 + 2 * N] - vF
#     return f_vF
#
#
# cons = []
# for k in range(ndim):
#     for i in range(N):
#         f1, f2 = func_x_i(i, k)
#         cons.append({'type': 'eq', 'fun': f1})
#         cons.append({'type': 'eq', 'fun': f2})
#     cons.append({'type': 'eq', 'fun': set_xB()})
#     cons.append({'type': 'eq', 'fun': set_xF()})
#     cons.append({'type': 'eq', 'fun': set_vB()})
#     cons.append({'type': 'eq', 'fun': set_vF()})
#
# xguess = np.concatenate((np.linspace(0, 1, N+1)*dt, np.ones(N+1), np.zeros(N+1)))
# xguess = xguess.reshape(1,-1)
# xguess = np.concatenate((xguess, xguess))
# t1 = time.time()
# res = minimize(obj, xguess, method='SLSQP',
#                #bounds=bnds,
#                constraints=cons,
#                # jac=jac_obj,
#                # hess=hess_obj
#                )
# t2 = time.time()
# print(t2-t1)
#
#
# t = np.linspace(0, 1, N+1)
# x = res.x[0 : N + 1]
# u = res.x[2*N + 2 :]
# xs = 3 * t ** 2 - 2 * t ** 3
# us = 6 - 12 * t
#
# import matplotlib.pyplot as plt
# plt.plot(t,x, label="x", color='red', marker='o')
# plt.plot(t,xs, label="xs", color='red', linestyle="--")
# # plt.plot(t,u, label="u", color='blue', marker= 'o')
# # plt.plot(t,us, label="us", color='blue', linestyle="--")
# plt.legend()
# plt.show()


from scipy.optimize import minimize
import time
import numpy as np

ndim, ntime = 2, 40
tbegin, tend = 0, 1
xB = np.zeros(ndim) + np.pi
xF = np.zeros(ndim)
vB = np.zeros(ndim)
vF = np.zeros(ndim)
dt = (tend - tbegin) / ntime


def obj(X):
    assert X.ndim == 1
    Xs = X.copy().reshape((ntime+1, 3, ndim))

    obj_val = 0
    for i in range(ntime):
        if i in [0, ntime]:
            w = 1
        else:
            w = 2
        for k in range(ndim):
            obj_val += Xs[i, 2, k] ** 2 * w
    return obj_val


def prepare_x(X, i):
    assert X.ndim == 1
    Xs = X.copy().reshape((ntime+1, 3, ndim))

    x = []; v = []; u = []; dx = []; dv = []
    for k in range(ndim):
        x.append(1 / 2 * (Xs[i, 0, k] + Xs[i + 1, 0, k]))
        v.append(1 / 2 * (Xs[i, 1, k] + Xs[i + 1, 1, k]))
        u.append(1 / 2 * (Xs[i, 2, k] + Xs[i + 1, 2, k]))
        dx.append(1 / dt * (Xs[i + 1, 0, k] - Xs[i, 0, k]))
        dv.append(1 / dt * (Xs[i + 1, 1, k] - Xs[i, 1, k]))
    return x, v, u, dx, dv


def func_x_i(env, i, k):
    def f1(X):
        x, v, u, dx, dv = prepare_x(X, i)
        return dx[k] - v[k]

    def f2(X):
        x, v, u, dx, dv = prepare_x(X, i)
        eqs = env.setup_equation(u, x + v, u)
        return eqs[k]

    return f1, f2


def func_x_boundaries(k):
    def f_xB(X):
        X = X.reshape((ntime+1, 3, ndim))
        return X[0, 0, k] - xB[k]

    def f_xF(X):
        X = X.reshape((ntime+1, 3, ndim))
        return X[ntime, 0, k] - xF[k]

    def f_vB(X):
        X = X.reshape((ntime+1, 3, ndim))
        return X[0, 1, k] - vB[k]

    def f_vF(X):
        X = X.reshape((ntime+1, 3, ndim))
        return X[ntime, 1, k] - vF[k]

    return f_xB, f_xF, f_vB, f_vF


from gym.envs.registration import make
if __name__ == '__main__':
    env = make("Pendulum2-v0", dt=dt, x_0=np.array([1.,1.,0.,0.]))

env.reset()

cons = []
for k in range(ndim):
    for i in range(ntime):
        f1, f2 = func_x_i(env, i, k)
        cons.append({'type': 'eq', 'fun': f1})
        cons.append({'type': 'eq', 'fun': f2})
    f_xB, f_xF, f_vB, f_vF = func_x_boundaries(k)
    cons.append({'type': 'eq', 'fun': f_xB})
    cons.append({'type': 'eq', 'fun': f_xF})
    cons.append({'type': 'eq', 'fun': f_vB})
    cons.append({'type': 'eq', 'fun': f_vF})

# Xguess = env.env.x_0.tolist() + [0., 0.]
# for i in range(ntime):
#     Xguess += env.step([0., 0.])[0].tolist()
#     Xguess += [0., 0.]
#
# env.render()
# xguess = np.linspace(0, 1, ntime + 1) * dt
# vguess = np.ones(ntime + 1)
# uguess = np.zeros(ntime + 1)
#
# Xguess = np.array([])
# for x, v, u in zip(xguess, vguess, uguess):
#     Xguess = np.append(Xguess, x)
#     Xguess = np.append(Xguess, x)
#     Xguess = np.append(Xguess, v)
#     Xguess = np.append(Xguess, v)
#     Xguess = np.append(Xguess, u)
#     Xguess = np.append(Xguess, u)


t1 = time.time()
# res = minimize(obj, Xguess, method='SLSQP',
#                # bounds=bnds,
#                constraints=cons,
#                # jac=jac_obj,
#                # hess=hess_obj
#                options={"disp": True, "maxiter": 200},
#                )
# t2 = time.time()
# x = res.x.reshape((ntime + 1, 3, ndim))
# uvec = np.squeeze(x[:,2,:])
#
# env.reset()
# env.env.reset()
#
# for u in uvec:
#     env.step(u)
# env.render()


# print(t2 - t1)

from gym.envs.registration import make
env = make("Pendulum3-v0", dt=0.02, x_0=np.array([np.pi, np.pi, np.pi, 0., 0., 0.]))
env.reset()

uvec = np.loadtxt("/home/amaleki/Downloads/OptimTraj-master (1)/OptimTraj-master/demo/triplePendulum/myFile.txt")
for u in uvec.T:
    env.step(u)
env.render()