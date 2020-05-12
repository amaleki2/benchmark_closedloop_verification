using TrajectoryOptimization
using LinearAlgebra
import TrajectoryOptimization: dynamics  # the dynamics function must be imported
using Ipopt

struct AirPlane{T} <: TrajectoryOptimization.AbstractModel
    m::T   # mass of the airplane
    Ix::T  # moment of inertia
    Iy::T  # moment of inertia
    Iz::T  # moment of inertia
    Ixz::T # moment of inertia
    g::T   # gravity
end

function rot_matrix(phi, theta, psi)
    T_psi = hcat([
            [cos(psi), -sin(psi), 0.],
            [sin(psi),  cos(psi), 0.],
            [0., 0., 1.]
                 ]...)
    T_theta = hcat([
        [ cos(theta), 0., sin(theta)],
        [0., 1., 0.],
        [-sin(theta), 0., cos(theta)]
                   ]...)
    T_phi = hcat([
        [1., 0., 0.],
        [0., cos(phi), -sin(phi)],
        [0., sin(phi),  cos(phi)]
                 ]...)

    mat_1 = T_psi*T_theta*T_phi

    mat_2 = hcat([
        [cos(theta), sin(theta) * sin(phi),  sin(theta) * cos(phi)],
        [0.,         cos(theta) * cos(phi),  cos(theta) * sin(phi)],
        [0.,         sin(phi),               cos(phi)]
            ]...)
    mat_2 = 1 / cos(theta) * mat_2

    return mat_1, mat_2
end

model_AirPlane = AirPlane(1., 1., 2., 3., 0.5, 1.)

function dynamics!(dstates, states, actions)
    dstates = dynamics(model_AirPlane, states, actions)
end

function dynamics(model::AirPlane, states, actions)
    m = model.m  # mass of the airplane
    Ix = model.Ix  # moment of inertia
    Iy = model.Iy  # moment of inertia
    Iz = model.Iz  # moment of inertia
    Ixz = model.Ixz # moment of inertia
    g = model.g

    x, y, z, u, v, w, phi, theta, psi, p, q, r = states
    X, Y, Z, L, M, N = actions
    rot_mat1, rot_mat2 = rot_matrix(phi, theta, psi)

    a1 = [u, v, w]
    a2 = rot_mat1 * a1
    a3 = [p, q, r]
    a4 = rot_mat2 * a3
    a5 = hcat([
        [Ix,  Ixz],
        [Ixz, Iz]
             ]...)
    a6 = [L - (Iz - Iy) * q * r - Ixz * q * p,
          N - (Iy - Ix) * q * p + Ixz * q * r]

    a7 = a5\a6

    dx = a2[1]
    dy = a2[2]
    dz = a2[3]

    du = -g * sin(theta) + X / m - q * w + r * v
    dv =  g * cos(theta) * sin(phi) + Y / m - r * u + p * w
    dw =  g * cos(theta) * cos(phi) + Z / m - p * v + q * u

    dphi = a4[1]
    dthe = a4[2]
    dpsi = a4[3]

    dp = a7[1]
    dq = 1. / Iy *(M - Ixz * (r ^ 2 - p ^ 2) - (Ix - Iz) * p * r)
    dr = a7[2]

    dstates = [dx, dy, dz, du, dv, dw, dphi, dthe, dpsi, dp, dq, dr]
    return dstates
end

m, n = 6, 12
model = Model(dynamics!, n, m)
model_d = rk3(model)

x0 = zeros(12) # initial state
xf = zeros(12) # goal state
xf[1] = 20; xf[3] = 60

N = 41 # number of knot points
dt = 0.02 # time step

U0 = [0.01*rand(m) for k = 1:N-1]; # initial control trajectory

Q = 1.0*Diagonal(I,n)
Qf = 1.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)
obj = LQRObjective(Q,R,Qf,xf,N) # objective

#bnd = BoundConstraint(n,m,u_max=1.5, u_min=-1.5) # control limits
goal = goal_constraint(xf) # terminal constraint

constraints = Constraints(N) # define constraints at each time step
# for k = 1:N-1
#     constraints[k] += bnd
# end
constraints[N] += goal

prob = Problem(model_d, obj, constraints=constraints, x0=x0, xf=xf, N=N, dt=dt) # construct problem
initial_controls!(prob, U0) # initialize problem with controls

solver = solve!(prob, ALTROSolverOptions{Float64}())
