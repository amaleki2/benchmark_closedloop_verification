using NPZ
using LinearAlgebra
using StaticArrays
using TrajectoryOptimization
using NLsolve

import TrajectoryOptimization: dynamics


"""
----------------------------------------------
Airplane Dynamics
----------------------------------------------
"""

struct TriplePendulum{T} <: AbstractModel
    n::Int  # number of states
    m::Int  # number of actions
    mass::T # point mass
    l::T  # length of pendulum links
    c::T  # viscous friction coefficient
    g::T   # gravity
end

function du(th1, th2, th3, u1, u2, u3, g, mass, l, c, T1, T2, T3)
    du1 = (g*sin(th1))/l - (u3^2*sin(th1 - th3))/3 - (cos(th1 - th3)*(u1^2*sin(th1 - th3) + u2^2*sin(th2 - th3) - cos(th1 - th3)*((2*cos(th1 - th2)*(2*u1^2*sin(th1 - th2) - u3^2*sin(th2 - th3) + 2*cos(th1 - th2)*((2*u1^2*sin(th1 - th2))/3 + (u3^2*sin(th1 - th3))/3 - (g*sin(th1))/l - (T1 - c*u1)/(3*l^2*mass)) + (2*g*sin(th2))/l + (T2 - c*u2)/(l^2*mass)))/(3*((4*cos(th1 - th2)^2)/3 - 2)) - (u3^2*sin(th1 - th3))/3 - (2*u1^2*sin(th1 - th2))/3 + (g*sin(th1))/l + (T1 - c*u1)/(3*l^2*mass)) + (cos(th2 - th3)*(2*u1^2*sin(th1 - th2) - u3^2*sin(th2 - th3) + 2*cos(th1 - th2)*((2*u1^2*sin(th1 - th2))/3 + (u3^2*sin(th1 - th3))/3 - (g*sin(th1))/l - (T1 - c*u1)/(3*l^2*mass)) + (2*g*sin(th2))/l + (T2 - c*u2)/(l^2*mass)))/((4*cos(th1 - th2)^2)/3 - 2) + (g*sin(th3))/l + (T3 - c*u3)/(l^2*mass)))/(3*((cos(th2 - th3)*(cos(th2 - th3) - (2*cos(th1 - th2)*cos(th1 - th3))/3))/((4*cos(th1 - th2)^2)/3 - 2) - cos(th1 - th3)*(cos(th1 - th3)/3 + (2*cos(th1 - th2)*(cos(th2 - th3) - (2*cos(th1 - th2)*cos(th1 - th3))/3))/(3*((4*cos(th1 - th2)^2)/3 - 2))) + 1)) - (2*u1^2*sin(th1 - th2))/3 + (2*cos(th1 - th2)*(2*cos(th1 - th2)*((2*u1^2*sin(th1 - th2))/3 + (u3^2*sin(th1 - th3))/3 + (cos(th1 - th3)*(u1^2*sin(th1 - th3) + u2^2*sin(th2 - th3) - cos(th1 - th3)*((2*cos(th1 - th2)*(2*u1^2*sin(th1 - th2) - u3^2*sin(th2 - th3) + 2*cos(th1 - th2)*((2*u1^2*sin(th1 - th2))/3 + (u3^2*sin(th1 - th3))/3 - (g*sin(th1))/l - (T1 - c*u1)/(3*l^2*mass)) + (2*g*sin(th2))/l + (T2 - c*u2)/(l^2*mass)))/(3*((4*cos(th1 - th2)^2)/3 - 2)) - (u3^2*sin(th1 - th3))/3 - (2*u1^2*sin(th1 - th2))/3 + (g*sin(th1))/l + (T1 - c*u1)/(3*l^2*mass)) + (cos(th2 - th3)*(2*u1^2*sin(th1 - th2) - u3^2*sin(th2 - th3) + 2*cos(th1 - th2)*((2*u1^2*sin(th1 - th2))/3 + (u3^2*sin(th1 - th3))/3 - (g*sin(th1))/l - (T1 - c*u1)/(3*l^2*mass)) + (2*g*sin(th2))/l + (T2 - c*u2)/(l^2*mass)))/((4*cos(th1 - th2)^2)/3 - 2) + (g*sin(th3))/l + (T3 - c*u3)/(l^2*mass)))/(3*((cos(th2 - th3)*(cos(th2 - th3) - (2*cos(th1 - th2)*cos(th1 - th3))/3))/((4*cos(th1 - th2)^2)/3 - 2) - cos(th1 - th3)*(cos(th1 - th3)/3 + (2*cos(th1 - th2)*(cos(th2 - th3) - (2*cos(th1 - th2)*cos(th1 - th3))/3))/(3*((4*cos(th1 - th2)^2)/3 - 2))) + 1)) - (g*sin(th1))/l - (T1 - c*u1)/(3*l^2*mass)) + 2*u1^2*sin(th1 - th2) - u3^2*sin(th2 - th3) - (cos(th2 - th3)*(u1^2*sin(th1 - th3) + u2^2*sin(th2 - th3) - cos(th1 - th3)*((2*cos(th1 - th2)*(2*u1^2*sin(th1 - th2) - u3^2*sin(th2 - th3) + 2*cos(th1 - th2)*((2*u1^2*sin(th1 - th2))/3 + (u3^2*sin(th1 - th3))/3 - (g*sin(th1))/l - (T1 - c*u1)/(3*l^2*mass)) + (2*g*sin(th2))/l + (T2 - c*u2)/(l^2*mass)))/(3*((4*cos(th1 - th2)^2)/3 - 2)) - (u3^2*sin(th1 - th3))/3 - (2*u1^2*sin(th1 - th2))/3 + (g*sin(th1))/l + (T1 - c*u1)/(3*l^2*mass)) + (cos(th2 - th3)*(2*u1^2*sin(th1 - th2) - u3^2*sin(th2 - th3) + 2*cos(th1 - th2)*((2*u1^2*sin(th1 - th2))/3 + (u3^2*sin(th1 - th3))/3 - (g*sin(th1))/l - (T1 - c*u1)/(3*l^2*mass)) + (2*g*sin(th2))/l + (T2 - c*u2)/(l^2*mass)))/((4*cos(th1 - th2)^2)/3 - 2) + (g*sin(th3))/l + (T3 - c*u3)/(l^2*mass)))/((cos(th2 - th3)*(cos(th2 - th3) - (2*cos(th1 - th2)*cos(th1 - th3))/3))/((4*cos(th1 - th2)^2)/3 - 2) - cos(th1 - th3)*(cos(th1 - th3)/3 + (2*cos(th1 - th2)*(cos(th2 - th3) - (2*cos(th1 - th2)*cos(th1 - th3))/3))/(3*((4*cos(th1 - th2)^2)/3 - 2))) + 1) + (2*g*sin(th2))/l + (T2 - c*u2)/(l^2*mass)))/(3*((4*cos(th1 - th2)^2)/3 - 2)) + (T1 - c*u1)/(3*l^2*mass)

    du2 = -(2*cos(th1 - th2)*((2*u1^2*sin(th1 - th2))/3 + (u3^2*sin(th1 - th3))/3 + (cos(th1 - th3)*(u1^2*sin(th1 - th3) + u2^2*sin(th2 - th3) - cos(th1 - th3)*((2*cos(th1 - th2)*(2*u1^2*sin(th1 - th2) - u3^2*sin(th2 - th3) + 2*cos(th1 - th2)*((2*u1^2*sin(th1 - th2))/3 + (u3^2*sin(th1 - th3))/3 - (g*sin(th1))/l - (T1 - c*u1)/(3*l^2*mass)) + (2*g*sin(th2))/l + (T2 - c*u2)/(l^2*mass)))/(3*((4*cos(th1 - th2)^2)/3 - 2)) - (u3^2*sin(th1 - th3))/3 - (2*u1^2*sin(th1 - th2))/3 + (g*sin(th1))/l + (T1 - c*u1)/(3*l^2*mass)) + (cos(th2 - th3)*(2*u1^2*sin(th1 - th2) - u3^2*sin(th2 - th3) + 2*cos(th1 - th2)*((2*u1^2*sin(th1 - th2))/3 + (u3^2*sin(th1 - th3))/3 - (g*sin(th1))/l - (T1 - c*u1)/(3*l^2*mass)) + (2*g*sin(th2))/l + (T2 - c*u2)/(l^2*mass)))/((4*cos(th1 - th2)^2)/3 - 2) + (g*sin(th3))/l + (T3 - c*u3)/(l^2*mass)))/(3*((cos(th2 - th3)*(cos(th2 - th3) - (2*cos(th1 - th2)*cos(th1 - th3))/3))/((4*cos(th1 - th2)^2)/3 - 2) - cos(th1 - th3)*(cos(th1 - th3)/3 + (2*cos(th1 - th2)*(cos(th2 - th3) - (2*cos(th1 - th2)*cos(th1 - th3))/3))/(3*((4*cos(th1 - th2)^2)/3 - 2))) + 1)) - (g*sin(th1))/l - (T1 - c*u1)/(3*l^2*mass)) + 2*u1^2*sin(th1 - th2) - u3^2*sin(th2 - th3) - (cos(th2 - th3)*(u1^2*sin(th1 - th3) + u2^2*sin(th2 - th3) - cos(th1 - th3)*((2*cos(th1 - th2)*(2*u1^2*sin(th1 - th2) - u3^2*sin(th2 - th3) + 2*cos(th1 - th2)*((2*u1^2*sin(th1 - th2))/3 + (u3^2*sin(th1 - th3))/3 - (g*sin(th1))/l - (T1 - c*u1)/(3*l^2*mass)) + (2*g*sin(th2))/l + (T2 - c*u2)/(l^2*mass)))/(3*((4*cos(th1 - th2)^2)/3 - 2)) - (u3^2*sin(th1 - th3))/3 - (2*u1^2*sin(th1 - th2))/3 + (g*sin(th1))/l + (T1 - c*u1)/(3*l^2*mass)) + (cos(th2 - th3)*(2*u1^2*sin(th1 - th2) - u3^2*sin(th2 - th3) + 2*cos(th1 - th2)*((2*u1^2*sin(th1 - th2))/3 + (u3^2*sin(th1 - th3))/3 - (g*sin(th1))/l - (T1 - c*u1)/(3*l^2*mass)) + (2*g*sin(th2))/l + (T2 - c*u2)/(l^2*mass)))/((4*cos(th1 - th2)^2)/3 - 2) + (g*sin(th3))/l + (T3 - c*u3)/(l^2*mass)))/((cos(th2 - th3)*(cos(th2 - th3) - (2*cos(th1 - th2)*cos(th1 - th3))/3))/((4*cos(th1 - th2)^2)/3 - 2) - cos(th1 - th3)*(cos(th1 - th3)/3 + (2*cos(th1 - th2)*(cos(th2 - th3) - (2*cos(th1 - th2)*cos(th1 - th3))/3))/(3*((4*cos(th1 - th2)^2)/3 - 2))) + 1) + (2*g*sin(th2))/l + (T2 - c*u2)/(l^2*mass))/((4*cos(th1 - th2)^2)/3 - 2)


    du3 = (u1^2*sin(th1 - th3) + u2^2*sin(th2 - th3) - cos(th1 - th3)*((2*cos(th1 - th2)*(2*u1^2*sin(th1 - th2) - u3^2*sin(th2 - th3) + 2*cos(th1 - th2)*((2*u1^2*sin(th1 - th2))/3 + (u3^2*sin(th1 - th3))/3 - (g*sin(th1))/l - (T1 - c*u1)/(3*l^2*mass)) + (2*g*sin(th2))/l + (T2 - c*u2)/(l^2*mass)))/(3*((4*cos(th1 - th2)^2)/3 - 2)) - (u3^2*sin(th1 - th3))/3 - (2*u1^2*sin(th1 - th2))/3 + (g*sin(th1))/l + (T1 - c*u1)/(3*l^2*mass)) + (cos(th2 - th3)*(2*u1^2*sin(th1 - th2) - u3^2*sin(th2 - th3) + 2*cos(th1 - th2)*((2*u1^2*sin(th1 - th2))/3 + (u3^2*sin(th1 - th3))/3 - (g*sin(th1))/l - (T1 - c*u1)/(3*l^2*mass)) + (2*g*sin(th2))/l + (T2 - c*u2)/(l^2*mass)))/((4*cos(th1 - th2)^2)/3 - 2) + (g*sin(th3))/l + (T3 - c*u3)/(l^2*mass))/((cos(th2 - th3)*(cos(th2 - th3) - (2*cos(th1 - th2)*cos(th1 - th3))/3))/((4*cos(th1 - th2)^2)/3 - 2) - cos(th1 - th3)*(cos(th1 - th3)/3 + (2*cos(th1 - th2)*(cos(th2 - th3) - (2*cos(th1 - th2)*cos(th1 - th3))/3))/(3*((4*cos(th1 - th2)^2)/3 - 2))) + 1)

    return du1, du2, du3
end


function dynamics(model::TriplePendulum, states, actions)
    mass = model.mass  # point mass
    l = model.l  # length of pendulum links
    c = model.c  # viscous friction coefficient
    g = model.g

    th1, th2, th3, u1, u2, u3 = states
    T1, T2, T3 = actions

    # function f!(F, x)
    #     x1, x2, x3  = x
    #     F[1] = 3 * x1 + 2 * x2 * cos(th1 - th2) + x3 * cos(th1 - th3) +
    #           2 * u1 ^ 2 * sin(th1 - th2) + u3 ^ 2 * sin(th1 - th3) +
    #               - g / l * sin(th1) * 3 + (c * u1 - T1) / (mass * l ^ 2)
    #
    #     F[2] = 2 * x1 * cos(th1 - th2) + 2 * x2 + x3 * cos(th2 - th3) +
    #           - 2 * u1 ^ 2 * sin(th1 - th2) + u3 ^ 2 * sin(th2 - th3) +
    #           - g / l * sin(th2) * 2 + (c * u2 - T2) / (mass * l ^ 2)
    #     F[3] = x1 * cos(th1 - th3) + x2 * cos(th2 - th3) + x3 +
    #           - u1 ^ 2 * sin(th1 - th3) - u2 ^ 2 * sin(th2 - th3) +
    #           - g / l * sin(th3) + (c * u3 - T3) / (mass * l ^ 2)
    # end
    # init_u = [0., 0., 0.]
    # sol = nlsolve(f!, init_u)#, autodiff = :forward)
    # du1, du2, du3 = sol.zero


    dth1 = u1
    dth2 = u2
    dth3 = u3
    du1, du2, du3 = du(th1, th2, th3, u1, u2, u3, g, mass, l, c, T1, T2, T3)


    dstates = [dth1, dth2, dth3, du1, du2, du3]
    return dstates
end

Base.size(model::TriplePendulum) = model.n, model.m
TriplePendulum() = TriplePendulum(6, 3,
    0.5, 0.5, 0.0, 9.8)

"""
----------------------------------------------
One Run
----------------------------------------------
"""
function run_opt_traj(x0, xf; tf=5., N=51)
    """
    xf: goal state
    N : number of knot points
    tf: final time
    """
    model = TriplePendulum()
    n, m = size(model)

    # Objective
    Q = Diagonal(@SVector fill(1e-2,n))
    R = Diagonal(@SVector fill(1e-1,m))
    Qf = Diagonal(@SVector fill(1e+1,n))
    obj = LQRObjective(Q, R, Qf, xf, N)

    # Constraints
    conSet = ConstraintSet(n,m,N)
    # bnd1 = BoundConstraint(n,m, u_min=-10.0, u_max=10.0)
    # add_constraint!(conSet, bnd1, 1:N-1)
    goal = GoalConstraint(xf)


    add_constraint!(conSet, goal, Int32((N-1)*0.9):N)

    # Problem
    prob = Problem(model, obj, xf, tf, constraints=conSet, x0=x0, integration=RK3)

    # Solution options
    # opts_al = AugmentedLagrangianSolverOptions()
    # opts_al.opts_uncon.iterations = 50  # set iLQR iterations
    # opts_al.iterations = 20
    # opts_al.verbose = true
    # opts_al.opts_uncon.verbose = true

    # Solve
    print("solving ...  ")
    #solver = AugmentedLagrangianSolver(prob)
    solver = ALTROSolver(prob)
    solve!(solver)

    # Outputs
    s = TrajectoryOptimization.states(solver)
    u = TrajectoryOptimization.controls(solver)

    # Save states and controls
    println("saving ...")
    controls_data = zeros(N-1, m)
    states_data = zeros(N, n)

    for i = 1:N
        if i > 1
            c_data = u[i-1].data
            c_data_array = [c for c in c_data]
            controls_data[i-1, :] = c_data_array
        end
        s_data = s[i].data
        s_data_array = [s for s in s_data]
        states_data[i, :] = s_data_array
    end
    return states_data, controls_data
end

"""
----------------------------------------------
Many Runs
----------------------------------------------
"""
xf = [0., 0., 0., 0., 0., 0.,]
xf_static = SVector(xf...)

x0 = [2., 0., 2., 0., 0., 0.]
x0_static = SVector(x0...)

function generate_data(;ndata=1000, eps=0.05)
    # final points is always fixed
    xf = [0., 0., 0., 0., 0., 0.]
    xf_static = SVector(xf...)

    for n in 1:ndata
        test1 = isfile("data_pend3/julia/state_$n.npz")
        if test1
            continue
        end
        print("n=$n,    ")
        # create a random starting point
        x0 = rand(6)
        x0[1:3] = x0[1:3]*3.
        x0[4:6] = x0[4:6]*5.0 - 2.5
        x0_static = SVector(x0...)

        # run optimized trajectory algorithm
        st, cn = run_opt_traj(x0_static, xf_static; tf=2., N=201)

        # check if algorithm has succeeded.
        if norm(st[end, :] - xf) < eps
            npzwrite("data_airplane/julia2/state_$n.npz", st)
            npzwrite("data_airplane/julia2/control_$n.npz", cn)
        else
            println("it did not succeed.")
        end
        println(round.(cn[end, :], digits=3))
        println(round.(st[end, :], digits=3))
    end
end
#
# function generate_data_g0(adrs ;ndata=1000, eps=0.05, ns=1)
#     # final points is always fixed
#     xf = [25., 0., 50., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
#     xf_static = SVector(xf...)
#
#     for n in ns:ndata
#         test1 = isfile("$(adrs)state_$n.npz")
#         test2 = isfile("$(adrs)short_state_$n.npz")
#         test3 = isfile("$(adrs)long_state_$n.npz")
#         if test1 | test2 | test3
#             continue
#         end
#         print("n=$n,    ")
#         # create a random starting point
#         x0 = rand(12)
#         x0[1:6]  = x0[1:6]*5.0 .- 2.5
#         x0[7:12] = x0[7:12]*1.0 .- 0.5
#         x0_static = SVector(x0...)
#
#         # run optimized trajectory algorithm
#         st, cn = run_opt_traj(x0_static, xf_static; tf=20., N=51)
#
#         # check if algorithm has succeeded.
#         if norm(st[end, :]-xf) < eps
#             npzwrite("$(adrs)state_$n.npz", st)
#             npzwrite("$(adrs)control_$n.npz", cn)
#         else
#             print("   trying short:")
#             st, cn = run_opt_traj(x0_static, xf_static; tf=20., N=26)
#             if norm(st[end, :]-xf) < eps
#                 npzwrite("$(adrs)short_state_$n.npz", st)
#                 npzwrite("$(adrs)short_control_$n.npz", cn)
#             else
#                 print("   trying long:")
#                 st, cn = run_opt_traj(x0_static, xf_static; tf=20., N=101)
#                 if norm(st[end, :]-xf) < eps
#                     npzwrite("$(adrs)long_state_$n.npz", st)
#                     npzwrite("$(adrs)long_control_$n.npz", cn)
#                 else
#                     npzwrite("$(adrs)dnw_state_$n.npz", x0)
#                 end
#             end
#         end
#         println(round.(cn[end, :], digits=3))
#         println(round.(st[end, :], digits=3))
#     end
# end


# x0 = @SVector[0.,  2., 0.,  -5., 10., -3., 1., 1., 1., 0., 0., 0.]  # initial state
# xf = @SVector[25., 0., 25.,  0., 0., 0., 0., 0., 0., 0., 0., 0.]  # goal state (i.e. swing up)
#
# tf = 10.
# N = 51
# N_intp = 501
# st, cn = run_model(x0, xf; tf=tf, N=N)
#
# t = collect(range(0, stop=tf, length=N))
# t_intp = collect(range(0, stop=tf, length=N_intp))
# cn_intp = zeros(N_intp-1, size(cn)[2])
# for i = 1:size(cn)[2]
#     spl = Spline1D(t[2:end], cn[:, i])
#     cn_intp[:, i] = spl(t_intp[2:end])
# end
