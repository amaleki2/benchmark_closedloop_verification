"""
This file is generated with matlab
The functions are used to populate the matrices A and B for the linearized dynamics
of the double pendulum environment.
                                 f = Ax + Bu
That's why indeces start from 1; e.g. pend2_A_1_1 is row 1 column 1 of A: A[0, 0] in python.
"""

from numpy import sin, cos

def pend2_A_1_1(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return 0.

def pend2_A_1_2(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return 0.

def pend2_A_1_3(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return 1.

def pend2_A_1_4(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return 0.

def pend2_A_2_1(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return 0.

def pend2_A_2_2(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return 0.

def pend2_A_2_3(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return 0.

def pend2_A_2_4(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return 1.

def pend2_A_3_1(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return (2 * cos(th1 - th2) * sin(th1 - th2) * (
                    2 * c * u1 - 2 * T1 + T2 * cos(th1 - th2) - c * u2 * cos(th1 - th2) - 4 * L * g * m * sin(
                th1) + 2 * L ** 2 * m * u2 ** 2 * sin(th1 - th2) + L * g * m * cos(th1 - th2) * sin(
                th2) + L ** 2 * m * u1 ** 2 * cos(th1 - th2) * sin(th1 - th2))) / (
                                  L ** 2 * m * (cos(th1 - th2) ** 2 - 4) ** 2) - (
                                  T2 * sin(th1 - th2) - c * u2 * sin(th1 - th2) + 4 * L * g * m * cos(
                              th1) - 2 * L ** 2 * m * u2 ** 2 * cos(th1 - th2) - L ** 2 * m * u1 ** 2 * cos(
                              th1 - th2) ** 2 + L ** 2 * m * u1 ** 2 * sin(th1 - th2) ** 2 + L * g * m * sin(
                              th1 - th2) * sin(th2)) / (
                                  L ** 2 * m * (cos(th1 - th2) ** 2 - 4))

def pend2_A_3_2(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return (T2 * sin(th1 - th2) - c * u2 * sin(th1 - th2) - 2 * L ** 2 * m * u2 ** 2 *cos(
                th1 - th2) - L ** 2 * m * u1 ** 2 * cos(th1 - th2) ** 2 + L ** 2 * m * u1 ** 2 * sin(
                th1 - th2) ** 2 + L * g * m * cos(th1 - th2) * cos(th2) + L * g * m * sin(th1 - th2) * sin(th2)) / (
                                  L ** 2 * m * (cos(th1 - th2) ** 2 - 4)) - (2 * cos(th1 - th2) * sin(th1 - th2) * (
                    2 * c * u1 - 2 * T1 + T2 * cos(th1 - th2) - c * u2 * cos(th1 - th2) - 4 * L * g * m * sin(
                th1) + 2 * L ** 2 * m * u2 ** 2 * sin(th1 - th2) + L * g * m * cos(th1 - th2) * sin(
                th2) + L ** 2 * m * u1 ** 2 * cos(th1 - th2) * sin(th1 - th2))) / (
                                  L ** 2 * m * (cos(th1 - th2) ** 2 - 4) ** 2)

def pend2_A_3_3(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return (2 * m * u1 * cos(th1 - th2) * sin(th1 - th2) * L ** 2 + 2 * c) / (
                L ** 2 * m * (cos(th1 - th2) ** 2 - 4))

def pend2_A_3_4(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return -(- 4 * m * u2 * sin(th1 - th2) * L ** 2 + c * cos(th1 - th2)) / (
                L ** 2 * m * (cos(th1 - th2) ** 2 - 4))

def pend2_A_4_1(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return - (T1 * sin(th1 - th2) - c * u1 * sin(th1 - th2) + 2 * L ** 2 * m * u1 ** 2 * cos(
                th1 - th2) + L ** 2 * m * u2 ** 2 * cos(th1 - th2) ** 2 - L ** 2 * m * u2 ** 2 * sin(
                th1 - th2) ** 2 - 2 * L * g * m * cos(th1 - th2) * cos(th1) + 2 * L * g * m * sin(th1 - th2) * sin(
                th1)) / (
                                  L ** 2 * m * (cos(th1 - th2) ** 2 - 4)) - (2 * cos(th1 - th2) * sin(th1 - th2) * (
                    2 * T2 - 2 * c * u2 - T1 * cos(th1 - th2) + c * u1 * cos(th1 - th2) + 2 * L * g * m * sin(
                th2) + 2 * L ** 2 * m * u1 ** 2 * sin(th1 - th2) - 2 * L * g * m * cos(th1 - th2) * sin(
                th1) + L ** 2 * m * u2 ** 2 * cos(th1 - th2) * sin(th1 - th2))) / (
                                  L ** 2 * m * (cos(th1 - th2) ** 2 - 4) ** 2)

def pend2_A_4_2(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return (T1 * sin(th1 - th2) - c * u1 * sin(th1 - th2) - 2 * L * g * m * cos(
                th2) + 2 * L ** 2 * m * u1 ** 2 * cos(
                th1 - th2) + L ** 2 * m * u2 ** 2 * cos(th1 - th2) ** 2 - L ** 2 * m * u2 ** 2 * sin(
                th1 - th2) ** 2 + 2 * L * g * m * sin(th1 - th2) * sin(th1)) / (
                                      L ** 2 * m * (cos(th1 - th2) ** 2 - 4)) + (
                                  2 * cos(th1 - th2) * sin(th1 - th2) * (
                                  2 * T2 - 2 * c * u2 - T1 * cos(th1 - th2) + c * u1 * cos(
                              th1 - th2) + 2 * L * g * m * sin(
                              th2) + 2 * L ** 2 * m * u1 ** 2 * sin(th1 - th2) - 2 * L * g * m * cos(th1 - th2) * sin(
                              th1) + L ** 2 * m * u2 ** 2 * cos(th1 - th2) * sin(th1 - th2))) / (
                                  L ** 2 * m * (cos(th1 - th2) ** 2 - 4) ** 2)

def pend2_A_4_3(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return -(4 * m * u1 * sin(th1 - th2) * L ** 2 + c * cos(th1 - th2)) / (
                L ** 2 * m * (cos(th1 - th2) ** 2 - 4))

def pend2_A_4_4(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return (- 2 * m * u2 * cos(th1 - th2) * sin(th1 - th2) * L ** 2 + 2 * c) / (
                L ** 2 * m * (cos(th1 - th2) ** 2 - 4))

def pend2_B_1_1(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return 0.

def pend2_B_1_2(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return 0.

def pend2_B_2_1(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return 0.

def pend2_B_2_2(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return 0.

def pend2_B_3_1(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return -2 / (L ** 2 * m * (cos(th1 - th2) ** 2 - 4))

def pend2_B_3_2(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return cos(th1 - th2) / (L ** 2 * m * (cos(th1 - th2) ** 2 - 4))

def pend2_B_4_1(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return cos(th1 - th2) / (L ** 2 * m * (cos(th1 - th2) ** 2 - 4))

def pend2_B_4_2(th1, th2, u1, u2, g, m, L, c, T1, T2):
    return -2 / (L ** 2 * m * (cos(th1 - th2) ** 2 - 4))