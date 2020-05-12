"""
This file is generated with matlab
The functions are used to populate the matrices A and B for the linearized dynamics
of the airplane environment.
                                 f = Ax + Bu
That's why indeces start from 1; e.g. airplaneA_1_1 is row 1 column 1 of A: A[0, 0] in python.
"""


from numpy import sin, cos

def airplane_A_1_1(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_1_2(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_1_3(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_1_4(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return cos(psi)*cos(theta)

def airplane_A_1_5(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return cos(psi)*sin(phi)*sin(theta) - cos(phi)*sin(psi)

def airplane_A_1_6(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)

def airplane_A_1_7(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return v*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)) + w*(cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(theta))

def airplane_A_1_8(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return w*cos(phi)*cos(psi)*cos(theta) - u*cos(psi)*sin(theta) + v*cos(psi)*cos(theta)*sin(phi)

def airplane_A_1_9(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return w*(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta)) - v*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta)) - u*cos(theta)*sin(psi)

def airplane_A_1_10(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_1_11(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_1_12(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_2_1(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_2_2(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_2_3(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_2_4(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return cos(theta)*sin(psi)

def airplane_A_2_5(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta)

def airplane_A_2_6(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return cos(phi)*sin(psi)*sin(theta) - cos(psi)*sin(phi)

def airplane_A_2_7(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return - v*(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta)) - w*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta))

def airplane_A_2_8(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return w*cos(phi)*cos(theta)*sin(psi) - u*sin(psi)*sin(theta) + v*cos(theta)*sin(phi)*sin(psi)

def airplane_A_2_9(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return w*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)) - v*(cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(theta)) + u*cos(psi)*cos(theta)

def airplane_A_2_10(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_2_11(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_2_12(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_3_1(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_3_2(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_3_3(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_3_4(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -sin(theta)

def airplane_A_3_5(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return cos(theta)*sin(phi)

def airplane_A_3_6(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return cos(phi)*cos(theta)

def airplane_A_3_7(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return v*cos(phi)*cos(theta) - w*cos(theta)*sin(phi)

def airplane_A_3_8(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return - u*cos(theta) - w*cos(phi)*sin(theta) - v*sin(phi)*sin(theta)

def airplane_A_3_9(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_3_10(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_3_11(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_3_12(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_4_1(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_4_2(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_4_3(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_4_4(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_4_5(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return r

def airplane_A_4_6(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -q

def airplane_A_4_7(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_4_8(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return g*cos(theta)

def airplane_A_4_9(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_4_10(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_4_11(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -w

def airplane_A_4_12(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return v

def airplane_A_5_1(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_5_2(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_5_3(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_5_4(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -r

def airplane_A_5_5(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_5_6(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return p

def airplane_A_5_7(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return g*cos(phi)*cos(theta)

def airplane_A_5_8(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -g*sin(phi)*sin(theta)

def airplane_A_5_9(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_5_10(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return w

def airplane_A_5_11(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_5_12(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -u

def airplane_A_6_1(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_6_2(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_6_3(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_6_4(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return q

def airplane_A_6_5(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -p

def airplane_A_6_6(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_6_7(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -g*cos(theta)*sin(phi)

def airplane_A_6_8(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -g*cos(phi)*sin(theta)

def airplane_A_6_9(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_6_10(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -v

def airplane_A_6_11(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return u

def airplane_A_6_12(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_7_1(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_7_2(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_7_3(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_7_4(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_7_5(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_7_6(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_7_7(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return (q*cos(phi)*sin(theta))/cos(theta) - (r*sin(phi)*sin(theta))/cos(theta)

def airplane_A_7_8(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return r*cos(phi) + q*sin(phi) + (r*cos(phi)*sin(theta)**2)/cos(theta)**2 + (q*sin(phi)*sin(theta)**2)/cos(theta)**2

def airplane_A_7_9(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_7_10(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 1

def airplane_A_7_11(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return (sin(phi)*sin(theta))/cos(theta)

def airplane_A_7_12(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return (cos(phi)*sin(theta))/cos(theta)

def airplane_A_8_1(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_8_2(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_8_3(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_8_4(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_8_5(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_8_6(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_8_7(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return - r*cos(phi) - q*sin(phi)

def airplane_A_8_8(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_8_9(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_8_10(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_8_11(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return cos(phi)

def airplane_A_8_12(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -sin(phi)

def airplane_A_9_1(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_9_2(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_9_3(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_9_4(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_9_5(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_9_6(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_9_7(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return (q*cos(phi))/cos(theta) - (r*sin(phi))/cos(theta)

def airplane_A_9_8(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return (r*cos(phi)*sin(theta))/cos(theta)**2 + (q*sin(phi)*sin(theta))/cos(theta)**2

def airplane_A_9_9(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_9_10(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_9_11(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return sin(phi)/cos(theta)

def airplane_A_9_12(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return cos(phi)/cos(theta)

def airplane_A_10_1(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_10_2(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_10_3(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_10_4(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_10_5(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_10_6(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_10_7(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_10_8(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_10_9(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_10_10(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return (Ix*Ixz*q - Ixz*Iy*q + Ixz*Iz*q)/(Ixz**2 - Ix*Iz)

def airplane_A_10_11(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return (Ixz**2*r + Iz**2*r + Ix*Ixz*p - Ixz*Iy*p + Ixz*Iz*p - Iy*Iz*r)/(Ixz**2 - Ix*Iz)

def airplane_A_10_12(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return (Ixz**2*q + Iz**2*q - Iy*Iz*q)/(Ixz**2 - Ix*Iz)

def airplane_A_11_1(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_11_2(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_11_3(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_11_4(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_11_5(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_11_6(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_11_7(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_11_8(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_11_9(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_11_10(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return (2*Ixz*p - r*(Ix - Iz))/Iy

def airplane_A_11_11(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_11_12(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -(2*Ixz*r + p*(Ix - Iz))/Iy

def airplane_A_12_1(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_12_2(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_12_3(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_12_4(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_12_5(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_12_6(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_12_7(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_12_8(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_12_9(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_A_12_10(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -(Ix**2*q + Ixz**2*q - Ix*Iy*q)/(Ixz**2 - Ix*Iz)

def airplane_A_12_11(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -(Ix**2*p + Ixz**2*p - Ix*Iy*p + Ix*Ixz*r - Ixz*Iy*r + Ixz*Iz*r)/(Ixz**2 - Ix*Iz)

def airplane_A_12_12(x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m):
    return -(Ix*Ixz*q - Ixz*Iy*q + Ixz*Iz*q)/(Ixz**2 - Ix*Iz)

def airplane_B_1_1(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_1_2(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_1_3(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_1_4(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_1_5(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_1_6(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_2_1(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_2_2(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_2_3(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_2_4(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_2_5(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_2_6(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_3_1(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_3_2(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_3_3(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_3_4(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_3_5(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_3_6(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_4_1(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 1/m

def airplane_B_4_2(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_4_3(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_4_4(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_4_5(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_4_6(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_5_1(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_5_2(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 1/m

def airplane_B_5_3(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_5_4(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_5_5(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_5_6(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_6_1(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_6_2(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_6_3(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 1/m

def airplane_B_6_4(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_6_5(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_6_6(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_7_1(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_7_2(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_7_3(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_7_4(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_7_5(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_7_6(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_8_1(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_8_2(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_8_3(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_8_4(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_8_5(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_8_6(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_9_1(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_9_2(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_9_3(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_9_4(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_9_5(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_9_6(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_10_1(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_10_2(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_10_3(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_10_4(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return -Iz/(Ixz**2 - Ix*Iz)

def airplane_B_10_5(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_10_6(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return Ixz/(Ixz**2 - Ix*Iz)

def airplane_B_11_1(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_11_2(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_11_3(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_11_4(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_11_5(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 1/Iy

def airplane_B_11_6(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_12_1(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_12_2(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_12_3(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_12_4(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return Ixz/(Ixz**2 - Ix*Iz)

def airplane_B_12_5(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return 0

def airplane_B_12_6(X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m):
    return -Ix/(Ixz**2 - Ix*Iz)





