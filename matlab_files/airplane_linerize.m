syms x y z u v w phi theta psi p q r
syms X Y Z M N L
syms Ix Iy Iz Ixz g m 
T_psi = [cos(psi), -sin(psi), 0.;
         sin(psi),  cos(psi), 0.;
         0.,          0.,     1.];
T_theta = [cos(theta), 0., sin(theta);
           0.,         1.,     0.;
          -sin(theta), 0., cos(theta)];
T_phi = [1., 0.,        0.;
         0., cos(phi), -sin(phi);
         0., sin(phi),  cos(phi)];

mat_1 = T_psi*T_theta*T_phi;
mat_2 = [cos(theta), sin(theta) * sin(phi),  sin(theta) * cos(phi);
         0.,         cos(theta) * cos(phi), -cos(theta) * sin(phi);
         0.,                      sin(phi),               cos(phi)];
mat_2 = 1 / cos(theta) * mat_2;
 
a1 = [u; v; w];
a2 = mat_1*a1;
a3 = [p; q; r];
a4 = mat_2*a3;
a5 = [Ix,  Ixz;
      Ixz, Iz];
a6 = [L - (Iz - Iy) * q * r - Ixz * q * p;
      N - (Iy - Ix) * q * p + Ixz * q * r];
a7 = a5 \ a6;
dx = a2(1);
dy = a2(2);
dz = a2(3);

du = g * sin(theta) + X / m - q * w + r * v;
dv = g * cos(theta) * sin(phi) + Y / m - r * u + p * w;
dw = g * cos(theta) * cos(phi) + Z / m - p * v + q * u;

dphi = a4(1);
dthe = a4(2);
dpsi = a4(3);

dp = a7(1);
dq = 1. / Iy *(M - Ixz * (r^2 - p^2) - (Ix - Iz) * p * r);
dr = a7(2);

all_vars = [x y z u v w phi theta psi p q r];
all_acts = [X Y Z L M N];
all_dvars = [dx dy dz du dv dw dphi dthe dpsi dp dq dr];
A = sym('A', [length(all_dvars)  length(all_vars)]);
B = sym('B', [length(all_dvars)  length(all_acts)]);
for i = 1:length(all_dvars)
    dvar = all_dvars(i);
    for j = 1:length(all_vars)
        var = all_vars(j);
        A(i, j) = diff(dvar, var);
    end
    
    for j = 1:length(all_acts)
        act = all_acts(j);
        B(i, j) = diff(dvar, act);
    end
end

fprintMatPy('A', {'x', 'y', 'z', 'u', 'v', 'w', 'phi', 'theta',...
                             'psi', 'p', 'q', 'r', 'Ix', 'Iy', 'Iz', 'Ixz', 'g', 'm'}, A);

fprintMatPy('B', {'X', 'Y', 'Z', 'L', 'M', 'N', 'Ix', 'Iy', 'Iz', 'Ixz', 'g', 'm'}, B);