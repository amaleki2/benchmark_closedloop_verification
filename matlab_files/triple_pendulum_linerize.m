% syms t m l th1 th2 th3 u1 u2 u3 g 
% 
% T = m*l^2*(3/2*u1^2 + u2^2 + 1/2*u3^2 + 2*cos(th1-th2)*u1*u2 + ...
%     cos(th1-th3)*u1*u3 + cos(th2-th3)*u2*u3);
% U = m*g*l*(cos(th1)*3 + cos(th2)*2 + cos(th3));
% L = T - U;
% 
% dLdth1 = diff(L, th1);
% dLdth2 = diff(L, th2);
%  
% 
% dLdth3 = diff(L, th3);
% dLdu1 = diff(L, u1);
% dLdu2 = diff(L, u2);
% dLdu3 = diff(L, u3);
% 


syms th1 th2 th3 u1 u2 u3 T1 T2 T3 m g c L

M = [3.0, 2.0 * cos(th1 - th2), 1.0 * cos(th1 - th3); 
     2.0 * cos(th1 - th2), 2.0, 1.0 * cos(th2 - th3);
     1.0 * cos(th1 - th3), 1.0 * cos(th2 - th3), 1.0];
 
C = [ 2.0 * sin(th1 - th2) * u2^2 + 1.0 * sin(th1 - th3) * u3^2 ;
     -2.0 * sin(th1 - th2) * u1^2 + 1.0 * sin(th2 - th3) * u3^2 ;
     -1.0 * sin(th1 - th3) * u1^2 - 1.0 * sin(th1 - th3) * u3^2 ];
C = -C;

G = g/L * [sin(th1) * 3 ;
           sin(th2) * 2 ;
           sin(th3) * 1 ];
       
T =  1/(m*L^2) * [T1; T2; T3];
F = -c/(m*L^2) * [u1; u2; u3];

aMat = M\(T + F + C + G);

q0 = diff(aMat, th1);
q1 = diff(aMat, th2);
q2 = diff(aMat, th3);
q3 = diff(aMat, u1);
q4 = diff(aMat, u2);
q5 = diff(aMat, u3);
q6 = diff(aMat, T1);
q7 = diff(aMat, T2);
q8 = diff(aMat, T3);

A = [0, 0, 0, 1, 0, 0;
     0, 0, 0, 0, 1, 0;
     0, 0, 0, 0, 0, 1;
     q0(1), q1(1), q2(1), q3(1), q4(1), q5(1);
     q0(2), q1(2), q2(2), q3(2), q4(2), q5(2);
     q0(3), q1(3), q2(3), q3(3), q4(3), q5(3)
     ];
 
B = [0, 0, 0;
     0, 0, 0;
     0, 0, 0;
     q6(1), q7(1), q8(1);
     q6(2), q7(2), q8(2);
     q6(3), q7(3), q8(3)];
 
%A(4, 1)
 
fprintMatPy('A_funcs', {'th1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A)

%fprintMatPy('B_funcs', {'th1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, B)

% fprintMatPy('A40', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(5,1))
% fprintMatPy('A50', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(6,1))
% fprintMatPy('A31', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(4,2))
% fprintMatPy('A41', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(5,2))
% fprintMatPy('A51', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(6,2))
% fprintMatPy('A32', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(4,3))
% fprintMatPy('A42', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(5,3))
% fprintMatPy('A52', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(6,3))
% fprintMatPy('A33', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(4,4))
% fprintMatPy('A43', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(5,4))
% fprintMatPy('A53', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(6,4))
% fprintMatPy('A34', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(4,5))
% fprintMatPy('A44', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(5,5))
% fprintMatPy('A54', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(6,5))
% fprintMatPy('A35', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(4,6))
% fprintMatPy('A45', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(5,6))
% fprintMatPy('A55', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, A(6,6))
% 
% fprintMatPy('B30', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, B(4,1))
% fprintMatPy('B40', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, B(5,1))
% fprintMatPy('B50', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, B(6,1))
% fprintMatPy('B31', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, B(4,2))
% fprintMatPy('B41', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, B(5,2))
% fprintMatPy('B51', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, B(6,2))
% fprintMatPy('B32', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, B(4,3))
% fprintMatPy('B42', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, B(5,3))
% fprintMatPy('B52', {'ht1', 'th2', 'th3', 'u1', 'u2', 'u3', 'T1', 'T2', 'T3', 'm', 'g', 'L', 'c'}, B(6,3))
