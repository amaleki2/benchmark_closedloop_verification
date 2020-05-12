syms th1 th2 u1 u2 T1 T2 m g c L

M = [2, cos(th2 - th1); 
     cos(th2 - th1), 2];
 
C = [ sin(th2-th1)*u2^2;
     -sin(th2-th1)*u1^2];

G = g/L * [sin(th1)*2; 
           sin(th2)];
       
T = 1/(m*L^2) * [T1;
                 T2];

F = -c/(m*L^2) * [u1;
                  u2];

aMat = M\(T + F + C + G);

q0 = diff(aMat, th1);
q1 = diff(aMat, th2);
q2 = diff(aMat, u1);
q3 = diff(aMat, u2);
q4 = diff(aMat, T1);
q5 = diff(aMat, T2);

A = [0, 0, 1, 0;
     0, 0, 0, 1;
     q0(1), q1(1), q2(1), q3(1);
     q0(2), q1(2), q2(2), q3(2)];
 
B = [0, 0;
     0, 0;
     q4(1), q5(1);
     q4(2), q5(2)];
 
A
B
 