syms x1 x2 x3 th1 th2 th3 u1 u2 u3 g mass l c T1 T2 T3 'real'

ex1 = 3 * x1 + 2 * x2 * cos(th1 - th2) + x3 * cos(th1 - th3) + ...
      2 * u1 ^ 2 * sin(th1 - th2) + u3 ^ 2 * sin(th1 - th3) + ...
          - g / l * sin(th1) * 3 + (c * u1 - T1) / (mass * l ^ 2);
      
 
ex2 = 2 * x1 * cos(th1 - th2) + 2 * x2 + x3 * cos(th2 - th3) + ...
      - 2 * u1 ^ 2 * sin(th1 - th2) + u3 ^ 2 * sin(th2 - th3) + ...
      - g / l * sin(th2) * 2 + (c * u2 - T2) / (mass * l ^ 2);

ex3 = x1 * cos(th1 - th3) + x2 * cos(th2 - th3) + x3 + ...
      - u1 ^ 2 * sin(th1 - th3) - u2 ^ 2 * sin(th2 - th3) + ...
      - g / l * sin(th3) + (c * u3 - T3) / (mass * l ^ 2) ;
  
x1s = solve(ex1, x1);
ex2s = subs(ex2, x1, x1s);
ex3s = subs(ex3, x1, x1s);
x2s = solve(ex2s, x2);

ex3ss = subs(ex3s, x2, x2s);
x3ss = solve(ex3ss, x3);
x2ss = subs(x2s, x3, x3ss);
x1ss = subs(x1s, [x2, x3], [x2ss, x3ss]);

fprintMatPy('du1', {'th1', 'th2', 'th3', 'u1', 'u2', 'u3', 'g', 'mass', 'l', 'c', 'T1', 'T2', 'T3'}, x1ss);
fprintMatPy('du2', {'th1', 'th2', 'th3', 'u1', 'u2', 'u3', 'g', 'mass', 'l', 'c', 'T1', 'T2', 'T3'}, x2ss);
fprintMatPy('du3', {'th1', 'th2', 'th3', 'u1', 'u2', 'u3', 'g', 'mass', 'l', 'c', 'T1', 'T2', 'T3'}, x3ss);
