syms x1 x2 th1 th2 u1 u2 g mass l c T1 T2 'real'


ex1 = 2 * x1 + x2 * cos(th2 - th1) - u2 ^ 2 * sin(th2 - th1) + ...
          - 2 * g / l * sin(th1) + (c * u1 - T1) / (mass * l ^ 2);
     
ex2 = x1 * cos(th2 - th1) + x2 + u1 ^ 2 * sin(th2 - th1) + ...
      - g / l * sin(th2) + (c * u2 - T2) / (mass * l ^ 2);
  
x1s = solve(ex1, x1);
ex2s = subs(ex2, x1, x1s);
x2ss = solve(ex2s, x2);
x1ss = subs(x1s, x2, x2ss);

x1ss_sub = subs(x1ss, [mass, g, l, c], [0.5, 1, 0.5, 0]);
%disp(x1ss_sub);
disp(simplify(x1ss_sub));


x2ss_sub = subs(x2ss, [mass, g, l, c], [0.5, 1, 0.5, 0]);
%disp(x2ss_sub);
disp(simplify(x2ss_sub));

% fprintMatPy('du1', {'th1', 'th2', 'u1', 'u2', 'g', 'mass', 'l', 'c', 'T1', 'T2'}, x1ss);
% fprintMatPy('du2', {'th1', 'th2', 'u1', 'u2', 'g', 'mass', 'l', 'c', 'T1', 'T2'}, x2ss);

v1 = sin(th1);
v2 = sin(th2);
v3 = sin(th1 - th2);
v4 = cos(th1 - th2);
v5 = u1^2*v3;
v6 = u2^2*v3;
v7 = sin(th1-th2*2);
v8 = (8*T1 + 3*v1 -v4*(8*T2 + v5) + v7 - v6)/ (2-v4^2);
v9 = (2*v3*u1^2 +16*T2 +4*v2 -v4*(8*T1 - v3*u2^2 +4*v1))/(2-v4^2);

for i = 1:100
    vals = rand(6,1)*5-2.5;
    a1 = subs(v8, [th1, th2, u1, u2, T1, T2], vals');
    a2 = subs(x1ss_sub, [th1, th2, u1, u2, T1, T2], vals');
    a3 = abs(eval(a1)-eval(a2));
    assert(abs(a3)<1E-6);
end

for i = 1:100
    vals = rand(6,1)*5-2.5;
    a4 = subs(v9, [th1, th2, u1, u2, T1, T2], vals');
    a5 = subs(x2ss_sub, [th1, th2, u1, u2, T1, T2], vals');
    a6 = abs(eval(a4)-eval(a5));
    assert(abs(a3)<1E-6);
end

max_1 =-inf; min_1 = inf
for i = 1:10000
    if mod(i,100)==1, disp(i); end
    vals = rand(6,1)*2-1;
    a2 = subs(x1ss_sub, [th1, th2, u1, u2, T1, T2], vals');
    max_1 = max(max_1, a2);
    min_1 = min(min_1, a2);
end
% 
% for i = 1:1
%     vals = rand(6,1)*5-5;
%     a4 = subs(v9, [th1, th2, u1, u2, T1, T2], vals');
%     a5 = subs(x2ss_sub, [th1, th2, u1, u2, T1, T2], vals');
%     a6 = abs(eval(a4)-eval(a5));
%     assert(abs(a3)<1E-6);
% end