clear
syms x1 x2;

f = [-x2;
     x1 + (x1^2-1)*x2];
 
A_sym = jacobian(f,[x1 x2]);
Ah_sym = A_sym;
for i = 1:length(f)
    Ah_sym(i,:) = .5*[x1 x2] * hessian(f(i), [x1, x2]);
end

% for fast linearization (use this repeatedly, not useful for single goal
% points)
A_func = matlabFunction(A_sym,'Vars',[x1 x2]);
Ah_func = matlabFunction(Ah_sym,'Vars',[x1 x2]);

A = A_func(0,0) + Ah_func(0,0);

Q = eye(2);
P = lyap(A,Q);

%[K,S] = lqr(A,B,Q,R);
S = P;

%% Region of attraction analysis (note this requires cvx)
rho = 1000;
p = [1;1];
cvx_begin sdp
    variable Q(3,3) semidefinite;
    expressions h(3) m(2) ff(2) z(2)
    subject to
        h = [z; 1]'*Q*[z; 1];
        ff(1) = -z(2);
        ff(2) =  z(1) + (z(1)^2-1)*z(2);
        -2*z'*S*ff - h*(rho - z'*S*z) -eps*z'*z > 0;
cvx_end

%% Region of attraction analysis (using SOSTools)
%vartables = [x1;x2];
%prog = sosprogram(vartables);
%m = monomials([x1; x2],[0 1 2]);