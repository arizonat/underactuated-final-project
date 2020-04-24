% References: http://underactuated.mit.edu/acrobot.html#section3
% Q, R References: 

%% Reset
close all
clear all

%% General parameters
% state = [x, y, theta, x_d, y_d, theta_d], input = [right, left]

use_discrete = true;

m = 0.486;
r = 0.25;
iz = 0.00383;
g = 0.5;

dt = 0.01;
plot_limit = 1;
final_eps = 0.05;
max_sim_time = 10;

% goal conditions
xg = [0.5 0.5 0 0 0 0]';
ug = m*g*0.5*[1 1]';

% LQR
Q = diag([10 10 10 1 1 r/2/pi]);
R = [0.1 0.05;
     0.05 0.1];

%% Dynamics
syms x1 x2 x3 x4 x5 x6 u1 u2

f = [x4;
     x5;
     x6;
     -(1/m)*(u1+u2)*sin(x3);
     (1/m)*(u1+u2)*cos(x3)-m*g;
     (1/iz)*r*(u1-u2)];
 
f_func = @(x, u) [x(4); x(5); x(6); -(1/m)*(u(1)+u(2))*sin(x(3)); (1/m)*(u(1)+u(2))*cos(x(3))-m*g; (1/iz)*r*(u(1)-u(2))];

%% Linearize
A_sym = jacobian(f,[x1 x2 x3 x4 x5 x6]);
B_sym = jacobian(f,[u1 u2]);

% for fast linearization (use this repeatedly, not useful for single goal
% points)
A_func = matlabFunction(A_sym,'Vars',[x1 x2 x3 x4 x5 x6 u1 u2]);
B_func = matlabFunction(B_sym,'Vars',[x1 x2 x3 x4 x5 x6 u1 u2]);

A = A_func(xg(1),xg(2),xg(3),xg(4),xg(5),xg(6),ug(1),ug(2));
B = B_func(xg(1),xg(2),xg(3),xg(4),xg(5),xg(6),ug(1),ug(2));

%% Discrete-LQR
[Kd Sd] = lqrd(A,B,Q,R,dt);
[K S] = lqr(A,B,Q,R);

%% Simulate
ts = 0:dt:max_sim_time;
x = rand(6,1);
x(1:2) = x(1:2) * 2*plot_limit - plot_limit;
xs = [x];

for t = ts
    % Update dynamics
    if use_discrete
        u = ug-Kd*(x-xg);
    else
        u = ug-K*(x-xg);
    end
    x = x + f_func(x,u) * dt;
    xs = [xs x];
end

%% Plot simulation
figure;
hold on;
N = 1:size(ts,2);
xlim([-plot_limit plot_limit]);
ylim([-plot_limit plot_limit]);

qx = xs(1,1);
qy = xs(2,1);

plot(xs(1,1),xs(2,1),'gx');
plot(xg(1),xg(2),'rx');
plot(xs(1,end),xs(2,end),'ro');

p = plot(qx,qy);
p.XDataSource = 'qx';
p.YDataSource = 'qy';

qpx = [xs(1,1)-cos(xs(3,1)) xs(1)+cos(xs(3,1))];
qpy = [xs(2,1)-sin(xs(3,1)) xs(2)+sin(xs(3,1))];
qp = plot(qpy, qpx, 'r-');
qp.XDataSource = 'qpx';
qp.YDataSource = 'qpy';
drawnow

for n=N
    % Update simulation data
    qx = xs(1,1:n);
    qy = xs(2,1:n);
    
    qpx = [xs(1,n)-cos(xs(3,n)) xs(1,n)+cos(xs(3,n))];
    qpy = [xs(2,n)-sin(xs(3,n)) xs(2,n)+sin(xs(3,n))];
    
    % Update the simulation
    refreshdata
    drawnow
    
    if norm(x-xg) < eps
        break;
    end
end


disp('done')
hold off;

%% Region of attraction analysis (note this requires cvx)
%{
rho = 100000;
p = [1;1;1;1;1;1];
cvx_begin
    variable Q semidefinite
    expressions h(6) z(6) m(6)
    subject to
        m = polyval(p,z);
        h = m'*Q*m;
        2*z'*S*(A*(xg+z)+B*(ug-K*z)) + h*(rho - z'*S*z) <= -eps*z'*z;
cvx_end
%}
%% Region of attraction analysis using SOSTools