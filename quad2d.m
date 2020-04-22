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
g = 9.81;

dt = 0.01;
plot_limit = 2;
final_eps = 0.05;
max_sim_time = 10;

% goal conditions
xg = [1 1 0 0 0 0]';
ug = m*g*0.5*[1 1]';

% LQR
Q = diag([10 10 90 1 1 r/2/pi]);
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

% for fast linearization:
% A_func = matlabFunction(A_sym)
% B_func = matlabFunction(B_sym)

A = eval(subs(A_sym,[x1 x2 x3 x4 x5 x6 u1 u2],[xg; ug]'));
B = eval(subs(B_sym,[x1 x2 x3 x4 x5 x6 u1 u2],[xg; ug]'));

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
    
    if norm(x-xg) < final_eps
        break;
    end
end


disp('done')
hold off;

