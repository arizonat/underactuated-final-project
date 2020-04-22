% References: http://underactuated.mit.edu/acrobot.html#section3
% Q, R References: 

%% Reset
close all
clear all

%% General parameters
% state = [x, y, theta, x_d, y_d, theta_d], input = [right, left]

use_discrete = false;

m = 0.486;
r = 0.25;
iz = 0.00383;
g = 9.81;

dt = 0.01;
plot_limit = 2;
final_eps = 0.05;
max_sim_time = 4;

% nominal conditions
x0 = [0 0 0 0 0 0];
u0 = m*g*0.5*[1 1];

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

A = eval(subs(A_sym,[x1 x2 x3 x4 x5 x6 u1 u2],[x0 u0]));
B = eval(subs(B_sym,[x1 x2 x3 x4 x5 x6 u1 u2],[x0 u0]));

%% Discrete-LQR
[Kd Sd] = lqrd(A,B,Q,R,dt);
[K S] = lqr(A,B,Q,R);

%% Slow Simulation
figure;
hold on;
xlim([-plot_limit plot_limit]);
ylim([-plot_limit plot_limit]);

ts = 0:dt:max_sim_time;
x = rand(6,1);
x(1:2) = x(1:2) * plot_limit;
xs = [x];
qx = x(1);
qy = x(2);

plot(x(1,1),x(2,1),'gx');

p = plot(qx,qy);
p.XDataSource = 'qx';
p.YDataSource = 'qy';

qpx = [x(1)-cos(x(3)) x(1)+cos(x(3))];
qpy = [x(2)-sin(x(3)) x(2)+sin(x(3))];
qp = plot(qpy, qpx, 'r-');
qp.XDataSource = 'qpx';
qp.YDataSource = 'qpy';

for t = ts
    % Update dynamics
    if use_discrete
        u = -Kd*x;
        xd = A*x + B*u;
    else
        u = -K*x;
        xd = f_func(x,u);
    end
    x = x + xd * dt;
    xs = [xs x];
    
    % Update simulation data
    qx = xs(1,:);
    qy = xs(2,:);
    
    qpx = [x(1)-cos(x(3)) x(1)+cos(x(3))];
    qpy = [x(2)-sin(x(3)) x(2)+sin(x(3))];
    
    % Update the simulation
    refreshdata
    drawnow
    %pause(dt);
    
    if norm(x) < final_eps
        break;
    end
end

plot(xs(1,end),xs(2,end),'ro');
disp('done')
hold off;

