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
plot_limit = 1;
final_eps = 0.05;
max_sim_time = 4;
N = 10; % time-horizon in dt units (N*dt = horizon in seconds)

% nominal condition
x0 = zeros(6,1);
u0 = m*g*0.5*[1 1]';

% initial condition
xs = [1;1;0;0;0;0];
us = u0;

% LQR
Q = diag([10 10 10 1 1 r/2.0/pi]);
R = [0.1 0.05;
     0.05 0.1];
 
%% Dynamics
syms x1 x2 x3 x4 x5 x6 u1 u2

f_func = @(x, u) [x(4); x(5); x(6); -(1/m)*(u(1)+u(2))*sin(x(3)); (1/m)*(u(1)+u(2))*cos(x(3))-m*g; (1/iz)*r*(u(1)-u(2))];
f_sym = f_func([x1 x2 x3 x4 x5 x6],[u1 u2]);


%% Linearize
A_sym = jacobian(f_sym,[x1 x2 x3 x4 x5 x6]);
B_sym = jacobian(f_sym,[u1 u2]);
fd_sym = jacobian(f_sym,[x1 x2 x3 x4 x5 x6 u1 u2]);

A_func = matlabFunction(A_sym,'Vars',[x1 x2 x3 x4 x5 x6 u1 u2]);
B_func = matlabFunction(B_sym,'Vars',[x1 x2 x3 x4 x5 x6 u1 u2]);
fd_func = matlabFunction(fd_sym,'Vars',[x1 x2 x3 x4 x5 x6 u1 u2]);

%% MPC Direct-shooting formulation

x_traj = [xs];
u_traj = [us];
xt = xs;
ut = us;
ts = 0:dt:max_sim_time;

for t = ts
    t
    A = A_func(xt(1),xt(2),xt(3),xt(4),xt(5),xt(6),ut(1),ut(2));
    B = B_func(xt(1),xt(2),xt(3),xt(4),xt(5),xt(6),ut(1),ut(2));
    % discretize my dynamics (one-step Euler approx?)
    %A = (eye(6) + A*dt);
    %B = B*dt;
    
    cvx_begin quiet
        variable u(2,N)
        expressions J x(6,N)
        minimize J
        subject to
            for i = 2:N
                x(:,i) = A*x(:,i-1) + B*u(:,i-1);
            end
            for i = 1:N
                if i == 1
                    J = x(:,i)'*Q*x(:,i)+u(:,i)'*R*u(:,i);
                else
                    J = J + x(:,i)'*Q*x(:,i)+u(:,i)'*R*u(:,i);
                end
            end
            x(:,1) = xt;
    cvx_end
    u
    xt = xt + f_func(xt,u(:,1))*dt;
    x_traj = [x_traj xt];
    u_traj = [u_traj u(:,1)];
end
disp('done');

%% Plot simulation
figure;
hold on;
N = 1:size(ts,2);
xlim([-plot_limit plot_limit]);
ylim([-plot_limit plot_limit]);

qx = x_traj(1,1);
qy = x_traj(2,1);

plot(x_traj(1,1),x_traj(2,1),'gx');

p = plot(qx,qy);
p.XDataSource = 'qx';
p.YDataSource = 'qy';

qpx = [x_traj(1,1)-cos(x_traj(3,1)) x_traj(1)+cos(x_traj(3,1))];
qpy = [x_traj(2,1)-sin(x_traj(3,1)) x_traj(2)+sin(x_traj(3,1))];
qp = plot(qpy, qpx, 'r-');
qp.XDataSource = 'qpx';
qp.YDataSource = 'qpy';
drawnow
plot(x_traj(1,end),x_traj(2,end),'ro');
disp(x_traj(:,end))

for n=N
    % Update simulation data
    qx = x_traj(1,1:n);
    qy = x_traj(2,1:n);
    
    qpx = [x_traj(1,n)-cos(x_traj(3,n)) x_traj(1,n)+cos(x_traj(3,n))];
    qpy = [x_traj(2,n)-sin(x_traj(3,n)) x_traj(2,n)+sin(x_traj(3,n))];
    
    % Update the simulation
    refreshdata
    drawnow
    
    %if norm(x) < final_eps
    %    break;
    %end
end