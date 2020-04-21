% References: http://underactuated.mit.edu/acrobot.html#section3
% Q, R References: 

%% Reset
close all
clear all

%% General parameters
% state = [x, y, theta, x_d, y_d, theta_d], input = [right, left]

m = 0.486;
r = 0.25;
iz = 0.00383;
g = 2;
dt = 0.1;

% initial conditions
x0 = [0 10 0 0 0 0];
u0 = m*g*0.5*[1 1];

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
 
%% Linearize
A_sym = jacobian(f,[x1 x2 x3 x4 x5 x6]);
B_sym = jacobian(f,[u1 u2]);

A = eval(subs(A_sym,[x1 x2 x3 x4 x5 x6 u1 u2],[x0 u0]));
B = eval(subs(B_sym,[x1 x2 x3 x4 x5 x6 u1 u2],[x0 u0]));

%% LQR
K = lqrd(A,B,Q,R,dt);

%% Simulation
ts = 0:dt:15;
x = rand(6,1);
xs = [x];

for t = ts
    u = -K*x;
    xd = A*x + B*u;
    x = x + xd * dt;
    xs = [xs x];
end

disp(x)
figure;
hold on;
plot(xs(1,1),xs(2,1),'gx');
plot(xs(1,end),xs(2,end),'ro');
plot(xs(1,:),xs(2,:))
hold off;