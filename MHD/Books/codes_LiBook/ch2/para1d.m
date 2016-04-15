%---------------------------------------------------------------
% para1d.m: 
%    use the explicit scheme to solve the parabolic equation
%    u_t(x,t) = u_{xx}(x,t),          xl < x < xr, 0 < t < tf
%    u(x,0) = f(x),                   xl < x < xr
%    u(0,t) = gl(t), u(1,t) = gr(t),  0  < t < tf
% 
% A special case is choosing f and g properly such that the
% analytic solution is:
%  u(x,t)= sin(pi*x)*e^(-pi^2*t) + sin(2*pi*x)*e^(-4*pi^2*t)
%
% we solve this program by the explicit scheme:
%    u(j,n+1) = u(j,n) + v*(u(j+1,n) - 2*u(j,n) + u(j-1,n))
%---------------------------------------------------------------
clear all;                  % clear all variables in memory

xl=0; xr=1;                 % x domain [xl,xr]
J = 40;                     % J: number of division for x
dx = (xr-xl) / J;           % dx: mesh size
tf = 0.1;                   % final simulation time
Nt = 50;                    % Nt: number of time steps
dt = tf/Nt/4;                 

mu = dt/(dx)^2; 

if mu > 0.5         % make sure dt satisy stability condition
    error('mu should < 0.5!')
end

% Evaluate the initial conditions
x = xl : dx : xr;              % generate the grid point
% f(1:J+1) since array index starts from 1
f = sin(pi*x) + sin(2*pi*x);  

% store the solution at all grid points for all time steps
u = zeros(J+1,Nt);   

% Find the approximate solution at each time step
for n = 1:Nt
    t = n*dt;         % current time
    % boundary condition at left side
    gl = sin(pi*xl)*exp(-pi*pi*t)+sin(2*pi*xl)*exp(-4*pi*pi*t);   
    % boundary condition at right side
    gr = sin(pi*xr)*exp(-pi*pi*t)+sin(2*pi*xr)*exp(-4*pi*pi*t);  
    if n==1    % first time step
       for j=2:J    % interior nodes     
       u(j,n) = f(j) + mu*(f(j+1)-2*f(j)+f(j-1));
       end     
       u(1,n) = gl;   % the left-end point
       u(J+1,n) = gr; % the right-end point 
    else 
       for j=2:J    % interior nodes
         u(j,n)=u(j,n-1)+mu*(u(j+1,n-1)-2*u(j,n-1)+u(j-1,n-1));
       end
       u(1,n) = gl;   % the left-end point
       u(J+1,n) = gr; % the right-end point 
    end
    
    % calculate the analytic solution 
    for j=1:J+1
        xj = xl + (j-1)*dx;
        u_ex(j,n)=sin(pi*xj)*exp(-pi*pi*t) ...
                 +sin(2*pi*xj)*exp(-4*pi*pi*t);
    end
end

% Plot the results
tt = dt : dt : Nt*dt;
figure(1)
colormap(gray);     % draw gray figure
surf(x,tt, u');     % 3-D surface plot
xlabel('x')
ylabel('t')
zlabel('u')
title('Numerical solution of 1-D parabolic equation')

figure(2)
surf(x,tt, u_ex');     % 3-D surface plot
xlabel('x')
ylabel('t')
zlabel('u')
title('Analytic solution of 1-D parabolic equation')

maxerr=max(max(abs(u-u_ex))),

