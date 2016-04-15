%---------------------------------------------------------------
% hyper1d.m: 
%    use Lax-Wendroff scheme to solve the hyperbolic equation
%     u_t(x,t) + u_x(x,t) = 0,        xl < x < xr, 0 < t < tf
%     u(x, 0) = f(x),                 xl < x < xr
%     u(0, t) = g(t),                 0  < t < tf
% 
% A special case is choosing f and g properly such that the
% The analytic solution is:
%        u(x,t)= f(x-t)=e^(-10(x-t-0.2)^2)
%---------------------------------------------------------------
clear all;                  % clear all variables in memory

xl=0; xr=1;                 % x domain [xl,xr]
J = 40;                     % J: number of division for x
dx = (xr-xl) / J;           % dx: mesh size
tf = 0.5;                   % final simulation time
Nt = 50;                    % Nt: number of time steps
dt = tf/Nt;                 
c = 50;                     % parameter for the solution

mu = dt/dx; 

if mu > 1.0      % make sure dt satisy stability condition
   error('mu should < 1.0!')
end

% Evaluate the initial conditions
x = xl : dx : xr;              % generate the grid point
f = exp(-c*(x-0.2).^2);        % dimension f(1:J+1) 

% store the solution at all grid points for all time steps
u = zeros(J+1,Nt);   

% Find the approximate solution at each time step
for n = 1:Nt
    t = n*dt;                    % current time
    gl = exp(-c*(xl-t-0.2)^2);   % BC at left side
    gr = exp(-c*(xr-t-0.2)^2);   % BC at right side
    if n==1               % first time step
       for j=2:J          % interior nodes     
       u(j,n) = f(j) - 0.5*mu*(f(j+1)-f(j-1)) + ...
                     0.5*mu^2*(f(j+1)-2*f(j)+f(j-1));
       end     
       u(1,n) = gl;       % the left-end point
       u(J+1,n) = gr;     % the  
    else 
       for j=2:J          % interior nodes
         u(j,n) = u(j,n-1) - 0.5*mu*(u(j+1,n-1)-u(j-1,n-1)) + ...
                   0.5*mu^2*(u(j+1,n-1)-2*u(j,n-1)+u(j-1,n-1));
       end
       u(1,n) = gl;       % the left-end point
       u(J+1,n) = gr;     % the right-end point 
    end
    
    % calculate the analytic solution 
    for j=1:J+1
        xj = xl + (j-1)*dx;
        u_ex(j,n)=exp(-c*(xj-t-0.2)^2);
    end
 
end

% plot the analytic and numerical solution at different times
figure;
hold on;
n=10;
plot(x,u(:,n),'r.',x,u_ex(:,n),'r-');   % r for red
n=30;
plot(x,u(:,n),'g.',x,u_ex(:,n),'g-');
n=50;
plot(x,u(:,n),'b.',x,u_ex(:,n),'b-');

legend('Numerical t=0.1','Analytic t=0.1',...
       'Numerical t=0.3','Analytic t=0.3',...
       'Numerical t=0.5','Analytic t=0.5');
title('Numerical and Analytic Solutions at t=0.1, 0.3, 0.5');
