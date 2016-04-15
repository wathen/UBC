%----------------------------------------------------------
% ellip2d.m: solve the Poisson's equation on unit rectangle.
%
% Need 2 functions: exU.m, srcF.m.
%
% O(h^2) convergence rate is observed by using J=5,10,20,40,
% which gives max errors as follows:
% 0.0304, 0.0083, 0.0021, 5.1420e-004.
%-----------------------------------------------------------
clear all;

xl=0; xr=1;    % x domain
yl=0; yr=1;    % y domain

J=80;       % number of points in both x- and y-directions
h = (xr-xl)/J;   % mesh size

% build up the coefficient matrix
nr = (J-1)^2;    % order of the matrix
matA = zeros(nr,nr);

% can set J=3,4,5 etc to check the coefficient matrix
for i=1:nr
    matA(i,i)=4;
    if i+1 <= nr & mod(i,J-1) ~= 0
        matA(i,i+1)=-1;
    end
    if i+J-1 <= nr
        matA(i,i+J-1)=-1;
    end
    if i-1 >= 1 & mod(i-1,J-1) ~= 0
        matA(i,i-1)=-1;
    end
    if i-(J-1) >= 1
       matA(i,i-(J-1))=-1;
    end
end


% build up the right-hand side vector
for j=1:J-1
    y(j) = j*h;
    for i=1:J-1
       x(i) = i*h;
       [fij]=feval(@srcF,x(i),y(j));  % evaluate f(xi,yj)
       vecF((J-1)*(j-1)+i)=h^2*fij;
       [uij]=feval(@exU,x(i),y(j));  % evaluate exact solution
       ExU(i,j)=uij;
    end
end

% solve the system 
vecU = matA\vecF';     % vecU is of order (J-1)^2 

for j=1:J-1
    for i=1:J-1
      U2d(i,j)=vecU((J-1)*(j-1)+i);  % change into 2-D array
  end
end

% display max error so that we can check convergence rate
disp('Max error ='), max(max(U2d-ExU)),   

figure(1);
surf(x,y,U2d);
title('Numerical solution');
figure(2);
surf(x,y,ExU);
title('Exact solution');
