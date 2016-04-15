%---------------------------------------------
% source function at the rhs of the equation
% -(u_xx+u_yy)+ u = src 

function val=SRC(x,y)

val = (2*pi^2+1)*cos(pi*x)*cos(pi*y);