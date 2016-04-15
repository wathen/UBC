%---------------------------------------------
% source function at the rhs of the equation
% -(u_xx+u_yy)+v_vec*grad u = src
% u=x^2*y^2, v_vec=(1,1), 

function val=SRC (x,y)

val = -2*y^2-2*x^2+2*x*y^2+2*y*x^2;
