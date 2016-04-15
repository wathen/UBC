% The RHS function f for the governing PDE
function uu=srcF(x,y)

uu=2*pi^2*sin(pi*x)*sin(pi*y);
