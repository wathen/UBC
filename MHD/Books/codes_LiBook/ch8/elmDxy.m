%-----------------------------------------------------
% Computation of the matrices D^x and D^y
%-----------------------------------------------------
function [Dx, Dy] = elmDxy ...
   (x1,y1, x2,y2, x3,y3, x4,y4, x5,y5, x6,y6, NQ)

% read the triangle quadrature
[xi, eta, w] = p2quad(NQ);

Dx = zeros(1,6); Dy = zeros(1,6);   %initilization

% perform the quadrature
for i=1:NQ
  [psi, gpsi, jac] = p2basis ...
    (x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,xi(i),eta(i));

 cf = 0.5*jac*w(i);
 for k=1:6
   Dx(k) = Dx(k) + gpsi(k,1)*cf;
   Dy(k) = Dy(k) + gpsi(k,2)*cf;
 end
end

return;