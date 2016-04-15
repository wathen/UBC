%-----------------------------------------------
% Evaluation of element matrix D_u for a 6-node 
% triangle using Gauss integration quadrature
%-----------------------------------------------
function [edm, arel] = elmD ...
   (x1,y1, x2,y2, x3,y3, x4,y4, x5,y5, x6,y6, NQ)

% read the triangle quadrature
[xi, eta, w] = p2quad(NQ);

edm=zeros(6,6);   % initilization

% perform the quadrature
arel = 0.0;  % element area 

for i=1:NQ
 [psi, gpsi, jac] = p2basis ...
   (x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,xi(i),eta(i));

 cf = 0.5*jac*w(i);
 for k=1:6
  for l=1:6
   edm(k,l) = edm(k,l) + (gpsi(k,1)*gpsi(l,1)   ...
                       +  gpsi(k,2)*gpsi(l,2) )*cf;
  end
 end

 arel = arel + cf;
end

return;
