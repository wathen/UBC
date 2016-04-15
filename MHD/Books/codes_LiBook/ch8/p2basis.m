%------------------------------------------------------
% computation of the basis functions and their gradients
% over a 6-node triangle.
%-------------------------------------------------------
function [psi, gpsi, jac] = p2basis ...
   (x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,xi,eta)
                                        
% compute the basis functions
psi(2) = xi*(2.0*xi-1.0);
psi(3) = eta*(2.0*eta-1.0);
psi(4) = 4.0*xi*(1.0-xi-eta);
psi(5) = 4.0*xi*eta;
psi(6) = 4.0*eta*(1.0-xi-eta);
psi(1) = 1.0-psi(2)-psi(3)-psi(4)-psi(5)-psi(6);
      
% compute xi derivatives of the basis functions
dps(2) =  4.0*xi-1.0;
dps(3) =  0.0;
dps(4) =  4.0*(1.0-xi-eta)-4.0*xi;
dps(5) =  4.0*eta;
dps(6) = -4.0*eta;
dps(1) = -dps(2)-dps(3)-dps(4)-dps(5)-dps(6);

% compute eta derivatives of the basis functions
pps(2) =  0.0;
pps(3) =  4.0*eta-1.0;
pps(4) = -4.0*xi;
pps(5) =  4.0*xi;
pps(6) =  4.0*(1.0-xi-eta)-4.0*eta;
pps(1) = -pps(2)-pps(3)-pps(4)-pps(5)-pps(6);

% compute the xi and eta derivatives of x
DxDxi = x1*dps(1) + x2*dps(2) + x3*dps(3) ...
      + x4*dps(4) + x5*dps(5) + x6*dps(6);
DyDxi = y1*dps(1) + y2*dps(2) + y3*dps(3) ...
      + y4*dps(4) + y5*dps(5) + y6*dps(6);

DxDeta = x1*pps(1) + x2*pps(2) + x3*pps(3) ...
       + x4*pps(4) + x5*pps(5) + x6*pps(6);
DyDeta = y1*pps(1) + y2*pps(2) + y3*pps(3) ...
       + y4*pps(4) + y5*pps(5) + y6*pps(6);

% compute the determinant of Jacobi matrix
jac = abs(DxDxi * DyDeta - DxDeta * DyDxi);

% compute the gradient of the basis functions
A11 = DxDxi;  A12 = DyDxi;
A21 = DxDeta; A22 = DyDeta;

Det = A11*A22-A21*A12;

for k=1:6
   B1 = dps(k); B2 = pps(k);
   Det1 = B1*A22 - B2*A12; Det2 = - B1*A21 + B2*A11;
   gpsi(k,1) = Det1/Det; gpsi(k,2) = Det2/Det;
end

return;