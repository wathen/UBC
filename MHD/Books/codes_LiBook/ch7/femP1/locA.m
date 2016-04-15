%-------------------------------------------------------
% evaluate the element diffusion matrix for P1 element
%-------------------------------------------------------
function [elmA] = locA(x1,y1,x2,y2,x3,y3)

dx23 = x2-x3; dy23 = y2-y3;
dx31 = x3-x1; dy31 = y3-y1;
dx12 = x1-x2; dy12 = y1-y2;

A = 0.5*(dx31*dy12 - dy31*dx12);  % triangle area

elmA(1,1) = 0.25*(dx23*dx23 + dy23*dy23)/A;
elmA(1,2) = 0.25*(dx23*dx31 + dy23*dy31)/A;
elmA(1,3) = 0.25*(dx23*dx12 + dy23*dy12)/A;

elmA(2,1) = 0.25*(dx31*dx23 + dy31*dy23)/A;
elmA(2,2) = 0.25*(dx31*dx31 + dy31*dy31)/A;
elmA(2,3) = 0.25*(dx31*dx12 + dy31*dy12)/A;

elmA(3,1) = 0.25*(dx12*dx23 + dy12*dy23)/A;
elmA(3,2) = 0.25*(dx12*dx31 + dy12*dy31)/A;
elmA(3,3) = 0.25*(dx12*dx12 + dy12*dy12)/A;

return;