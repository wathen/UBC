%-------------------------------------------------------------
% calculate the element mass matrix for P1 element
%-------------------------------------------------------------
function [elmM] = locM(x1,y1,x2,y2,x3,y3)

 dx23 = x2-x3; dy23 = y2-y3;
 dx31 = x3-x1; dy31 = y3-y1;
 dx12 = x1-x2; dy12 = y1-y2;

 A = 0.5*(dx31*dy12 - dy31*dx12);
 c_diag = A/6;   % diagnonal constant
 c_off = A/12;   % off-diagnonal constant
 
 for j=1:3
     for i=1:3
         if(i==j)
            elmM(i,j) = c_diag;
        else
            elmM(i,j) = c_off;
        end
    end
end
 
return;