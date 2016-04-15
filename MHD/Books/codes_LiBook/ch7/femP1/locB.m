%----------------------------------------------------
% calculate the element matrix from convection term
%----------------------------------------------------
function [elmB] = locB(vx,vy,x1,y1,x2,y2,x3,y3)

 dx32 = x3-x2; dy32 = y3-y2;
 dx13 = x1-x3; dy13 = y1-y3;
 dx21 = x2-x1; dy21 = y2-y1;

 elmB(1,1) = (-vx*dy32 + vy*dx32)/6.0;
 elmB(1,2) = (-vx*dy13 + vy*dx13)/6.0;
 elmB(1,3) = (-vx*dy21 + vy*dx21)/6.0;

 elmB(2,1) = elmB(1,1);   elmB(3,1) = elmB(1,1);
 elmB(2,2) = elmB(1,2);   elmB(3,2) = elmB(1,2);
 elmB(2,3) = elmB(1,3);   elmB(3,3) = elmB(1,3);
 
return;
