%------------------------------------------------------
% Calculate the exact solution of the given PDE

function u2d=F2dcone(x,y,t)
ID=4;
if ID==1
  rt=(2+sin(pi*t))/4;
  st=(2+cos(pi*t))/4;
  u2d=0.8*exp(-80*((x-rt)^2+(y-st)^2));
elseif ID==2
  u2d=exp(-8*pi*pi*t)*sin(2*pi*x)*sin(2*pi*y);
 elseif ID==3
  u2d=exp(-t)*sin(4*pi*x)*sin(4*pi*y);
 elseif ID==4
  u2d=exp(-t)*(sin(4*pi*x)+sin(4*pi*y));
end
