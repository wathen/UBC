%------------------------------------------------
% the residual R=-0.5*(u^2)_x-cc*(u_xxx)
% i.e., u1=(u^2)_x, u2=u_xxx
function uu=resJSC1(u1,u2,cc)
uu=-cc*u2;


