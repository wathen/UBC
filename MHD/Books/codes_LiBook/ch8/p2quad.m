%------------------------------------------------
% Abscissas (xi, eta) and weights (w) for Gaussian 
% integration over our reference triangle 
%
% m=7: order of the quadrature
%------------------------------------------------
function [xi, eta, w] = p2quad(m)

al = 0.797426958353087;
be = 0.470142064105115;
ga = 0.059715871789770;
de = 0.101286507323456;

wt1 = 0.125939180544827;
wt2 = 0.132394152788506;

xi(1) = de;
xi(2) = al;
xi(3) = de;
xi(4) = be;
xi(5) = ga;
xi(6) = be;
xi(7) = 1.0/3.0;

eta(1) = de;
eta(2) = de;
eta(3) = al;
eta(4) = be;
eta(5) = be;
eta(6) = ga;
eta(7) = 1.0/3.0;

w(1) = wt1;  w(2) = wt1;  w(3) = wt1;
w(4) = wt2;  w(5) = wt2;  w(6) = wt2;
w(7) = 0.225;

return;