clear all
close all
level = 2;

load('dim')
A = load(strcat('A_',num2str(level)));
A = A.(strcat('A_',num2str(level)));
b = load(strcat('b_',num2str(level)));
b = b.(strcat('b_',num2str(level)));
nu = dim(level,1);
mu = dim(level,2);
nb = dim(level,3);
mb = dim(level,4);

NS = A(1:nu+mu, 1:nu+mu);
MX = A(nu+mu+1:end,nu+mu+1:end);
F = A(1:nu, 1:nu);

spy(abs(A)>1e-5)