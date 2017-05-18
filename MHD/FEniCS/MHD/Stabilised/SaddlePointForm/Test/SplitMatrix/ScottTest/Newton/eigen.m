clear 
clc
level = 2;
dimensions = load(strcat('Matrix/dim_',num2str(level),'.mat'));
dimensions = dimensions.('bcr');

n_u = dimensions(1);
n_b = dimensions(3);
m_u = dimensions(2);
m_b = dimensions(4);
% 
F = load(strcat('Matrix/F_',num2str(level)));
F = F.('F');
B = load(strcat('Matrix/B_',num2str(level)));
B = B.('B');
M = load(strcat('Matrix/M_',num2str(level)));
M = M.('M');
D = load(strcat('Matrix/D_',num2str(level)));
D = D.('D');
C = load(strcat('Matrix/C_',num2str(level)));
C = C.('C');
Ftilde = load(strcat('Matrix/Ftilde_',num2str(level)));
Ftilde = Ftilde.('Ftilde');
Mtilde = load(strcat('Matrix/Mtilde_',num2str(level)));
Mtilde = Mtilde.('Mtilde');
Ctilde = load(strcat('Matrix/Ctilde_',num2str(level)));
Ctilde = Ctilde.('Ctilde');

% A = load(strcat('Matrix/A',num2str(level)));
% A = A.(strcat('A',num2str(level)));
% O = load(strcat('Matrix/O',num2str(level)));
% O = O.(strcat('O',num2str(level)));
% B = load(strcat('Matrix/B',num2str(level)));
% B = B.(strcat('B',num2str(level)));
% D = load(strcat('Matrix/D',num2str(level)));
% D = D.(strcat('D',num2str(level)));
% C = load(strcat('Matrix/C',num2str(level)));
% C = -C.(strcat('C',num2str(level)));
% M = load(strcat('Matrix/M',num2str(level)));
% M = M.(strcat('M',num2str(level)));
% L = load(strcat('Matrix/L',num2str(level)));
% L = L.(strcat('L',num2str(level)));
% X = load(strcat('Matrix/X',num2str(level)));
% X = X.(strcat('X',num2str(level)));
% Xs = load(strcat('Matrix/Xs',num2str(level)));
% Xs = Xs.(strcat('Xs',num2str(level)));
% Qs = load(strcat('Matrix/Qs',num2str(level)));
% Qs = Qs.(strcat('Qs',num2str(level)));
% Q = load(strcat('Matrix/Q',num2str(level)));
% Q = Q.(strcat('Q',num2str(level)));
% Fp = load(strcat('Matrix/Fp',num2str(level)));
% Fp = Fp.(strcat('Fp',num2str(level)));
% Mp = load(strcat('Matrix/Mp',num2str(level)));
% Mp = Mp.(strcat('Mp',num2str(level)));


uB = load(strcat('Matrix/bcu_',num2str(level),'.mat'));
uB = uB.('bcu');
bB = load(strcat('Matrix/bcb_',num2str(level),'.mat'));
bB = bB.('bcb');
rB = load(strcat('Matrix/bcr_',num2str(level),'.mat'));
rB = rB.('bcr');
B = B';
D = D';
Ctilde = Ctilde';
F(uB+1,:) = [];
F(:,uB+1) = [];
B(rB+1,:) = [];
B(:,uB+1) = [];
M(bB+1,:) = [];
M(:,bB+1) = [];
D(rB+1,:) = [];
D(:,bB+1) = [];
C(bB+1,:) = [];
C(:,uB+1) = [];
Ctilde(bB+1,:) = [];
Ctilde(:,uB+1) = [];
Mtilde(bB+1,:) = [];
Mtilde(:,bB+1) = [];
Ftilde(uB+1,:) = [];
Ftilde(:,uB+1) = [];
n_u = n_u - length(uB);
n_b = n_b - length(bB);
m_u = m_u - length(rB);
m_b = m_b - length(rB);


Null = null(full(M));
size(Null)
norm((M+Mtilde)*Null)

nullC = null(full(Ctilde));

norm(C*nullC)


