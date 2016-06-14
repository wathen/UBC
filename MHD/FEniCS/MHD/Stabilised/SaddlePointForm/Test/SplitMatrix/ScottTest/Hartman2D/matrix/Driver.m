clear all
close all
level = 2;
F = load(strcat('F_',num2str(level)));
F = F.(strcat('F_',num2str(level)));
B = load(strcat('B_',num2str(level)));
B = B.(strcat('B_',num2str(level)));
D = load(strcat('D_',num2str(level)));
D = D.(strcat('D_',num2str(level)));
C = load(strcat('C_',num2str(level)));
C = -C.(strcat('C_',num2str(level)));
M = load(strcat('M_',num2str(level)));
M = M.(strcat('M_',num2str(level)));
Lp = load(strcat('Lp_',num2str(level)));
Lp = Lp.(strcat('Lp_',num2str(level)));
Fs = load(strcat('Fs_',num2str(level)));
Fs = Fs.(strcat('Fs_',num2str(level)));
Fp = load(strcat('Fp_',num2str(level)));
Fp = Fp.(strcat('Fp_',num2str(level)));
Qp = load(strcat('Qp_',num2str(level)));
Qp = Qp.(strcat('Qp_',num2str(level)));
MX = load(strcat('MX_',num2str(level)));
MX = MX.(strcat('MX_',num2str(level)));
Stab = load(strcat('Stab_',num2str(level)));
Stab = Stab.(strcat('Stab_',num2str(level)));

n_u = size(F,1);
m_u = size(B,2);
n_b = size(M,1);
m_b = size(Lp,1);

C = -C';
B = B';
D = D';
K = [F, B', C', sparse(n_u,m_b);
     B, sparse(m_u,m_u+n_b+m_b);
     -C, sparse(n_b,m_u) M D';
     sparse(m_b,n_u+m_b) D Stab];
 
S = B*(F\B');
Kc = [C', sparse(n_u,m_b);sparse(m_u,n_b+m_b)];

Fs = F + C'*((M+D'*(Lp\D))\C);
Km = [M+D'*(Lp\D), 0*D'; 0*D, Lp];
Kns = [Fs, B';0*B,  -S];
P = [Kns, Kc; 0*Kc',Km];

plot(real(eig(full(K), full(P))),'*') 