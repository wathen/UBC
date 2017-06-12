clear 
clc
close all


level = 3;
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
L = load(strcat('Matrix/L_',num2str(level)));
L = L.('L');
X = load(strcat('Matrix/X_',num2str(level)));
X = X.('X');
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
X(bB+1,:) = [];
X(:,bB+1) = [];
L(rB+1,:) = [];
L(:,rB+1) = [];
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
dimensions(1) = n_u;
dimensions(3) = n_b;
dimensions(2) = m_u;
dimensions(4) = m_b;

alpha = 1;
Km = [M-alpha*Mtilde D';
    D zeros(m_b, m_b)];
Kns = [F+alpha*Ftilde, B';
     B, zeros(m_u, m_u)];
Kct = [C'+alpha*Ctilde', zeros(n_u,m_b);
     zeros(m_u, n_b+m_b)];
Kc = [-C, zeros(n_b, m_u);
     zeros(m_b, n_u+m_u)];
K = [Kns, Kct; Kc, Km];
S1 = Kc*inv(Kns)*Kct;
S = Km - S1;
Mf = S(1:n_b, 1:n_b) + D'*(L\D);

e = eig(full(M-alpha*Mtilde+X), full(Mf));

plot(real(e), 'o')
hold on
plot(imag(e), 'x')
cond(full(M+X)\Mf)
sss














G = Mf\D';
Gt = D/Mf;

% norm(full(G*(C'+alpha*Ctilde')))
norm(full((C'+alpha*Ctilde')*G))
ss

alpha = 0;
Km = [M-alpha*Mtilde D';
    D zeros(m_b, m_b)];
Kns = [F+alpha*Ftilde, B';
     B, zeros(m_u, m_u)];
Kct = [C'+alpha*Ctilde', zeros(n_u,m_b);
     zeros(m_u, n_b+m_b)];
Kc = [-C, zeros(n_b, m_u);
     zeros(m_b, n_u+m_u)];
K = [Kns, Kct; Kc, Km];

S1 = Kc*inv(Kns)*Kct;
S = Km - S1;
Mf = S(1:n_b, 1:n_b) + D'*(L\D);

G1 = Mf\D'; 
Gt1 = D/Mf;


ssss
% norm(full(G*(C'+alpha*Ctilde')))
norm(full((C'+alpha*Ctilde')*G))

H = (speye(n_b) - D'*(L\Gt));
invSS = [Mf\H G/L;
        L\Gt zeros(m_b)];

Mxx = M+D'*inv(L)*D;
S = B*(F\B');
invF = inv(F);
invS = inv(S);
invMx = inv(Mxx);
invL = inv(L);
N = invF-invF*B'*invS*B*invF;

K1 = N;
K2 = invF*B'*invS;
K3 = invS*B*invF;
K4 = -invS;
Chat = C'+alpha*Ctilde';
invCT = -[K1*Chat*inv(Mf)*H K1*Chat*G*invL;
         K3*Chat*inv(Mf)*H K3*Chat*G*invL];
invC  = [inv(Mf)*C*K1 inv(Mf)*C*K2;
         zeros(m_b, n_u+m_u)];
Z = Chat*inv(Mf)*C;
invNS = [K1-K1*Z*K1 K2-K1*Z*K2;
         K3-K3*Z*K1 K4-K3*Z*K2];
invK = [invNS invCT;
        invC invSS];
spy(abs(inv(K))>1e-6)

Mf0 = M + C*K1*C' + D'*(L\D);
e = eig(full(Mf), full(Mf0));
plot(imag(e), 'o')
ssss

    
spy(abs(invS-inv(S))>1e-10)
size(null(full(Mtilde')))
fprintf('%4.0f\n',(length(rB)/4)^2)
close all
stop

SS = inv(S);
norm(full(invS-inv(S)))
invK = inv(K);

ss = invK(n_u+m_u+1:end, n_u+m_u+1:end);
alpha = 1;
A = M-alpha*Mtilde;
Km = [M-alpha*Mtilde D';
    D zeros(m_b, m_b)];

spy(abs(inv(Km))>1e-6)
Ahat = inv(A + D'*(L\D));

Kinv = [Ahat - Ahat*D'*inv(L)*D*Ahat Ahat*D'*inv(L);
        inv(L)*D*Ahat 0*L];
close all

% spy(abs(A*(Ahat*D'))>1e-10);
stop
A = S(1:n_b, 1:n_b);
Z = null(full(D));
V = Z*inv(Z'*A*Z)*Z';
Kinv = [V, (speye(n_b) - V*A)*D'*inv(D*D');
        inv(D*D')*D*(speye(n_b) - A*V), -inv(D*D')*D*(A - A*V*A)*D'*inv(D*D')];
figure
spy(abs(Kinv-SS)>1e-10)


% figure
% alpha = 1;
% A = M-alpha*Mtilde;
% Km1 = [M-alpha*Mtilde D';
%     D zeros(m_b, m_b)];
% 
% P = [M-alpha*Mtilde + D'*(L\D) ,0*D';
%     0*D L];
% 
% alpha = 0;
% A = M-alpha*Mtilde;
% Km0 = [M-alpha*Mtilde D';
%     D zeros(m_b, m_b)];
% 
% e = eig(full(Km1), full(P));
% 
% plot(sort(real(e)), '*')
% hold on 
% plot(sort(imag(e)))


% Null = null(full(M));
% size(Null)
% norm((M+Mtilde)*Null)
% 
% nullC = null(full(Ctilde));
% 
% norm(C*nullC)
% alpha = 1;
% K = full([F+alpha*Ftilde, B', C'+alpha*Ctilde', zeros(n_u,m_b);
%     B, zeros(m_u,m_u+n_b+m_b);
%      -C, zeros(n_b,m_u) M-alpha*Mtilde D';
%      zeros(m_b,n_u+m_b) D zeros(m_b,m_b)]);
% alpha = 0;
% K1 = full([F+alpha*Ftilde, B', C'+alpha*Ctilde', zeros(n_u,m_b);
%     B, zeros(m_u,m_u+n_b+m_b);
%      -C-alpha*Ctilde, zeros(n_b,m_u) M-alpha*Mtilde D';
%      zeros(m_b,n_u+m_b) D zeros(m_b,m_b)]); 
%  
%  e = eig(K, K1);
% %  plot(sort(real(e)), '*')
% 
% Maxwell = [M-alpha*Mtilde D'; D zeros(m_b,m_b)];
% spy(abs(inv(Maxwell))>1e-6)
% NullM = M-alpha*Mtilde;
% norm(full(Mtilde*NullM))
% size(null(full(alpha*Mtilde)))
% size(null(full(M)))
% size(null(full(M+alpha*Mtilde)))
% 




% fprintf('norm(full(F))         = %4.4f\n', norm(full(F)))
% fprintf('norm(full(M))         = %4.4f\n', norm(full(M)))
% fprintf('norm(full(C))         = %4.4f\n\n', norm(full(C)))
% 
% fprintf('norm(full(Ftilde))    = %4.4f\n', norm(full(Ftilde)))
% fprintf('norm(full(Mtilde))    = %4.4f\n', norm(full(Mtilde)))
% fprintf('norm(full(Ctilde))    = %4.4f\n\n', norm(full(Ctilde)))
% 
% fprintf('norm(full(F+Ftilde))  = %4.4f\n', norm(full(F+Ftilde)))
% fprintf('norm(full(M+Mtilde))  = %4.4f\n', norm(full(M+Mtilde)))
% fprintf('norm(full(C+Ctilde))  = %4.4f\n\n', norm(full(C+Ctilde)))
% 
% Km = [M+alpha*Mtilde D';
%     D zeros(m_b, m_b)];
% Kns = [F+alpha*Ftilde, B';
%      B, zeros(m_u, m_u)];
% Kct = [C'+alpha*Ctilde', zeros(n_u,m_b);
%      zeros(m_u, n_b+m_b)];
% Kc = [-C, zeros(n_b, m_u);
%      zeros(m_b, n_u+m_u)];
% K = [Kns, Kct; Kc, Km];
% fprintf('norm(full(K_a1))      = %4.4f\n', norm(full(K)))
% fprintf('cond(full(K_a1))      = %4.4f\n', cond(full(K)))
% 
% alpha = 0;
% Km = [M+alpha*Mtilde D';
%     D zeros(m_b, m_b)];
% Kns = [F+alpha*Ftilde, B';
%      B, zeros(m_u, m_u)];
% Kct = [C'+alpha*Ctilde', zeros(n_u,m_b);
%      zeros(m_u, n_b+m_b)];
% Kc = [-C, zeros(n_b, m_u);
%      zeros(m_b, n_u+m_u)];
% K = [Kns, Kct; Kc, Km];
% fprintf('norm(full(K_a0))      = %4.4f\n', norm(full(K)))
% fprintf('cond(full(K_a0))      = %4.4f\n', cond(full(K)))
