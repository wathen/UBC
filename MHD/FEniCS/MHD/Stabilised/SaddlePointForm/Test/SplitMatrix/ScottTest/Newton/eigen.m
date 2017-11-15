clear 
clc
close all


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
XX = load(strcat('Matrix/XX_',num2str(level)));
XX = XX.('XX');
A = load(strcat('Matrix/A_',num2str(level)));
A = A.('A');
W = load(strcat('Matrix/W_',num2str(level)));
W = W.('W');
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
G = load(strcat('Matrix/G_',num2str(level)));
G = G.('G');
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
Qp = load(strcat('Matrix/Qp_',num2str(level)));
Qp = Qp.('Qp');


uB = load(strcat('Matrix/bcu_',num2str(level),'.mat'));
uB = uB.('bcu');
bB = load(strcat('Matrix/bcb_',num2str(level),'.mat'));
bB = bB.('bcb');
rB = load(strcat('Matrix/bcr_',num2str(level),'.mat'));
rB = rB.('bcr');


B = B';
D = D';
G = G';
Ctilde = Ctilde';

F(uB+1,:) = [];
F(:,uB+1) = [];
XX(uB+1,:) = [];
XX(:,uB+1) = [];
A(uB+1,:) = [];
A(:,uB+1) = [];
W(uB+1,:) = [];
W(:,uB+1) = [];
B(rB+1,:) = [];
B(:,uB+1) = [];
M(bB+1,:) = [];
M(:,bB+1) = [];
X(bB+1,:) = [];
X(:,bB+1) = [];
L(rB+1,:) = [];
L(:,rB+1) = [];
Qp(rB+1,:) = [];
Qp(:,rB+1) = [];
D(rB+1,:) = [];
D(:,bB+1) = [];
G(rB+1,:) = [];
G(:,bB+1) = [];
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
G = G';
% K = [M zeros(n_b, m_u);
%      zeros(m_u, n_b+m_u)];
K = [M 0*-C;0*C' A];

BhatT = [-C D'; B zeros(m_u, m_b)];
Bhat = [C' B'; D zeros(m_u, m_b)];
Chat = [null(full(M)) zeros(n_b, m_u);
        zeros(m_u, m_u) eye(m_u)];
norm(full(K*Chat))
spy(abs(pinv(G')*B)>1e-6)
norm(G'*C)
% X = [null(full(M)) D'; X zeros(n_b, m_b)];
% sort(real(eig(full(K))))
eig(full([B'*rand(m_u, m_u) B; L zeros(m_u,m_u)]))

G'*pinv(G')*B-B
% 
% C = inv(M  + D'*(L\D))*D';
% norm(full(X-D'*pinv(full(C))))
% alpha = 1.0;
% Km = [M-alpha*Mtilde D';
%     D zeros(m_b, m_b)];
% Kns = [F+alpha*Ftilde, B';
%      B, zeros(m_u, m_u)];
% Kct = [C'+alpha*Ctilde', zeros(n_u,m_b);
%      zeros(m_u, n_b+m_b)];
% Kc = [-C, zeros(n_b, m_u);
%      zeros(m_b, n_u+m_u)];
% K = [Kns, Kct; Kc, Km];
% invF = inv(F+alpha*Ftilde);
% Sf = B*invF*B';
% invS = inv(Sf);
% 
% N = invF-invF*B'*invS*B*invF;
% K1 = N;
% K2 = invF*B'*invS;
% K3 = invS*B*invF;
% K4 = -invS;
% invNS = [K1 K2;
%          K3 K4];
% invL = inv(L);
% invMx = inv(M  + D'*(L\D));
% G = invMx*D';
% Chat = C'+alpha*Ctilde';
% P = [K1, K2, -N*Chat*invMx zeros(n_u, m_b);
%     K3, K4, -K3*Chat*invMx zeros(m_u, m_b);
%     invMx*C*K1 invMx*C*K2 invMx G*invL
%     zeros(m_b, n_u+m_u) invL*G' 0*L];
% PK = [eye(n_u)+K1*Chat*invMx*C zeros(n_u, m_u) K1*Chat*(eye(n_b)-invMx*M) -K1*Chat*invMx*D';
%        K3*Chat*invMx*C eye(m_u) K3*Chat*(eye(n_b)-invMx*M) -K3*Chat*invMx*D';
%        zeros(n_b, n_u+m_u) invMx*(M+C*K1*Chat+D'*invL*D) invMx*D';
%        zeros(m_b, n_b+n_u+m_u) eye(m_b)];
% 
% 
% [V, E] = eig(full(PK));
% [E,i] = sort(diag(E));
% V = V(:,i);
% tol = 1e-4;
% size(null(full(alpha*Mtilde+C*K1*Chat)),2)+size(null(full(C)),2)+m_u+m_b
% 
% fprintf('   n_u = %4.0d, m_u = %4.0d, n_b = %4.0d, m_b = %4.0d \n\n', n_u, m_u, n_b, m_b);
% Eig1 =find(E>1-tol & E<1+tol);
% fprintf('   Number of eigenvalues of 1 %4.0d with tolerance  %1.1e \n\n', length(Eig1),tol);
% 
% Eig2 = find(E>-1-tol & E<-1+tol);
% fprintf('   Number of eigenvalues of -1 %4.0d with tolerance  %1.1e \n\n\n', length(Eig2),tol);
% nu2 = V(1:n_u,Eig1);
% mu2 = V(n_u+1:n_u+m_u,Eig1);
% nb2 = V(n_u+m_u+1:n_u+m_u+n_b,Eig1);
% mb2 = V(n_u+m_u+n_b+1:end,Eig1);
% 
% norm((alpha*Mtilde+C*N*Chat)*nb2)
% 
% e = eig(full(eye(n_b)+invMx*(alpha*Mtilde+C*N*Chat)));
% 
% plot(real(e),'*')
% figure;
% plot(imag(e),'*')
% 
% ssss
% % 
% 
% size(null(full(invMx*(-Mtilde+C*K1*Chat))))
% size(null(full(M)))
% 
% norm(full(eye(n_u)+K1*(Ftilde + (C'+Ctilde')*invMx*C) - PK(1:n_u, 1:n_u)))
% 
% spy(abs(Chat*(eye(n_b)-invMx*(M-alpha*Mtilde))-C'*invMx*Mtilde-Ctilde'*(eye(n_b)-invMx*(M-alpha*Mtilde)))>1e-6)
% % spy(abs(Chat*(eye(n_b)-invMx*(M-alpha*Mtilde)))>1e-6)
% 
% spy(abs(P*K)>1e-6)
% M = M - alpha*Mtilde;
% figure
% spy(abs(PK-P*K)>1e-6)
% % ssss
% e = eig(full(P*K));
% figure
% plot(real(e), '*')
% ssss
% Chat = C'+alpha*Ctilde';
% 
% Mf = M - alpha*Mtilde + C*K1*Chat + D'*(L\D);
% Mx = M - alpha*Mtilde + D'*(L\D);
% 
% e = eig(full(Mx), full(Mf));
% plot(sort(real(e)), '*')
% % figure
% ssss
% % sss
% % norm(full(inv(Mf) + (inv(Mx) - inv(Mx)*C*inv((K1)+Chat*inv(Mx)*C)*Chat*inv(Mx))))
% % spy(abs(inv(Mf) + (inv(Mx) - inv(Mx)*C*inv((K1)+Chat*inv(Mx)*C)*Chat*inv(Mx)))>1e-6)
% % sss
% G = Mf\D';
% Gt = D/Mf;
% H = (speye(n_b) - D'*(L\Gt));
% Mhat = M - alpha*Mtilde;
% spy(abs(Gt*Mhat)>1e-6)
% sss
% invSS = [Mf\H G/L;
%         L\Gt zeros(m_b)];
% 
% G = Mx\D';
% Gt = D/Mx;
% invS = [Mx G/L;
%         L\Gt zeros(m_b)];
% spy(abs(invS-invSS)>1e-6)
% figure
% e = eig(full(invS*S));
% 
% plot(real(e), '*')
% cond(full(invS*S))
% 
% invL = inv(L);
% invCT = -[K1*Chat*inv(Mx) K1*Chat*G*invL;
%          K3*Chat*inv(Mx) K3*Chat*G*invL];
% invC  = [inv(Mx)*C*K1 inv(Mx)*C*K2;
%          zeros(m_b, n_u+m_u)];
% Z = Chat*inv(Mx)*C;
% invNS = [K1 K2;
%          K3 K4];
% invK = [invNS invCT;
%         invC invS];
% e = eig(full(invK*K));
% figure
% spy(abs(invK*K)>1e-6);
% figure
% plot(real(e), '*')
% sss
% 
% 
% 
% 
% 
% 
% 
% 
% G = Mf\D';
% Gt = D/Mf;
% H = (speye(n_b) - D'*(L\Gt));
% invSS = [Mf\H G/L;
%         L\Gt zeros(m_b)];
% 
% % Mxx = M+D'*inv(L)*D;
% S = B*((F+alpha*Ftilde)\B');
% invF = inv(F+alpha*Ftilde);
% invS = inv(S);
% % invMx = inv(Mxx);
% invL = inv(L);
% N = invF-invF*B'*invS*B*invF;
% 
% K1 = N;
% K2 = invF*B'*invS;
% K3 = invS*B*invF;
% K4 = -invS;
% Chat = C'+alpha*Ctilde';
% invCT = -[K1*Chat*inv(Mf)*H K1*Chat*G*invL;
%          K3*Chat*inv(Mf)*H K3*Chat*G*invL];
% invC  = [inv(Mf)*C*K1 inv(Mf)*C*K2;
%          zeros(m_b, n_u+m_u)];
% Z = Chat*inv(Mf)*C;
% invNS = [K1-K1*Z*K1 K2-K1*Z*K2;
%          K3-K3*Z*K1 K4-K3*Z*K2];
% invK = [invNS invCT;
%         invC invSS];
% spy(abs(inv(K)-invK)>1e-6)
% sss
% % Mf0 = M + C*K1*C' + D'*(L\D);
% % e = eig(full(Mf), full(Mf0));
% % plot(imag(e), 'o')
% % ssss
% 
%     
% spy(abs(invS-inv(S))>1e-10)
% size(null(full(Mtilde')))
% fprintf('%4.0f\n',(length(rB)/4)^2)
% % close all
% % stop
% 
% SS = inv(S);
% norm(full(invS-inv(S)))
% invK = inv(K);
% sss
% ss = invK(n_u+m_u+1:end, n_u+m_u+1:end);
% alpha = 1;
% A = M-alpha*Mtilde;
% Km = [M-alpha*Mtilde D';
%     D zeros(m_b, m_b)];
% 
% spy(abs(inv(Km))>1e-6)
% Ahat = inv(A + D'*(L\D));
% 
% Kinv = [Ahat - Ahat*D'*inv(L)*D*Ahat Ahat*D'*inv(L);
%         inv(L)*D*Ahat 0*L];
% % close all
% 
% % spy(abs(A*(Ahat*D'))>1e-10);
% stop
% A = S(1:n_b, 1:n_b);
% Z = null(full(D));
% V = Z*inv(Z'*A*Z)*Z';
% Kinv = [V, (speye(n_b) - V*A)*D'*inv(D*D');
%         inv(D*D')*D*(speye(n_b) - A*V), -inv(D*D')*D*(A - A*V*A)*D'*inv(D*D')];
% figure
% spy(abs(Kinv-SS)>1e-10)
% 
% 
% % figure
% % alpha = 1;
% % A = M-alpha*Mtilde;
% % Km1 = [M-alpha*Mtilde D';
% %     D zeros(m_b, m_b)];
% % 
% % P = [M-alpha*Mtilde + D'*(L\D) ,0*D';
% %     0*D L];
% % 
% % alpha = 0;
% % A = M-alpha*Mtilde;
% % Km0 = [M-alpha*Mtilde D';
% %     D zeros(m_b, m_b)];
% % 
% % e = eig(full(Km1), full(P));
% % 
% % plot(sort(real(e)), '*')
% % hold on 
% % plot(sort(imag(e)))
% 
% 
% % Null = null(full(M));
% % size(Null)
% % norm((M+Mtilde)*Null)
% % 
% % nullC = null(full(Ctilde));
% % 
% % norm(C*nullC)
% % alpha = 1;
% % K = full([F+alpha*Ftilde, B', C'+alpha*Ctilde', zeros(n_u,m_b);
% %     B, zeros(m_u,m_u+n_b+m_b);
% %      -C, zeros(n_b,m_u) M-alpha*Mtilde D';
% %      zeros(m_b,n_u+m_b) D zeros(m_b,m_b)]);
% % alpha = 0;
% % K1 = full([F+alpha*Ftilde, B', C'+alpha*Ctilde', zeros(n_u,m_b);
% %     B, zeros(m_u,m_u+n_b+m_b);
% %      -C-alpha*Ctilde, zeros(n_b,m_u) M-alpha*Mtilde D';
% %      zeros(m_b,n_u+m_b) D zeros(m_b,m_b)]); 
% %  
% %  e = eig(K, K1);
% % %  plot(sort(real(e)), '*')
% % 
% % Maxwell = [M-alpha*Mtilde D'; D zeros(m_b,m_b)];
% % spy(abs(inv(Maxwell))>1e-6)
% % NullM = M-alpha*Mtilde;
% % norm(full(Mtilde*NullM))
% % size(null(full(alpha*Mtilde)))
% % size(null(full(M)))
% % size(null(full(M+alpha*Mtilde)))
% % 
% 
% 
% 
% 
% % fprintf('norm(full(F))         = %4.4f\n', norm(full(F)))
% % fprintf('norm(full(M))         = %4.4f\n', norm(full(M)))
% % fprintf('norm(full(C))         = %4.4f\n\n', norm(full(C)))
% % 
% % fprintf('norm(full(Ftilde))    = %4.4f\n', norm(full(Ftilde)))
% % fprintf('norm(full(Mtilde))    = %4.4f\n', norm(full(Mtilde)))
% % fprintf('norm(full(Ctilde))    = %4.4f\n\n', norm(full(Ctilde)))
% % 
% % fprintf('norm(full(F+Ftilde))  = %4.4f\n', norm(full(F+Ftilde)))
% % fprintf('norm(full(M+Mtilde))  = %4.4f\n', norm(full(M+Mtilde)))
% % fprintf('norm(full(C+Ctilde))  = %4.4f\n\n', norm(full(C+Ctilde)))
% % 
% % Km = [M+alpha*Mtilde D';
% %     D zeros(m_b, m_b)];
% % Kns = [F+alpha*Ftilde, B';
% %      B, zeros(m_u, m_u)];
% % Kct = [C'+alpha*Ctilde', zeros(n_u,m_b);
% %      zeros(m_u, n_b+m_b)];
% % Kc = [-C, zeros(n_b, m_u);
% %      zeros(m_b, n_u+m_u)];
% % K = [Kns, Kct; Kc, Km];
% % fprintf('norm(full(K_a1))      = %4.4f\n', norm(full(K)))
% % fprintf('cond(full(K_a1))      = %4.4f\n', cond(full(K)))
% % 
% % alpha = 0;
% % Km = [M+alpha*Mtilde D';
% %     D zeros(m_b, m_b)];
% % Kns = [F+alpha*Ftilde, B';
% %      B, zeros(m_u, m_u)];
% % Kct = [C'+alpha*Ctilde', zeros(n_u,m_b);
% %      zeros(m_u, n_b+m_b)];
% % Kc = [-C, zeros(n_b, m_u);
% %      zeros(m_b, n_u+m_u)];
% % K = [Kns, Kct; Kc, Km];
% % fprintf('norm(full(K_a0))      = %4.4f\n', norm(full(K)))
% % fprintf('cond(full(K_a0))      = %4.4f\n', cond(full(K)))
