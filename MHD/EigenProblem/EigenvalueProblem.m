%% MHD eigenvalue problem
clf
n = 10; m = 10;

F = randn(n,n);
B = randn(n,m)';
C =  [randn(n-1,n);sparse(1,n)];
M = [randn(n-1,n);sparse(1,n)];
D = randn(n,m)';
Q = speye(n);
L = randn(m,m);
Ms = (B*Q*B')*((B*Q*F*Q*B')\(B*Q*B'));
S = M + D'*(L\D);

A = [F B' C' sparse(n,m);B sparse(2*m+n,m)';-C sparse(m,n)' M D';sparse(m,m+n) D sparse(m,m)];

P = [F B' C' sparse(n,m); sparse(n,m)' -Ms sparse(m,n+m); -C sparse(n,m) S sparse(n,m); sparse(m,n) sparse(n+m,m)' L];

[V,D] = eig(full(P),full(A));

plot(diag(D),'*')



%% MHD eigenvalue problem
clf
clear
n = 20; nhat = 9; m = 30; mhat = 13;

F = randn(n,n);
B = randn(n,nhat)';
C =  [randn(n,m-1),sparse(n,1)]';
% C = randn(n,m)';
M = [randn(m-1,m);sparse(1,m)];
% M = randn(m,m);
D = randn(m,mhat)';
Q = speye(n);
L = randn(mhat,mhat);
% Ms = (B*Q*B')*((B*Q*F*Q*B')\(B*Q*B'));
Ms = B*(F\B');
S = M + D'*(L\D);

A = [F B' C' sparse(n,mhat);B sparse(nhat+m+mhat,nhat)';-C sparse(nhat,m)' M D';sparse(mhat,nhat+n) D sparse(mhat,mhat)];

P = [F B' C' sparse(n,mhat); sparse(n,nhat)' -Ms sparse(nhat,m+mhat); -C sparse(m,nhat) S sparse(m,mhat); sparse(mhat,n) sparse(m+nhat,mhat)' L];

[V,D] = eig(full(P),full(A),'qz');

plot(diag(D),'*')
