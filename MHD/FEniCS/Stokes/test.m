load B.mat
load C.mat
load L.mat
load rhs.mat

[m1,m2] = size(C);

n = min(m1,m2);

a22 = sparse(n,n);
rhs2 = sparse(n,1);


b = [rhs';rhs2];
A = [L,B';C',a22];

t

M =  blkdiag(L,C'*(L\B'));

[X,FLAG,RELRES,ITER] = minres(A,b,1e-6,1000,M);