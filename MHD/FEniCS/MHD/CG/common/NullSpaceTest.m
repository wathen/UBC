n = 176;
m = 81;

load('C.mat')
load('grad.mat')
load('A.mat')


Couple = C(1:n,1:n);
CurlCurl = A(1:n,1:n);



nnz(abs(CurlCurl*grad))