load('S.mat')
load('T.mat')
load('T1.mat')
load('M.mat')
load('Z.mat')

Approx = M*(Z\T1);
spy(abs(T1-T)>1e-6)