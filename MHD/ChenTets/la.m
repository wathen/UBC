
[n,m]=size(B);
GQ=[A+B*(Q\B') sparse(n,m); sparse(m,n) Q];
GS=[A+B*(S\B') sparse(n,m); sparse(m,n) S];
GSA=[A+M sparse(n,m); sparse(m,n) S];

%tic;[xQ,flagQ,relresQ,iterQ,resvecQ] = minres(K,F,1e-10,200,GQ); flagQ, iterQ , toc
%tic;[xS,flagS,relresS,iterS,resvecS] = minres(K,F,1e-10,200,GS); flagS, iterS  ,toc
%tic;[xSA,flagSA,relresSA,iterSA,resvecSA] = minres(K,F,1e-10,200,GSA); flagSA, iterSA , toc

%eigQ=sort(eig(full(K),full(GQ)));
%eigS=sort(eig(full(K),full(GS)));
%eigSA=sort(eig(full(K),full(GSA)));