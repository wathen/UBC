%-------------------------------------------------------
% Ref to S.K. Lele's 1992 paper
% Reconstruct the 2nd derivative of a periodic function
%-------------------------------------------------------
function matAB=reconux2p(N,h)

matA=zeros(N,N); matB=zeros(N,N);
% for interior point
% for 6th-order scheme (2.2.7)
alfa=2.0/11.0; aa=4./3.*(1-alfa); bb=1./3.*(-1+10*alfa);

for i=3:N-2
    matA(i,i)=1;
    matA(i,i+1)=alfa;  
    matA(i,i-1)=alfa;
    matB(i,i-2)=bb/4;
    matB(i,i-1)=aa;
    matB(i,i)=-bb/2-2*aa;
    matB(i,i+1)=aa;
    matB(i,i+2)=bb/4;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for boundary point 1: 6th-order at boundary nodes!!
for i=1:1
    matA(i,i)=1;
    matA(i,i+1)=alfa;  
    matA(i,N-1)=alfa;

    matB(i,N-2)=bb/4;
    matB(i,N-1)=aa;
    matB(i,i)=-bb/2-2*aa;
    matB(i,i+1)=aa;
    matB(i,i+2)=bb/4;
end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%
% for boundry point 2
for i=2:2
    matA(i,i)=1;
    matA(i,i+1)=alfa;  
    matA(i,i-1)=alfa;
 
    matB(i,N-1)=bb/4;
    matB(i,i-1)=aa;
    matB(i,i)=-bb/2-2*aa;
    matB(i,i+1)=aa;
    matB(i,i+2)=bb/4;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for boundary point N-1
for i=N-1:N-1
    matA(i,i)=1;
    matA(i,i+1)=alfa;  
    matA(i,i-1)=alfa;
 
    matB(i,i-2)=bb/4;
    matB(i,i-1)=aa;
    matB(i,i)=-bb/2-2*aa;
    matB(i,i+1)=aa;
    matB(i,2)=bb/4;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for boundary point N
for i=N:N
    matA(i,i)=1;
    matA(i,2)=alfa;  
    matA(i,i-1)=alfa;

    matB(i,i-2)=bb/4;
    matB(i,i-1)=aa;
    matB(i,i)=-bb/2-2*aa;
    matB(i,2)=aa;
    matB(i,3)=bb/4;
end
%%%%%%%%%%%%%%%%%%%%%%%

matB=matB./(h*h);

%matA, matB,

matAB=inv(matA)*matB;    % the matrix for u_xx=A^(-1)B*u

