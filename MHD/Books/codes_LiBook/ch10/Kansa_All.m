%--------------------------------------------------------
% KansaAll.m: use the so-called Kansa method to solve 
%             an elliptic problem.
% This implementation includes most popular RBFs.
%--------------------------------------------------------

mn=20;
h=1.0/(1.0+mn);
NI=mn*mn;     %total # of interior nodes

NBC=16;    %total BC points 
c=0.9;    
%c=1.8;     %shape parameter
N=NI+NBC;  %total # of collocation points
Tol=1.0e-15;

% ID=1 for MQ, ID=2 for r^{2k+1}, 
% ID=3 for r^4ln(r), ID=4 for e^{-beta*r^2}
ID=2;
NK=3;    % the parameter k in r^{2k+1}
beta=10;

XBC=zeros(1,NBC);   %x-coordinate of BC nodes
YBC=zeros(1,NBC);   %y-coordinate of BC nodes
XBC=[0.00 0.25 0.50 0.75 1.00 1.00 1.00 1.00 ...
    1.00 0.75 0.50 0.25 0.00 0.00 0.00 0.00];
YBC=[0.00 0.00 0.00 0.00 0.00 0.25 0.50 0.75 ...
    1.00 1.00 1.00 1.00 1.00 0.75 0.50 0.25];

X=zeros(1,N);
Y=zeros(1,N);
for j=1:mn
  for i=1:mn
    X(i+(j-1)*mn)=h*i;
    Y(i+(j-1)*mn)=h*j;
  end
end

%after this, 'h' is used for scaling, so h=1 means no scaling! 
h=1;
for i=1:NBC
   X(i+NI)=XBC(i);
   Y(i+NI)=YBC(i);
end

A=zeros(N,N);  %the Kansa's matrix
A11=zeros(NI,NI);  %A1 submatrix of A
A12=zeros(NI,NBC);  %submatrix of A
A21=zeros(NBC,NI);   %submatrix of A
A22=zeros(NBC,NBC);   %submatrix of A
GF=zeros(N,1);        %global RHS

%form global matrix A & rescale it
for i=1:NI
   GF(i)=13*exp(-2*X(i)+3*Y(i));
   GF(i)=GF(i)*(h*h);
   for j=1:N
     if ID == 1
      tmp=((X(i)-X(j))^2+(Y(i)-Y(j))^2)+c*c;
      A(i,j)=(tmp+c*c)/(tmp^1.5);
     elseif ID == 2     
      tmp=sqrt((X(i)-X(j))^2+(Y(i)-Y(j))^2);
      A(i,j)=(2*NK+1)^2*tmp^(2*NK-1);
     elseif ID == 3
        tmp= sqrt((X(i)-X(j))^2+(Y(i)-Y(j))^2);
        if tmp > Tol
           A(i,j)=16*tmp^2*log(tmp)+8*tmp^2;
        end
     elseif ID == 4
        tmp= sqrt((X(i)-X(j))^2+(Y(i)-Y(j))^2);
        A(i,j)=(-4*beta+4*beta^2*tmp^2)*exp(-beta*tmp^2);
     end 
      A(i,j)=A(i,j)*(h*h);
   end
end

%here we rescale the BC nodes
for i=NI+1:NI+NBC
   GF(i)=exp(-2*X(i)+3*Y(i));
%   GF(i)=GF(i)/(h*h);
   for j=1:N
    if ID == 1
     tmp=(X(i)-X(j))^2+(Y(i)-Y(j))^2+c*c;
     A(i,j)=sqrt(tmp);
    elseif ID == 2
          tmp=sqrt((X(i)-X(j))^2+(Y(i)-Y(j))^2);
          A(i,j)=tmp^(2*NK+1);
    elseif ID == 3
       tmp=sqrt((X(i)-X(j))^2+(Y(i)-Y(j))^2);
       if tmp > Tol
          A(i,j)=tmp^4*log(tmp);
       end
    elseif ID == 4
       tmp=sqrt((X(i)-X(j))^2+(Y(i)-Y(j))^2);
       A(i,j)=exp(-beta*tmp^2);
    end
%     A(i,j)=A(i,j)/(h*h);
   end
end

%Solve the equation: u_xx+u_yy=f(x,y), u=g(x,y) on BC
u=zeros(N,1);     %approx solution at all points
uex=zeros(N,1);   % exact solution

for i=1:N
     uex(i)=exp(-2*X(i)+3*Y(i));
end

%Solving the global system directly gives worse 
%accuracy than solved by LU decomposition!
%u=A\GF

%solve by LU decomposition
[L,U]=lu(A);
u=U\(L\GF);

fid2=fopen('out_KansaAll.txt','w');
for i=1:N
   %find the approx solution at each point
   uappr=0.0;
   for j=1:N
        r2=(X(i)-X(j))^2+(Y(i)-Y(j))^2;
     if ID == 1
      tmp=sqrt(r2+c*c);
     elseif ID == 2
        tmp=sqrt(r2)^(2*NK+1);
     elseif ID == 3
        if r2 > Tol
           tmp=r2^2*log(r2)/2;
        end
     elseif ID == 4
        tmp=exp(-beta*r2);
     end
      uappr=uappr+u(j)*tmp;
   end
   err(i)=abs(uappr-uex(i))/uex(i);
   fprintf(fid2,'%9.4f %9.4f %10.6e %10.6e %10.6e\n',...
                   X(i),Y(i), uappr, uex(i),err(i));
end
status=fclose(fid2);
disp('max relative err='), max(err)