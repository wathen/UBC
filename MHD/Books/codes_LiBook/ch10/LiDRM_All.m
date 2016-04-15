%-------------------------------------------------------------
% LiDRM_all.m: use the RBF-DRM method solve elliptic problem.
% 
% This code implements most RBF functions.
% Run directly the code gives the relative max error 0.0066.
%-------------------------------------------------------------

tt=cputime;

%total intervals in each direction
mn=20;
h=1.0/mn;
%total collocation points
mn1=mn+1;
nn=mn1*mn1;
cs=0.2;  %shape parameter for MQ

% NK: order of r^{2k+1}, k=1,2,3,...
%     or order of TPS r^{2k}*ln(r), k=1,2, ...
%ID=1 for MQ; ID=2 for r^3; 
%ID=3 for r^4ln(r); ID=4 for e^(-beta*r^2)
ID=2;
NK=3;
beta=5;

p1=zeros(1,nn);
p2=zeros(1,nn);
for j=1:mn1
  for i=1:mn1
    p1(i+(j-1)*mn1)=h*(i-1);
    p2(i+(j-1)*mn1)=h*(j-1);
  end
end

aa=zeros(nn,nn);
%t=cputime;

% Form matrix 'aa'. Here 'nn' is the # of collocation pts
% aa=[L\phi_j(P_i)]
for i=1:nn
   for j=1:nn
     if ID == 1     % use MQ
       r2=(p1(i)-p1(j))^2+(p2(i)-p2(j))^2+cs*cs;
       aa(i,j)=(r2+cs*cs)/(r2^1.5);
     elseif ID == 2 % use r^{2k+1}, k=1
       tmp=sqrt((p1(i)-p1(j))^2+(p2(i)-p2(j))^2);
       aa(i,j)=(2*NK+1)^2*tmp^(2*NK-1);        
     elseif ID == 3
        tmp= sqrt((p1(i)-p1(j))^2+(p2(i)-p2(j))^2);
        if tmp > 0
           aa(i,j)=16*tmp^2*log(tmp)+8*tmp^2;
        end
     elseif ID == 4
        tmp= sqrt((p1(i)-p1(j))^2+(p2(i)-p2(j))^2);
        aa(i,j)=(-4*beta+4*beta^2*tmp^2)*exp(-beta*tmp^2);
     end
   end
end

% RHS of the original equation
g=zeros(nn,1);
%g=2*exp(p1'-p2');
g=13*exp(-2*p1'+3*p2');
[L,U]=lu(aa);
c=U\(L\g);

% boundary nodes
q1=[0 0 0 0 0 0 .2 .4 .6 .8 1 1 1 1 1 1 .8 .6 .4 .2 ];
q2=[0 .2 .4 .6 .8 1 1 1 1 1 1 .8 .6 .4 .2 0 0 0 0 0 ];

% here we get particular solutions at all points
n=20;
for i=1:nn+n
   sum=0;
   if i <= nn 
      x=p1(i); y=p2(i);
   else 
      x=q1(i-nn); y=q2(i-nn);
   end
   for j=1:nn
      if ID == 1 % use MQ
        phi=sqrt((x-p1(j))^2+(y-p2(j))^2+cs*cs);
      elseif ID == 2 % use r^{2k+1}, k=1,2, ...
         phi=sqrt((x-p1(j))^2+(y-p2(j))^2)^(2*NK+1);
      elseif ID == 3
        r2=(x-p1(j))^2+(y-p2(j))^2;
        if r2 > 0
           phi=r2^2*log(r2)/2;
        end
      elseif ID == 4
        r2=(x-p1(j))^2+(y-p2(j))^2;
        phi=exp(-beta*r2);
      end
     sum=sum+c(j)*phi;
  end
  v2(i)=sum;
end

% BC values for MFS = original BC - particular solution
v3=v2(nn+1:nn+n);
g2=zeros(1,n);
%g2=exp(q1-q2)+exp(q1).*cos(q2)-v3;
g2=exp(-2*q1+3*q2)-v3;

% form the source points for MFS
 m=n-1;
rad=0:2.*pi/m:2.*pi*(1-1/m);
x1=5*cos(rad)+.5;
y1=5*sin(rad)+.5;

% form matrix 'A' for MFS, 'n' is the # of boundary points
A=zeros(n,n);
 for i=1:n
    A(i,n)=1;
    for j=1:n-1
      A(i,j)=log((x1(j)-q1(i))^2+(y1(j)-q2(i))^2);
    end
 end

% solution returned in vector 'z'
[L,U]=lu(A);
z=U\(L\g2');

%  z=A\g2';
  err_max = 0;
  fid2=fopen('out_LiDRM.txt','w');
% evaluate the error at those interesting points only!
    for j=1:nn
% get the points (x,y) we are interesting in!
        x=p1(j);
        y=p2(j);
% 'n' is the # of BC points
        v=0;
        for i=1:n-1
           d1=x1(i)-x;
           d2=y1(i)-y;
           v=v+z(i)*log(d1*d1+d2*d2);
        end

% final solution = particular solution + MFS' solution
        total=v2(j)+v+z(n);
        exact=exp(-2*x+3*y);
        diff=abs(exact-total)/exact;
        fprintf(fid2,'%9.4f %9.4f %10.6e %10.6e %10.6e\n',...
                       x,    y,    total, exact, diff);
        if (err_max < diff) 
            err_max = diff;     % find the max relative error
        end
    end
status=fclose(fid2);

disp('max relative err='), err_max,  
