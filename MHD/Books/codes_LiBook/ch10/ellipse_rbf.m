%-------------------------------------------------------------
% ellipse_rbf.m: solve the biharmonic problem over an ellipse.
% 
% mod 1: uex=x^3+y^3;
% mod 2: uex=0.5*(x*sin(x)*cosh(y)-x*cos(x)*sinh(y))
% mod 3: uex=(x^2+y^2)/64;
%-------------------------------------------------------------
clear all;

Nc=49;   % # of nodes in theta direction
Nr=16;   % # of nodes in the radial direction

NI=(Nr-1)*Nc;    % # of interior nodes: one in the center
Nd=Nc;     % # of Diri BC nodes on the circle
Nn=Nc;     % # of Neum Bc nodes on the  circle
Nall=NI+Nd+Nn;   % total # of collocation points
NK=9;     % the degree of the basis function r^NK
MODEL=3;    % indicator which model 

dc=2*pi/Nc;   
Ra=1; Rb=0.8333;    %semi-major, semi-minor of the ellipse
dra=Ra/Nr; drb=Rb/Nr;

% generate the nodes: 1) inner; 2) Diri BC; 3) Neum BC
for j=1:Nc
    for i=1:Nr-1
        % uniform grid
        ra=i*dra;  rb=i*drb;
        theta=(j-1)*dc;
        ij=(j-1)*(Nr-1)+i;
        XX(ij)=ra*cos(theta);  YY(ij)=rb*sin(theta);
        XC(ij)=XX(ij);  YC(ij)=YY(ij);
     end
end

% Diri BC
for j=1:Nd
    theta=(j-1)*dc;
    ij=NI+j;
    XX(ij)=Ra*cos(theta); YY(ij)=Rb*sin(theta);
    XC(ij)=XX(ij);   YC(ij)=YY(ij);
 end


% Neum BC
for j=1:Nn
    theta=(j-1)*dc+0.5*dc;  % nodes between Diri BC nodes
    ij=NI+Nd+j;
    XX(ij)=Ra*cos(theta); YY(ij)=Rb*sin(theta);
    XC(ij)=XX(ij); YC(ij)=YY(ij);
end

% form the coeff. matrix and HRS
for i=1:NI       % inner nodes
    rhs(i)=0;
  for j=1:Nall
      dist=sqrt((XX(i)-XC(j))^2+(YY(i)-YC(j))^2);
     matA(i,j)=(NK*(NK-2))^2*dist^(NK-4);
 end
end

for i=1:Nd      % Diri BC nodes
    ii=NI+i;
    if MODEL==1
      % mod 1
      rhs(ii)=XX(ii)^3+YY(ii)^3;
    elseif MODEL==3
      rhs(ii)=-(XX(ii)^2+YY(ii)^2)^2/64;
    elseif MODEL==2
      % mod 2
      rhs(ii)=0.5*XX(ii)*(sin(XX(ii))*cosh(YY(ii)) ...
                         -cos(XX(ii))*sinh(YY(ii)));
    end
   for j=1:Nall
      dist=sqrt((XX(ii)-XC(j))^2+(YY(ii)-YC(j))^2);
     matA(ii,j)=dist^NK;
   end
end 


for i=1:Nn      % Neum BC nodes
    ii=NI+Nd+i;
    theta=(i-1)*dc+0.5*dc;
    dxdt=-Ra*sin(theta); dydt=Rb*cos(theta);
    % find the normal at this boundary node
    if abs(theta) < eps 
        cosc=1; sinc=0;
    elseif abs(theta-pi) < eps
        cosc=-1; sinc=0;
    else
        psi=atan(dydt/dxdt);
        cosc=sin(psi); sinc=cos(psi);
    end
    
    if MODEL==1
      % mod 1
      rhs(ii)=3*XX(ii)^2*cosc+3*YY(ii)^2*sinc;  % du/dn
    elseif MODEL==3
      xi=XX(ii); yi=YY(ii);
      t1=2*(xi^2+yi^2)/64*2*xi;
      t2=2*(xi^2+yi^2)/64*2*yi;
      rhs(ii)=-(t1*cosc+t2*sinc);
    elseif MODEL==2
      % mod 2
      xi=XX(ii); yi=YY(ii);
      c1=cosh(yi); s1=sinh(yi);
      t1=(xi*cos(xi)+sin(xi))*c1-(cos(xi)-xi*sin(xi))*s1;
      t2=xi*sin(xi)*s1-xi*cos(xi)*c1;
      rhs(ii)=0.5*(t1*cosc+t2*sinc);
    end
  for j=1:Nall
      dist=sqrt((XX(ii)-XC(j))^2+(YY(ii)-YC(j))^2);
      phix=NK*dist^(NK-2)*(XX(ii)-XC(j));
      phiy=NK*dist^(NK-2)*(YY(ii)-YC(j));
     matA(ii,j)=phix*cosc+phiy*sinc;
 end
end 

% solve the linear system
u=matA\rhs';

plot(XX(1:NI),YY(1:NI),'.',...    % interior node
   XX(NI+1:NI+Nd),YY(NI+1:NI+Nd),'s',...   % Diri node
   XX(NI+Nd+1:NI+Nd+Nn),YY(NI+Nd+1:NI+Nd+Nn),'o'); %Neum node
legend('interior node','Dirichlet boundary node',...
       'Neumann boundary node');

fid=fopen('V1.doc','w');   % save the results
% get the numerical solution and compare with the exact one
%for i=NI+1:Nall   % model 3: only compare boundary errors
for i=1:Nall
    if MODEL==1
      % mod 1
      uex(i)=XX(i)^3+YY(i)^3;     % exact solution
    elseif MODEL==3
       uex(i)=-(XX(i)^2+YY(i)^2)^2/64;     
    elseif MODEL==2
      % mod 2
      uex(i)=0.5*XX(i)*(sin(XX(i))*cosh(YY(i))...
                       -cos(XX(i))*sinh(YY(i)));
    end
    uapp(i)=0;
    for j=1:Nall
        dist=sqrt((XX(i)-XC(j))^2+(YY(i)-YC(j))^2);
        uapp(i)=uapp(i)+u(j)*dist^NK;
    end
    err(i)=abs(uex(i)-uapp(i));
    if abs(uex(i)) > 1e-6
        Rerr(i)=err(i)/abs(uex(i));   % relative error
    else
        Rerr(i)=0;
    end
    fprintf(fid, '%9.4f %9.4f %10.6e %10.6e %10.6e %10.6e\n', ...
        XX(i),YY(i),uapp(i),uex(i),err(i),Rerr(i));
end
disp('Nall=, max err, Relative err='), ...
      Nall, max(err), max(Rerr),
% plot only the errors on the boundaries
figure(2),
plot(NI+1:Nall,err(NI+1:Nall),'b-', ...
     NI+1:Nall,err(NI+1:Nall),'.');   
status=fclose(fid);
