%--------------------------------------------------------------
% hoc2d.m: solve 2D parabolic equ: 
%        u_t=u_{xx}+u_{yy}+F(x,y,t), (x,y)\in (0,1)^2
% by high-order compact ADI scheme.
%
% Other functions needed:
%     F2dconeF.m: source term F
%     F2dcone.m: the exact solution u
%     reconux2p.m: generate the matrix used by reconstruction 
%--------------------------------------------------------------
clear all;

    T=1;  %hr
    XL=0; XR=1;  % left and right end points
    Nx=51;    % total number of points
    dx=(XR-XL)/(Nx-1);
    YL=0; YR=1;  % left and right end points
    Ny=51;    % total number of points
    dy=(YR-YL)/(Ny-1);
    
    dt=1/80;     
    uold=zeros(Nx,Ny);    

Nstep=round(T/dt);

dtx=dt/dx; dty=dt/dy;

for i=1:Nx
    XX(i)=XL+(i-1)*dx;
end
for i=1:Ny
    YY(i)=YL+(i-1)*dy;
end

% set up initial cond
for i=1:Nx
    for j=1:Ny
        uold(i,j)=F2dcone(XX(i),YY(j),0);
    end
end

%%%%%%%%%march-in-time%%%%%%%%%%%%
matx=zeros(Nx,Nx);  % define matrix for reconstructing u_xx
maty=zeros(Ny,Ny);  % define matrix for reconstructing u_yy

matx=reconux2p(Nx,dx);   % reconstruction in x-direction 
maty=reconux2p(Ny,dy);  % reconstruction in y-direction

t1=cputime;

Ainv=inv(eye(Nx,Nx)-0.5*dt*matx);
Binv=inv(eye(Ny,Ny)-0.5*dt*maty);

for k=1:Nstep
  % reconstruct u_{yy}
  for i=1:Nx
      tmp=(uold(i,1:Ny))';     % has to transpose
      tmp2=maty*tmp;           % construct u_yy
      uyy(i,1:Ny)=tmp2';
  end
  
    % start ADI scheme
  for j=1:Ny
      for i=1:Nx
        TMPF(i,j)=0.5*dt*F2dconeF(XX(i),YY(j),(k-0.5)*dt);
        rhs(i)=uold(i,j)+0.5*dt*uyy(i,j)+TMPF(i,j);
      end
      uHaf(1:Nx,j)=Ainv*rhs';
  end
  
  % reconstruct u_{xx} 
  for j=1:Ny
      uxx(1:Nx,j)=matx*uHaf(1:Nx,j);   % reconstruct u_{xx}
  end
  
  for i=1:Nx
      for j=1:Ny
         rhs(j)=uHaf(i,j)+0.5*dt*uxx(i,j)+TMPF(i,j);
      end
      unew(i,1:Ny)=(Binv*rhs')';
  end
    
  uold=unew;    % update for next time step
end

    % draw numerical solution and max errors
    if k == Nstep
        disp('k='), k
        errL2=0;
        for i=1:Nx
            for j=1:Ny
              % get the exact solution
              u2d(i,j)=feval(@F2dcone,XX(i),YY(j),dt*k);  
              errL2=errL2+(u2d(i,j)-unew(i,j))^2;
            end
        end
        errL2=sqrt(errL2/(Nx*Ny));
        
        subplot(2,1,1), surf(XX,YY,unew');
        xlabel('x'); ylabel('y'); zlabel('u(x,y,t)');
        subplot(2,1,2), surf(XX,YY,(unew-u2d)');
        xlabel('x'); ylabel('y'); zlabel('error=u-u_{exact}');
        err=max(max(abs(unew-u2d)));
        disp('L2 error='),errL2,
        disp('absolute max error='), err,
        disp('relative max error='), err/max(max(abs(u2d))),
    end
       
t_used=cputime-t1;     % get time CPU time used in seconds
disp('CPU time used='), disp(t_used),
