%---------------------------------------------------------------
% disp1d.m: solve the linear dispersive equation 
%             u_t + c^{-2}u_{xxx}=0 
%      by high order compact difference method.
%
% Used functions: 
%     JSC1.m: the analytic solution
%     reconuxxxp.m: reconstrcution u_{xxx} from u
%     resJSC1.m: the residual function
%---------------------------------------------------------------
clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate 1D mesh

Nx=41; XL=0; XR=2*pi;
dx=(XR-XL)/(Nx-1);

for i=1:Nx
    XX(i)=XL+(i-1)*dx;
end

%%%%%%%%%march-in-time%%%%%%%%%%%%
% cfl has to be small!
cfl=0.125; dt=cfl*dx^3; tlast=1;
Nstep=tlast/dt;  
cc=8;   % the bigger, the larger frequency

IRK=4;

% get initial condition
[u0ex]=feval(@JSC1,XX,0,cc);
uold=u0ex;

subplot(2,1,1), plot(XX,u0ex, 'y-');
           hold on
% we use 4th-order RK scheme:IRK=2,4
for k=1:Nstep
  if IRK==4
    u0x=reconuxxxp(uold,Nx,dx);    % reconstruct ux from u_n
    k0=dt*resJSC1(uold,u0x,cc);    % k0=dt*R(dummy,uxxx,dummy)
    u1=uold+k0/2;
    
    u1x=reconuxxxp(u1,Nx,dx);
    k1=dt*resJSC1(uold,u1x,cc);
    u2=uold+k1/2;
    
    u2x=reconuxxxp(u2,Nx,dx);
    k2=dt*resJSC1(uold,u2x,cc);
    u3=uold+k2;
    
    u3x=reconuxxxp(u3,Nx,dx);
    k3=dt*resJSC1(uold,u3x,cc);
    
    unew=uold+(k0+2*k1+2*k2+k3)/6.;   % finish one-time step
  elseif IRK==2
    u0x=reconuxp(uold,Nx,dx);       % reconstruct ux from u_n
    u0xx=reconuxp(u0x,Nx,dx); 
    u0xxx=reconuxp(u0xx,Nx,dx);   % obtain u_xxx
    u2x=reconuxp(uold.^2,Nx,dx);  % reconstruct u^2
    k0=dt*resJSC1(u2x,u0xxx,cc);  % k0=dt*R((u_n)_x)
    u1=uold+k0;
    
    u1x=reconuxp(u1,Nx,dx);
    uxx=reconuxp(u1x,Nx,dx); 
    uxxx=reconuxp(uxx,Nx,dx);   % obtain u_xxx
    u2x=reconuxp(u1.^2,Nx,dx);  % reconstruct u^2
    k1=dt*resJSC1(u2x,uxxx,cc);
    u2=u1+k1;
    
    unew=(uold+u2)/2;
  end
    
    uold=unew;    % update for next time step
    eps=0.3*dt;   % error tolerance
    % plot the solution at some specific times
    if  abs(k*dt-0.25) < eps | abs(k*dt-0.5) < eps ...
           | abs(k*dt-0.75) < eps | abs(k*dt-1) < eps    
        disp(k),
        
       [u0ex]=feval(@JSC1,XX,k*dt,cc);
       subplot(2,1,1),plot(XX,unew, 'y-',XX,u0ex,'g.');
       xlabel('x'); ylabel('u(x,t)');
       hold on
       subplot(2,1,2), 
       if abs(k*dt-0.25) < eps, plot(XX,abs(u0ex-unew), 'r-'); end
       if abs(k*dt-0.5) < eps, plot(XX,abs(u0ex-unew), 'r-.'); end
       if abs(k*dt-0.75) < eps, plot(XX,abs(u0ex-unew), 'r--'); end
       if abs(k*dt-1) < eps, plot(XX,abs(u0ex-unew), 'r:'); end
           
       xlabel('x'); ylabel('inf error');
       hold on           
   end
       
end   

