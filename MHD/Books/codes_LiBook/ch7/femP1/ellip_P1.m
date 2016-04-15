%-------------------------------------------------------
% ellip.m: finite element code for solving a 2nd-order
%          convecton-diffusion problem using p1 element.
% 
% Functions used:
%    get_p1grid.m: generate a triangular grid
%    locA.m, locB.m, locM.m: evaluate element matrices
%    EXACT.m, SRC.m: exact solution and RHS function of the PDE
%
% nref=2,3,4,5 give max error as follows:
%    0.0044, 0.0014, 4.1030e-004, 1.2032e-004.
%-------------------------------------------------------
clear all;

% velocity components in the governing equation
Vx=1.0;  Vy=1.0;   

% generate a triangular grid by uniformly refining a coarse grid
nref = 3;      % level of refinement
[ne,np,p,conn,gbc] = gen_p1grid(nref); 

% plot the mesh to see if it is right
figure(1);
trimesh(conn,p(:,1),p(:,2));     

pause(2);

% specify the exact solution and use it for Dirichlet BC
for i=1:np
    u_ex(i)=feval(@EXACT,p(i,1),p(i,2));
    if(gbc(i,1)==1)    % indicator for Dirichlet BC
       gbc(i,2) = u_ex(i);
    end
end

% initialize those arrays
Ag = zeros(np,np); 
b   = zeros(np,1);

% loop over the elements
for l=1:ne 

  j=conn(l,1); x1=p(j,1); y1=p(j,2);
  j=conn(l,2); x2=p(j,1); y2=p(j,2);
  j=conn(l,3); x3=p(j,1); y3=p(j,2);

  % compute local element matrices
  [elmA] = locA(x1,y1,x2,y2,x3,y3);
  [elmB] = locB(Vx,Vy,x1,y1,x2,y2,x3,y3);
  [elmM] = locM(x1,y1,x2,y2,x3,y3);
  
   for i=1:3
     i1 = conn(l,i);
     for j=1:3
       j1 = conn(l,j);
       % assemble into the global coefficient matrix
       Ag(i1,j1) = Ag(i1,j1) + elmA(i,j) + elmB(i,j);
      % form the RHS of the FEM equation       
       b(i1) = b(i1) + elmM(i,j)*feval(@SRC,p(j1,1),p(j1,2));
     end
   end
end

% impose the Dirichlet BC
for m=1:np
 if(gbc(m,1)==1) 
   for i=1:np 
    b(i) = b(i) - Ag(i,m) * gbc(m,2);
    Ag(i,m) = 0; Ag(m,i) = 0;
   end
   Ag(m,m) = 1.0; b(m) = gbc(m,2);
  end
end

% solve the linear system
u_fem=Ag\b;

% plot the numerical solution
figure(2);
trisurf(conn,p(:,1),p(:,2),u_fem);

% compare with exact solution
disp('max err='), max(abs(u_fem-u_ex')),
