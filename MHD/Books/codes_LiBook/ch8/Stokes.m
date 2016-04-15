%-----------------------------------------------------------
% Stokes.m: solve the Stokes flow in a cavity 
%           using P2-P0 elements.
% 
% The code needs: 
%     1. gen_p2grid.m: generate 6-node triangle mesh
%     2. elmD.m: produce the element matrix D_u
%     3. elmDxy.m: produce the element matrix D_x, D_y
%     4. p2basis.m: calculate the P2 basis functions
%     5. p2quad.m: Gaussian quadrature on triangle
%-----------------------------------------------------------
clear all;

vis = 1.0; % viscosity
Vel = 1.0;    % lid velocity 
NQ = 7;     % gauss-triangle quadrature
nref = 3;   % discretization level

% generate p2 grid
[ne,np,p,conn,efl,gfl] = gen_p2grid(nref);
disp('Number of elements:'); ne

inodes = 0;
for j=1:np
 if(gfl(j,1)==0)   %interior nodes 
  inodes = inodes+1;
 end
end
disp('Number of interior nodes:'); inodes

% specify the boundary velocity
for i=1:np
 if(gfl(i,1)==1)
   gfl(i,2) = 0.0; gfl(i,3) = 0.0; % x and y velocities
   if(p(i,2) > 0.999)
    gfl(i,2) = Vel;  % x velocity on the lid
   end
 end
end

% assemble the global diffusion matrix, Dx and Dy matrices
gdm = zeros(np,np); % initialization
gDx = zeros(ne,np); 
gDy = zeros(ne,np); 

for l=1:ne          % loop over the elements
  j=conn(l,1); x1=p(j,1); y1=p(j,2);
  j=conn(l,2); x2=p(j,1); y2=p(j,2);
  j=conn(l,3); x3=p(j,1); y3=p(j,2);
  j=conn(l,4); x4=p(j,1); y4=p(j,2);
  j=conn(l,5); x5=p(j,1); y5=p(j,2);
  j=conn(l,6); x6=p(j,1); y6=p(j,2);

  [edm_elm, arel] = elmD ...
   (x1,y1, x2,y2, x3,y3, x4,y4, x5,y5, x6,y6, NQ);

  [Dx, Dy] = elmDxy ...
   (x1,y1, x2,y2, x3,y3, x4,y4, x5,y5, x6,y6, NQ);

   for i=1:6
     i1 = conn(l,i);
     for j=1:6
       j1 = conn(l,j);
       gdm(i1,j1) = gdm(i1,j1) + edm_elm(i,j);
     end
     gDx(l,i1) = Dx(i);
     gDy(l,i1) = Dy(i);
   end
end

% form the final global coefficient matrix
nsys = 2*np+ne;   % total number of unknowns
Gm=zeros(nsys,nsys);
b=zeros(1,nsys);

for i=1:np     % first big block
  for j=1:np
    Gm(i,j) = vis*gdm(i,j);  Gm(np+i,np+j) = Gm(i,j);
  end
  for j=1:ne
    Gm(i,  2*np+j) = -gDx(j,i);
    Gm(np+i,2*np+j) = -gDy(j,i);
  end
end

% second big block
for i=1:ne
  for j=1:np
    Gm(2*np+i,j)    = -gDx(i,j);
    Gm(2*np+i,np+j) = -gDy(i,j);
  end
end

% compute RHS of the system and implement the Dirichlet BC
for j=1:np
 if(gfl(j,1)==1) 
   for i=1:nsys
    b(i) = b(i) - Gm(i,j)*gfl(j,2) - Gm(i,np+j)*gfl(j,3);
    Gm(i,j) = 0; Gm(i,np+j) = 0;
    Gm(j,i) = 0; Gm(np+j,i) = 0;
   end
   Gm(j,j) = 1.0;
   Gm(np+j,np+j) = 1.0;
   b(j) = gfl(j,2);
   b(np+j) = gfl(j,3);
 end
end

% solve the linear system
Gm(:,nsys) = [];  % remove the last column
Gm(nsys,:) = [];  % remove the last row
b(nsys) = [];  % remove the last component

sol=Gm\b';

% recover the velocity
for i=1:np
 ux(i) = sol(i);  uy(i) = sol(np+i);
end

% plot the velocity field
figure(1);
quiver(p(:,1)',p(:,2)',ux,uy);
hold on;
xlabel('x','fontsize',10)
ylabel('y','fontsize',10)
set(gca,'fontsize',15)
axis('equal')

% plot the mesh
figure(2);
for i=1:ne
 i1=conn(i,1); i2=conn(i,2); i3=conn(i,3);
 i4=conn(i,4); i5=conn(i,5); i6=conn(i,6);
 xp(1)=p(i1,1); yp(1)=p(i1,2); xp(2)=p(i4,1); yp(2)=p(i4,2);
 xp(3)=p(i2,1); yp(3)=p(i2,2); xp(4)=p(i5,1); yp(4)=p(i5,2);
 xp(5)=p(i3,1); yp(5)=p(i3,2); xp(6)=p(i6,1); yp(6)=p(i6,2);
 xp(7)=p(i1,1); yp(7)=p(i1,2);
 plot(xp, yp, ':');
 hold on;
 plot(xp, yp,'o','markersize',5);
end