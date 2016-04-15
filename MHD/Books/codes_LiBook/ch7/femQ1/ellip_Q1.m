%-------------------------------------------------------
% 2D Q1 FEM for solving 
%     -Lap*u + u = f(x,y), on (0,length)x(0,height)
%      Neum BC=0  
% Max err=0.0242,0.0061,0.0015 when nx=10,20,40 
% so we did see O(h^2) convergence rate!
%--------------------------------------------------------

clear all;
length = 1.0;
height = 1.0;
nx=20;
ny=20;
gauss = [-1/sqrt(3), 1/sqrt(3)];  % Gaussian quadrature point

% construct Q1 mesh
[x,y,conn,ne,np] = getQ1mesh(length,height,nx,ny);

Ag = zeros(np);
bg=zeros(np,1);

nloc = 4;   % number of nodes per element
for ie = 1:ne    % loop over all elements
   rhs= (feval(@SRC,x(conn(ie,1)),y(conn(ie,1)))...
       + feval(@SRC,x(conn(ie,2)),y(conn(ie,2)))...
       + feval(@SRC,x(conn(ie,3)),y(conn(ie,3)))...
       + feval(@SRC,x(conn(ie,4)),y(conn(ie,4))))/nloc;
   
   [A_loc,rhse] = elemA(conn,x,y,gauss,rhs,ie);
   % assemble local matrices into the global matrix
   for i=1:nloc;
      irow = conn(ie,i);   % global row index
      bg(irow)=bg(irow) + rhse(i);
      for j=1:nloc;
         icol = conn(ie,j);  %global column index
         Ag(irow, icol) = Ag(irow, icol) + A_loc(i,j);
      end;
   end;
end;

%solve the equation
u_fem = Ag\bg;

u_ex=cos(pi*x).*cos(pi*y);  % get the exact solution

disp('Max error='),
max(u_fem-u_ex'),             % find the max error

% plot the FEM solution    
tri = delaunay(x,y);
trisurf(tri,x,y,u_fem);
