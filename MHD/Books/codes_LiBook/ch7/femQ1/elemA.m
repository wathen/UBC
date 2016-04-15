function [ke,rhse] = elemA(conn,x,y,gauss,rhs,e);

% 2d Q1 element stiffness matrix
ke = zeros(4,4);
rhse=zeros(4,1);
one = ones(1,4);
psiJ = [-1, +1, +1, -1]; etaJ = [-1, -1, +1, +1];

% get coordinates of element nodes 
for j=1:4
   je = conn(e,j); xe(j) = x(je); ye(j) = y(je);
end

for i=1:2  % loop over gauss points in eta
  for j=1:2   % loop over gauss points in psi
     eta = gauss(i);  psi = gauss(j);
  % shape function: countcockwise starting at left-low corner
     NJ=0.25*(one + psi*psiJ).*(one + eta*etaJ);
     % derivatives of shape functions in reference coordinates
     NJpsi = 0.25*psiJ.*(one + eta*etaJ);    % 1x4 array
     NJeta = 0.25*etaJ.*(one + psi*psiJ);    % 1x4 array
     % derivatives of x and y wrt psi and eta
     xpsi = NJpsi*xe';  ypsi = NJpsi*ye'; 
     xeta = NJeta*xe';  yeta = NJeta*ye';
     Jinv = [yeta, -xeta; -ypsi, xpsi];      % 2x2 array
     jcob = xpsi*yeta - xeta*ypsi;
     % derivatives of shape functions in element coordinates
     NJdpsieta = [NJpsi; NJeta];             % 2x4 array
     NJdxy = Jinv*NJdpsieta;                 % 2x4 array
     % assemble element stiffness matrix ke: 4x4 array
     ke = ke + (NJdxy(1,:))'*(NJdxy(1,:))/jcob ...
             + (NJdxy(2,:))'*(NJdxy(2,:))/jcob ...
             + NJ(1,:)'*NJ(1,:)*jcob;
     rhse = rhse + rhs*NJ'*jcob;
  end
end