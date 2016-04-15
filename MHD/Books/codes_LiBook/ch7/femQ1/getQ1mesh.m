
% Generate Q1 mesh on rectangle [0, length]x[0, height]
% nx,ny: number of elements in each direction
% x,y: 1-D array for nodal coordinates
% conn(1:ne,1:4): connectivity matrix
% ne, np: total numbers of elements, nodes generated

function[x,y,conn,ne,np] = getQ1mesh(length,height,nx,ny)

ne = nx*ny;
np = (nx+1)*(ny+1);

% create nodal coordinates
dx=length/nx;  dy=height/ny;
for i = 1:(nx+1)
   for j=1:(ny+1)
      x((ny+1)*(i-1)+j) = dx*(i-1);
      y((ny+1)*(i-1)+j) = dy*(j-1);
   end
end

% connectivity matrix: countclockwise start at low-left corner
for j=1:nx
   for i=1:ny
      ele = (j-1)*ny + i;
      conn(ele,1) = ele + (j-1);
      conn(ele,2) = conn(ele,1) + ny + 1;
      conn(ele,3) = conn(ele,2) + 1;
      conn(ele,4) = conn(ele,1) + 1;
   end
end