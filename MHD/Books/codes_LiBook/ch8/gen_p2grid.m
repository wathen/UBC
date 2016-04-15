%--------------------------------------------------------
% Generate the 6-node triangular mesh
%    ne: total number of elements
%    np: total number of nodes
%    p(1:np,1:2):  (x,y) coordinates of all nodes
%    conn(1:ne,1:6): connectivity matrix
%    efl(1:ne,1:6): nodal type indicator
%    gfl(1:np,1): global nodal type indicator
%--------------------------------------------------------
function [ne,np,p,conn,efl,gfl] = gen_p2grid(nref)

% create an initial mesh with eight 6-node triangles
ne = 8;

x(1,1) = 0.0; y(1,1) = 0.0; efl(1,1)=0;  % 1st element
x(1,2) = 1.0; y(1,2) = 0.0; efl(1,2)=1;
x(1,3) = 1.0; y(1,3) = 1.0; efl(1,3)=1;

x(2,1) = 0.0; y(2,1) = 0.0; efl(2,1)=0;  % 2nd element
x(2,2) = 1.0; y(2,2) = 1.0; efl(2,2)=1;
x(2,3) = 0.0; y(2,3) = 1.0; efl(2,3)=1;

x(3,1) = 0.0; y(3,1) = 0.0; efl(3,1)=0;  % 3rd element
x(3,2) = 0.0; y(3,2) = 1.0; efl(3,2)=1;
x(3,3) =-1.0; y(3,3) = 1.0; efl(3,3)=1;

x(4,1) = 0.0; y(4,1) = 0.0; efl(4,1)=0;  % 4th element
x(4,2) =-1.0; y(4,2) = 1.0; efl(4,2)=1;
x(4,3) =-1.0; y(4,3) = 0.0; efl(4,3)=1;

x(5,1) = 0.0; y(5,1) = 0.0; efl(5,1)=0;  % 5th element
x(5,2) =-1.0; y(5,2) = 0.0; efl(5,2)=1;
x(5,3) =-1.0; y(5,3) =-1.0; efl(5,3)=1;

x(6,1) = 0.0; y(6,1) = 0.0; efl(6,1)=0;  % 6th element
x(6,2) =-1.0; y(6,2) =-1.0; efl(6,2)=1;
x(6,3) = 0.0; y(6,3) =-1.0; efl(6,3)=1;

x(7,1) = 0.0; y(7,1) = 0.0; efl(7,1)=0;  % 7th element
x(7,2) = 0.0; y(7,2) =-1.0; efl(7,2)=1;
x(7,3) = 1.0; y(7,3) =-1.0; efl(7,3)=1;

x(8,1) = 0.0; y(8,1) = 0.0; efl(8,1)=0;  % 8th element
x(8,2) = 1.0; y(8,2) =-1.0; efl(8,2)=1;
x(8,3) = 1.0; y(8,3) = 0.0; efl(8,3)=1;

for ie=1:8   % mid-edge nodes: other examples are different
   x(ie,4) = 0.5*(x(ie,1)+x(ie,2));
   y(ie,4) = 0.5*(y(ie,1)+y(ie,2)); efl(ie,4)=0;
   x(ie,5) = 0.5*(x(ie,2)+x(ie,3));
   y(ie,5) = 0.5*(y(ie,2)+y(ie,3)); efl(ie,5)=1;
   x(ie,6) = 0.5*(x(ie,3)+x(ie,1));
   y(ie,6) = 0.5*(y(ie,3)+y(ie,1)); efl(ie,6)=0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% uniformly refine the grid
 for i=1:nref
   nm = 0; % count the new elements from each refinement
   for j=1:ne   % loop over current elements
      for k=1:4   % generate 4 sub-elements
        nm = nm+1;  % increse the element number by 1
        if (k==1) p1=1; p2=4; p3=6; end   % 1st sub-ele nodes
        if (k==2) p1=4; p2=2; p3=5; end   % 2nd sub-ele nodes
        if (k==3) p1=6; p2=5; p3=3; end   % 3rd sub-ele nodes
        if (k==4) p1=4; p2=5; p3=6; end   % 4th sub-ele nodes
        
        xn(nm,1)=x(j,p1); yn(nm,1)=y(j,p1); efln(nm,1)=efl(j,p1); 
        xn(nm,2)=x(j,p2); yn(nm,2)=y(j,p2); efln(nm,2)=efl(j,p2);
        xn(nm,3)=x(j,p3); yn(nm,3)=y(j,p3); efln(nm,3)=efl(j,p3);
        xn(nm,4) = 0.5*(xn(nm,1)+xn(nm,2));   % mid-edge node
        yn(nm,4) = 0.5*(yn(nm,1)+yn(nm,2));
        xn(nm,5) = 0.5*(xn(nm,2)+xn(nm,3));
        yn(nm,5) = 0.5*(yn(nm,2)+yn(nm,3));
        xn(nm,6) = 0.5*(xn(nm,3)+xn(nm,1));
        yn(nm,6) = 0.5*(yn(nm,3)+yn(nm,1));

        if (efln(nm,1)==1 & efln(nm,2)==1)   % nodal type indicator
            efln(nm,4) = 1; 
        else
            efln(nm,4) = 0;    
        end
        if (efln(nm,2)==1 & efln(nm,3)==1) 
            efln(nm,5) = 1; 
        else
            efln(nm,5) = 0;    
        end
        if (efln(nm,3)==1 & efln(nm,1)==1) 
            efln(nm,6) = 1; 
        else
            efln(nm,6) = 0;    
        end
      end
    end % end of loop over current elements

    ne = 4*ne;  % number of elements increased by factor of four
    for k=1:ne     % relabel the new points
       for l=1:6
          x(k,l)=xn(k,l); y(k,l)=yn(k,l);  efl(k,l)=efln(k,l);
       end
    end
end % end of refinement loop

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define the global nodes and the connectivity table
% we set the first element: nodes and connectivity
p(1,1)=x(1,1); p(1,2)=y(1,1); gfl(1,1)=efl(1,1);
p(2,1)=x(1,2); p(2,2)=y(1,2); gfl(2,1)=efl(1,2);
p(3,1)=x(1,3); p(3,2)=y(1,3); gfl(3,1)=efl(1,3);
p(4,1)=x(1,4); p(4,2)=y(1,4); gfl(4,1)=efl(1,4);
p(5,1)=x(1,5); p(5,2)=y(1,5); gfl(5,1)=efl(1,5);
p(6,1)=x(1,6); p(6,2)=y(1,6); gfl(6,1)=efl(1,6);

conn(1,1) = 1;  conn(1,2) = 2;  conn(1,3) = 3;  
conn(1,4) = 4;  conn(1,5) = 5;  conn(1,6) = 6;  

np = 6;         % we already have 6 nodes from 1st element
eps = 1.0e-8;

for i=2:ne        % loop over the rest elements
 for j=1:6        % loop over nodes of each element

 Iflag=0;
 for k=1:np
  if(abs(x(i,j)-p(k,1)) < eps & abs(y(i,j)-p(k,2)) < eps)
     Iflag = 1;    % the node has been recorded previously
     conn(i,j) = k;   % the jth local node of element i 
   end
 end

 if(Iflag==0)  % record the node
   np = np+1;
   p(np,1)=x(i,j); p(np,2)=y(i,j); gfl(np,1) = efl(i,j);
   % the jth local node of element i becomes the new global node
   conn(i,j) = np;   
 end

 end
end  % end of loop over elements

return;