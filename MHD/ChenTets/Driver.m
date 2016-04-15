% Driver function for ChenTets

g =[2 0 1 0 0 1 0; 2 1 1 0 1 1 0;2 1 0 1 1 1 0;2 0 0 1 0 1 0]';

[p,e,t] = initmesh(g,'hmax',0.1);

[A11,M11,L1,S11,Q11,B11,u_b,p_b]=our_matrices(p,t,e,list_of_int_edges,...
    list_of_bnd_edges,list_of_int_nodes,list_of_bnd_nodes);

