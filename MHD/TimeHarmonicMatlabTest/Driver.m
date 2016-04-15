% Driver function

close all
clear
clc

% Initialising grid
FEMmesh.a = 0;FEMmesh.b = 1;
FEMmesh.n =4;
c = 1;
i = 1;
fprintf(' NumCells  |     x-inf         x-2        rate-inf     |      y-inf         y-2        rate-inf \n')
fprintf('------------------------------------------------------------------------------------------------\n')
for n = 10
    FEMmesh.n =n;
    t = parTimeHarmonic;
    % Creating mesh
    % fprintf('Creating mesh.....\n')
    [FEMmesh] = t.GetMesh(FEMmesh);
    
    Exact = @(x,y) [cos(2*pi*x)*sin(2*pi*y);...
        -sin(2*pi*x)*cos(2*pi*y)];
    
    F = @(x,y) (8*pi^2+c)*[cos(2*pi*x)*sin(2*pi*y);...
        -sin(2*pi*x)*cos(2*pi*y)];
    
%     Exact = @(x,y) [y*(1-y);x*(1-x)];
%     
%     F = @(x,y) [2+c*y*(1-y);...
%         2+c*x*(1-x)];
%     
    
    % Assembling the stiffness, mass matricies and RHS
    % fprintf('Creating matricies.....\n')
    [K,M,f,U] = t.AssembleMatrix(FEMmesh,F,'mid');
    
    % Applying boundary condition
    % fprintf('Applying boundary conditions.....\n')
    [A,f] = t.ApplyBC(K,M,c,f,FEMmesh);
    
    % Solving problem
    % fprintf('Solving system.....\n')
    U = A\f;
    
    % Displaying solution
    % fprintf('Displaying results.....\n')
    [Asol,Esol] = t.DispSolution(U,FEMmesh,Exact,'2');
    
    % fprintf('\n\n\nNumber of cells  = %4.0d\n',(FEMmesh.n-1)^2)
    % fprintf('X inf-norm error   %4.4e\n', norm(Esol.x(:)-Asol.x(:),inf))
    % fprintf('X 2-norm error     %4.4e\n', norm(Esol.x(:)-Asol.x(:),2))
    % fprintf('X 1-norm error     %4.4e\n', norm(Esol.x(:)-Asol.x(:),1))
    % fprintf('\nY inf-norm error   %4.4e\n', norm(Esol.y(:)-Asol.y(:),inf))
    % fprintf('Y 2-norm error     %4.4e\n', norm(Esol.y(:)-Asol.y(:),2))
    % fprintf('Y 1-norm error     %4.4e\n', norm(Esol.y(:)-Asol.y(:),1))
    newinfx = norm(Esol.x(:)-Asol.x(:),inf);
    new2x = norm(Esol.x(:)-Asol.x(:),2);
    newinfy = norm(Esol.y(:)-Asol.y(:),inf);
    new2y = norm(Esol.y(:)-Asol.y(:),2);
    if i == 1
        fprintf('   %4.0d    |   %4.4e   %4.4e                 |    %4.4e   %4.4e             \n',...
            (n-1)^2,newinfx,new2x,newinfy,new2y) 
    else
        ratex = old2x*((nold-1)^2/(n-1)^2)/new2x;
        ratey = old2y*((nold-1)^2/(n-1)^2)/new2y;
        
        fprintf('   %4.0d    |   %4.4e   %4.4e   %4.4d    |    %4.4e   %4.4e   %4.4d  \n',...
            (n-1)^2,newinfx,new2x,ratex ,newinfy,new2y,ratey)
    end
    oldinfx = norm(Esol.x(:)-Asol.x(:),inf);
    old2x = norm(Esol.x(:)-Asol.x(:),2);
    oldinfy = norm(Esol.y(:)-Asol.y(:),inf);
    old2y = norm(Esol.y(:)-Asol.y(:),2);
    nold = n;
    i = i+1;
    
end