clear
% close all
iter = 4;
size = 2;
dimensions = load('dimensions.t');
n_u = dimensions(log2(size),1);
n_b = dimensions(log2(size),3);
m_u = dimensions(log2(size),2);
m_b = dimensions(log2(size),4);

kappa = 1;
nu = 1/1;
nu_m = 10;

Type = 'EXACT';

if nu<1
    dir = strcat(Type,'matrix_kappa=',num2str(kappa),'.0_nu_m=',num2str(nu_m),'.0_nu=',num2str(nu),'/');
else
    dir = strcat(Type,'matrix_kappa=',num2str(kappa),'.0_nu_m=',num2str(nu_m),'.0_nu=',num2str(nu),'.0/');
end
A = load(strcat(dir,'A_',num2str(iter),'_',num2str(size)));
A = A.(strcat('A_',num2str(iter),'_',num2str(size)));
B = load(strcat(dir,'P_',num2str(iter),'_',num2str(size)));
B = B.(strcat('P_',num2str(iter),'_',num2str(size)));
tic
% e = eig(full(B\A));
e = eig(full(A),full(B));
toc
figure
a = plot(sort((e)),'o');
title(strcat('kappa=',num2str(kappa),'_.nu_m=',num2str(nu_m),'_.nu=',num2str(nu),'_.n=',num2str(size)),'FontSize',18)
% 
figure
plot((sort(real(e))),'*')


fprintf('Eigenvalues of 1: %4.0i  and -1: %4.0i\n',numel(e(1-1e-2<real(e) & real(e)<1+1e-2)),numel(e(-1+1e-2>real(e) & real(e)>-1-1e-2 )))
    
fprintf('Eigenvalues of 0.6....: %4.0i  and -1.6...: %4.0i\n',numel(e(-(1-sqrt(5))/2-1e-2<real(e) & real(e)<-(1-sqrt(5))/2+1e-2)),numel(e(-(1+sqrt(5))/2+1e-2>real(e) & real(e)>-(1+sqrt(5))/2-1e-2 )))


if strcmp(Type,'MASS') == 1
    fprintf('Eigenvalues of 9: %4.0i  and 6: %4.0i  and 3: %4.0i',numel(e(9-1e-2<real(e) & real(e)<9+1e-2)),...
        numel(e(6-1e-2<real(e) & real(e)<6+1e-2)),numel(e(3-1e-2<real(e) & real(e)<3+1e-2)))
else
    fprintf('Eigenvalues of 3: %4.0i ',numel(e(3-1e-2<real(e) & real(e)<3+1e-2)))
 
end

dimensions(log2(size),:)
% saveas(a,strcat('LaTex/',Type,'kappa=',num2str(kappa),'_nu_m=',num2str(nu_m),'_nu=',num2str(nu),'_n=',num2str(size)),'png')


% Eigenvalue of 3 or (9+6+3) is (sum(dim)-)