clear
close all
iter = 1;
size = 16;

kappa = 10;
nu = 1/1;
nu_m = 10;

Type = 'FLUID';

if nu<1
    dir = strcat(Type,'matrix_kappa=',num2str(kappa),'.0_nu_m=',num2str(nu_m),'.0_nu=',num2str(nu),'/');
else
    dir = strcat(Type,'matrix_kappa=',num2str(kappa),'.0_nu_m=',num2str(nu_m),'.0_nu=',num2str(nu),'.0/');
end
A = load(strcat(dir,'mass_',num2str(iter),'_',num2str(size)));
A = A.(strcat('mass_',num2str(iter),'_',num2str(size)));
B = load(strcat(dir,'Schur_',num2str(iter),'_',num2str(size)));
B = B.(strcat('Schur_',num2str(iter),'_',num2str(size)));

tic
D = eig(full(B\A));
toc
figure
a = plot(sort(real((D))),'*');
title(strcat('kappa=',num2str(kappa),'_.nu_m=',num2str(nu_m),'_.nu=',num2str(nu),'_.n=',num2str(size)),'FontSize',18)
saveas(a,strcat('LaTex/',Type,'kappa=',num2str(kappa),'_nu_m=',num2str(nu_m),'_nu=',num2str(nu),'_n=',num2str(size)),'png')