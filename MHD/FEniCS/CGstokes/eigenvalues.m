clear
close all
load('results/DoF.mat');
load('results/Vdim.mat');
% Vdim = vect;
A = cell([1,6]);
P = cell(1,6);
for i = 1:3
    Amat = sprintf('results/A%i.mat',DoF(i));
    Pmat = sprintf('results/P%i.mat',DoF(i));
    
    A = load( Amat );

    P = load (Pmat);
    
        
    A = struct2cell(A);
    A = A{1};        
    P = struct2cell(P);
    P = P{1};
    
    a11 = A(1:VDoF(i),1:VDoF(i));
    a21 = A(VDoF(i)+1:end,1:VDoF(i));
    a12 = a21';
    PP = blkdiag(a11,a21*(a11\a12));

    
    d = eig(full(P\A));
    dd = eig(full(PP\A));
    ddd = eig(full(P\PP));
    
    
    
    subplot(1,2,1);plot(sort(real(d)),'g*');hold on
    plot(sort(real(dd)),'ko')
    title(['eigenvalues for ideal vs approx ',num2str(DoF(i))])
    subplot(1,2,2);plot(sort(real(ddd)),'h')
    title(['eigenvalues for ideal vs approx ',num2str(DoF(i))])
    figure;
end