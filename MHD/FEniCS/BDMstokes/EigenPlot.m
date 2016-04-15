clear
% close all
load('eigenvalues/Wdim.mat');

for i = 1:4
   eVec = sprintf('results/e%i.mat',Wdim(i));
    
   e = load(eVec);
        
   e = struct2cell(e);
   e = e{1};        
    
   subplot(2,2,i)
   plot(sort(real(e)),'ko')
   title(['Generalised eigenvalue problem with n = ',num2str(Wdim(i)),' BDM'])
    
end

break

clear
close all
% load('results/DoF.mat');
load('eigenvalues/Wdim.mat');
% Vdim = vect;
% A = cell([1,6]);
% P = cell(1,6);
for i = 1:4
    fprintf('   %1.0f  \n',i)
    Amat = sprintf('eigenvalues/A%i.mat',Wdim(i));
    Pmat = sprintf('eigenvalues/P%i.mat',Wdim(i));
    
    A = load( Amat );

    P = load (Pmat);
    
        
    A = struct2cell(A);
    A = A{1};        
    P = struct2cell(P);
    P = P{1};
    
%     a11 = A(1:VDoF(i),1:VDoF(i));
%     a21 = A(VDoF(i)+1:end,1:VDoF(i));
%     a12 = a21';
%     PP = blkdiag(a11,a21*(a11\a12));

    
    e = eig(full(P\A));
%     dd = eig(full(PP\A));
%     ddd = eig(full(P\PP));
    
    
    emat = sprintf('results/e%i.mat',Wdim(i));
    
    save(emat,'e')
    
    
%     plot(sort(real(d)),'g*');
%     title(['eigenvalues for ideal vs approx ',num2str(Wdim(i))])
%     if i ~= 5
%         figure;
%     end
end
