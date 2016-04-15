clear
close all
iter = 1;
s = 1;
x = 1;
SS = [1];
NU = [10];
table = zeros(length(SS),length(NU));
for i = 1:length(NU)
    nu_m = NU(i);
    x = x+1;
    y = 0;
    for j = 1:length(SS)
        s = 4;
        kappa = SS(j);
        dimensions = load('dimensions.t');
        n_u = dimensions(log2(s),1);
        n_b = dimensions(log2(s),3);
        m_u = dimensions(log2(s),2);
        m_b = dimensions(log2(s),4);
        
%         kappa = 1;
        nu = 1.0;
%         nu_m = 10;
        
        Type = 'TEST';
        
        if nu<1
            dir = strcat(Type,'matrix_kappa=',num2str(kappa),'.0_nu_m=',num2str(nu_m),'.0_nu=',num2str(nu),'/');
        else
            dir = strcat(Type,'matrix_kappa=',num2str(kappa),'.0_nu_m=',num2str(nu_m),'.0_nu=',num2str(nu),'.0/');
        end
        A = load(strcat(dir,'A_',num2str(iter),'_',num2str(s)));
        A = A.(strcat('A_',num2str(iter),'_',num2str(s)));
        B = load(strcat(dir,'P_',num2str(iter),'_',num2str(s)));
        B = B.(strcat('P_',num2str(iter),'_',num2str(s)));
        Binner = load(strcat(dir,'Pinner_',num2str(iter),'_',num2str(s)));
        Binner = Binner.(strcat('Pinner_',num2str(iter),'_',num2str(s)));
        Bapprox = load(strcat(dir,'Papprox_',num2str(iter),'_',num2str(s)));
        Bapprox = Bapprox.(strcat('Papprox_',num2str(iter),'_',num2str(s)));
        SM = load(strcat(dir,'SM_',num2str(iter),'_',num2str(s)));
        SM = SM.(strcat('SM_',num2str(iter),'_',num2str(s)));
        
        M = load(strcat(dir,'M_',num2str(iter),'_',num2str(s)));
        M = M.(strcat('M_',num2str(iter),'_',num2str(s)));
        Mass = M;
        C = A(1:n_u,n_u+m_u+1:n_u+m_u+n_b)';
        stab = -A(n_u+1:n_u+m_u,n_u+1:n_u+m_u);
        nullity = size(C,2)-rank(full(C));
        MX = Bapprox(n_u+m_u+1:n_u+m_u+n_b,n_u+m_u+1:n_u+m_u+n_b);
        F = A(1:n_u,1:n_u);
        S =  -Bapprox(n_u+1:n_u+m_u,n_u+1:n_u+m_u);
        BB = A(n_u+1:n_u+m_u,1:n_u);
        total = sum(dimensions,2);
        D = A(1+total(log2(s))-m_b:end,n_u+m_u+1:n_u+m_u+n_b)';
        L = B(1+total(log2(s))-m_b:end,1+total(log2(s))-m_b:end);
        M = A(n_u+m_u+1:n_u+m_u+n_b,n_u+m_u+1:n_u+m_u+n_b);
        AA = [F,C';-C MX];
        PP = [(F+C'*inv(MX)*C), C';0*C MX];
        % syms x
        % A(1:n_u,1:n_u) = -A(1:n_u,1:n_u);
        % A(n_u+1:n_u+m_u,n_u+1:n_u+m_au) = 0;
        u = null(full(C));
        u = eye(n_u);
        NullM = null(full(M));
        b = eye(n_b);
        % b = b(:,1:size(u,2));
        % NullC = rand(size(u,1),size(u,2));
%         Eig = [u, zeros(size(u,1),n_b);
%             -(-stab+S)\BB*u, zeros(size(BB,1),n_b);
%             zeros(n_b,size(u,2)), b;
%             zeros(m_b,size(u,2)),  L\D'*b];
        % Eig = [zeros(n_u,size(NullM,2));
        %         zeros(m_u,size(NullM,2));
        %         NullM;
        %         L\D'*NullM];
        %
%         norm(A*Eig-Bapprox*Eig)
        % Bapprox(n_u+m_u+1:n_u+m_u+n_b,1+total(log2(s))-m_b:end) = D;
        
        tic
        Approx = C'*inv(MX)*C;
        toc
        size(F)
        size(Approx)
        size(Mass)
        %     Approx = C'*inv(MX)*C;
        [v,e] = eig(full(F+Approx),full(F+SM));
        e = diag(e);
%         table(j,1) = sum(dimensions(log2(s),:));
        table(j,i) = max(real(e));
%         figure
%         a = plot((sort((e))),'*');
%         set(gca,'FontSize',16)
%         xlabel('Real \lambda','FontSize',16)
%         ylabel('Imag \lambda','FontSize',16)
        %     axis([0.95 1.05 -0.001 0.001])
        %     b = plot(
        %     saveas(a,strcat('FIGURES/ScaledMass_Linv','_',num2str(iter),'_',num2str(s),'.png'),'png');
    end
end

%%
close all
plot(sort(real((e))), '*')
% xlabel('Real \lambda')
% ylabel('Imag \lambda')
% axis([0.995 1.001 -0.0005 0.0005])
set(gca,'FontSize',19)

%
% tic
% [v,e] = eig(full(A),full(Bapprox));
% toc
% e = diag(e);
%
%
% vi = imag(v);
% vr = real(v);
% vi(abs(vi) <= 1e-6) = 0;
% vr(abs(vr) <= 1e-6) = 0;
% v = vr +1i*vi;
% index1 = find(1-1e-5<real(e) & real(e)<1+1e-5);
% vv = v(:,index1);
% vu = vv(1:n_u,:);
% vp = vv(1+n_u:n_u+m_u,:);
% vb = vv(1+n_u+m_u:n_u+m_u+n_b,:);
% vr = vv(1+n_u+m_u+n_b:end,:);
%
fprintf('Eigenvalues of 1: %4.0i  and -1: %4.0i\n',numel(e(1-1e-8<real(e) & real(e)<1+1e-8)),numel(e(-1+1e-5>real(e) & real(e)>-1-1e-5 )))
%
% fprintf('Eigenvalues of 0.6....: %4.0i  and -1.6...: %4.0i\n',numel(e(-(1-sqrt(5))/2-1e-5<real(e) & real(e)<-(1-sqrt(5))/2+1e-5)),numel(e(-(1+sqrt(5))/2+1e-5>real(e) & real(e)>-(1+sqrt(5))/2-1e-5 )))
%
%
% if strcmp(Type,'MASS') == 1
%     fprintf('Eigenvalues of 9: %4.0i  and 6: %4.0i  and 3: %4.0i',numel(e(9-1e-2<real(e) & real(e)<9+1e-2)),...
%         numel(e(6-1e-2<real(e) & real(e)<6+1e-2)),numel(e(3-1e-2<real(e) & real(e)<3+1e-2)))
% else
%     fprintf('Eigenvalues of 3: %4.0i ',numel(e(3-1e-2<real(e) & real(e)<3+1e-2)))
%
% end
%
% dimensions(log2(s),:)
% % saveas(a,strcat('LaTex/',Type,'kappa=',num2str(kappa),'_nu_m=',num2str(nu_m),'_nu=',num2str(nu),'_n=',num2str(s)),'png')
%
%
%
% % Eigenvalue of 3 or (9+6+3) is (sum(dim)-)