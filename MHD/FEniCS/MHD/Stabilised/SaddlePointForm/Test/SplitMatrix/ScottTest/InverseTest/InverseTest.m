level = 2;
dimensions = load('Matrix/dimensions.t');
n_u = dimensions(level,1);
n_b = dimensions(level,3);
m_u = dimensions(level,2);
m_b = dimensions(level,4);

F = load(strcat('Matrix/F',num2str(level)));
F = F.(strcat('F',num2str(level)));
B = load(strcat('Matrix/B',num2str(level)));
B = B.(strcat('B',num2str(level)));
D = load(strcat('Matrix/D',num2str(level)));
D = D.(strcat('D',num2str(level)));
C = load(strcat('Matrix/C',num2str(level)));
C = -C.(strcat('C',num2str(level)));
M = load(strcat('Matrix/M',num2str(level)));
M = M.(strcat('M',num2str(level)));
L = load(strcat('Matrix/L',num2str(level)));
L = L.(strcat('L',num2str(level)));

uB = load(strcat('Matrix/vBoundary',num2str(level),'.t'));
bB = load(strcat('Matrix/bBoundary',num2str(level),'.t'));
rB = load(strcat('Matrix/rBoundary',num2str(level),'.t'));

F(uB+1,:) = [];
F(:,uB+1) = [];
B(rB+1,:) = [];
B(:,uB+1) = [];
M(bB+1,:) = [];
M(:,bB+1) = [];
D(rB+1,:) = [];
D(:,bB+1) = [];
L(rB+1,:) = [];
L(:,rB+1) = [];
C(bB+1,:) = [];
C(:,uB+1) = [];

n_b-rank(full(M))
rank(full(B'))


Inv2by2 = @(invA, invD, A, B, C, D) [inv(A-B*invD*C) -invA*B*inv(D-C*invA*B);
                                     -invD*C*inv(A-B*invD*C) inv(D-C*invA*B)];
InvSaddle = @(invA, A, B, C) [invA+invA*B*inv(-C*invA*B)*C*invA, -invA*B*inv(-C*invA*B);
                              -inv(-C*invA*B)*C*invA, inv(-C*invA*B)];
InvNullSaddle = @(invBB, W, A, B) [W, (eye(length(A)) - W*A)*B'*invBB;
                                   invBB*B*(speye(length(A)) - A*W), -invBB*B*(A-A*W*A)*B'*invBB];
                 
Kns = [F,B';B, zeros(rank(full(B')))];
norm(full(InvSaddle(inv(F),F,B',B)-inv(Kns)))


Km = [M,D';D, zeros(rank(full(D)))];
Z = null(full(D));
W = Z*inv(Z'*M*Z)*Z';
norm(full(InvNullSaddle(inv(D*D'),W,M,D)-inv(Km)))

AA = InvNullSaddle(inv(D*D'),W,M,D)-inv(Km);
% AA(end,1:16)
% AA(1:16,end)




% 
% 
%                  
         