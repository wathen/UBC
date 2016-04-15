
function [solve] = TimeHarmonic
solve.GetMesh=@GetMesh;
solve.AssembleMatrix=@AssembleMatrix;
solve.DispSolution=@DispSolution;
solve.ApplyBC=@ApplyBC;
% nargout(TimeHarmonic)
% [Asol,Esol] = run;

    function [Asol,Esol] = run()
        
        close all
        clear
        
        % Initialising grid
        FEMmesh.a = 0;FEMmesh.b = 1;
        FEMmesh.n = 11;
        c = 1;
        
        % Creating mesh
        [FEMmesh] = GetMesh(FEMmesh);
        
        Exact = @(x,y) [cos(2*pi*x)*sin(2*pi*y);...
            -sin(2*pi*x)*cos(2*pi*y)];
        
        F = @(x,y) (8*pi^2+c)*[cos(2*pi*x)*sin(2*pi*y);...
            -sin(2*pi*x)*cos(2*pi*y)];
        
        % Exact = @(x,y) [y*(1-y);x*(1-x)];
        %
        % F = @(x,y) [2+c*y*(1-y);...
        %             2+c*x*(1-x)];
        %
        %         F = @(x,y) [1;1];
        
        % Assembling the stiffness, mass matricies and RHS
        [K,M,f] = AssembleMatrix(FEMmesh,F,'2point');
        
        A = K+c*M;
        U = A\f;
        
        % Displaying solution
        [Asol,Esol] = DispSolution(U,FEMmesh,Exact,'yes');
        
    end

    function [FEMmesh] = GetMesh(FEMmesh)
        a = FEMmesh.a; b = FEMmesh.b;
        n = FEMmesh.n;
        h = (b-a)/(n-1);
        
        for j = 1:n-1
            if j>=2
                i = 1:n-1;
                m = i+2*(j-1)*n-(j-1);
                NNode(i+(j-1)*n-(j-1),1) = m;
                NNode(i+(j-1)*n-(j-1),2) = m+n;
                NNode(i+(j-1)*n-(j-1),3) = m+2*n-1;
                NNode(i+(j-1)*n-(j-1),4) = m+n-1;
                y1 = i+2*(j-1)*n-(j-1);
                y2 = min(NNode(i+(j-1)*n-(j-1),3)-n):max(NNode(i+(j-1)*n-(j-1),3)-n)+1;
                Y(y1) = a+h*(j-1);
                Y(y2) = a+h*(j-1)+h/2;
                X(y1) = a+h/2+h*(0:n-2);
                X(y2) = a+h*(0:n-1);
            else
                i = 1:n-1;
                NNode(i+(j-1)*n,1) = i+(j-1)*n;
                NNode(i+(j-1)*n,2) = i+j*n;
                NNode(i+(j-1)*n,3) = i+j*2*n-1;
                NNode(i+(j-1)*n,4) = i+j*n-1;
                y1 = i+(j-1)*n;
                y2 = min(i+j*n-1):max(i+j*n-1)+1;
                Y(y1) = a;
                Y(y2) = a+h/2;
                X(y1) = a+h/2+h*(0:n-2);
                X(y2) = a+h*(0:n-1);
            end
            
        end
        X(y1+2*n-1) = a+h/2+h*(0:n-2);
        Y(y1+2*n-1) = a+h*j;
        
        FEMmesh.NNode = NNode;
        FEMmesh.X = X;
        FEMmesh.Y = Y;
        %         = struct('NNode',NNode,'X',X,'Y',Y);
    end

    function [K,M,f,U] = AssembleMatrix(FEMmesh,F,quad)
        
        a = FEMmesh.a; b = FEMmesh.b;
        n = FEMmesh.n;
        
        % Number of cells
        NumCells = (n-1)^2;
        
        h = (b-a)/(n-1);
        
        N0 = @(x,y) [1-y;0];
        N1 = @(x,y) [0;x];
        N2 = @(x,y) [y;0];
        N3 = @(x,y) [0;1-x];
        
        K = sparse(length(FEMmesh.X),length(FEMmesh.X));
        M = sparse(length(FEMmesh.X),length(FEMmesh.X));
        f = sparse(length(FEMmesh.X),1);
        
        Int0 = @(x,y,i,j) F(x,y)'*N0((x-(i-1)*h)/h,(y-(j-1)*h)/h);
        Int1 = @(x,y,i,j) F(x,y)'*N1((x-(i-1)*h)/h,(y-(j-1)*h)/h);
        Int2 = @(x,y,i,j) F(x,y)'*N2((x-(i-1)*h)/h,(y-(j-1)*h)/h);
        Int3 = @(x,y,i,j) F(x,y)'*N3((x-(i-1)*h)/h,(y-(j-1)*h)/h);
        %         Int0 = @(x,y,i,j) F(x,y)'*N0(x,y);
        %         Int1 = @(x,y,i,j) F(x,y)'*N1(x,y);
        %         Int2 = @(x,y,i,j) F(x,y)'*N2(x,y);
        %         Int3 = @(x,y,i,j) F(x,y)'*N3(x,y);
        
        ii = 0;
        jj = 1;
        U = cell(NumCells,1);
        TF = strcmp('mid',quad);
        
        for e = 1:NumCells
            ii = ii+1;
            el=FEMmesh.NNode(e,:);
            
            %             Kh = h^(-2)*ones(4,4);
            Kh = h^(-2)*[1 1 -1 -1; 1 1 -1 -1; -1 -1 1 1;-1 -1 1 1];
            Mh = [1/3  0  1/6  0;
                0  1/3   0 1/6;
                1/6  0   1/3  0;
                0  1/6  0  1/3];
            
            if TF == 1
                %                 f1 = 1;
                %                 fl = f1*h^2*?[1;1;-1;-1]/2;
                fl = [Int0(FEMmesh.X(el(1)),FEMmesh.Y(el(1))+.5*h,ii,jj);...
                    Int1(FEMmesh.X(el(1)),FEMmesh.Y(el(1))+.5*h,ii,jj);...
                    Int2(FEMmesh.X(el(1)),FEMmesh.Y(el(1))+.5*h,ii,jj);...
                    Int3(FEMmesh.X(el(1)),FEMmesh.Y(el(1))+.5*h,ii,jj)];
            else
                
                xb = ii*h;
                xaa = (ii-1)*h;
                yb = jj*h;
                yaa = (jj-1)*h;
                
                x1 = @(a,b) (b+a)/2+ (b-a)*sqrt(1/3)/2;
                x2 = @(a,b) (b+a)/2- (b-a)*sqrt(1/3)/2;
                
                t1 = x1(xaa,xb);
                t2 = x2(xaa,xb);
                t11 = x1(yaa,yb);
                t22 = x2(yaa,yb);
                
                f1 = Int0(t1,t11,ii,jj)+Int0(t1,t22,ii,jj)+Int0(t2,t11,ii,jj)+Int0(t2,t22,ii,jj);
                f2 = Int1(t1,t11,ii,jj)+Int1(t1,t22,ii,jj)+Int1(t2,t11,ii,jj)+Int1(t2,t22,ii,jj);
                f3 = Int2(t1,t11,ii,jj)+Int2(t1,t22,ii,jj)+Int2(t2,t11,ii,jj)+Int2(t2,t22,ii,jj);
                f4 = Int3(t1,t11,ii,jj)+Int3(t1,t22,ii,jj)+Int3(t2,t11,ii,jj)+Int3(t2,t22,ii,jj);
                
                fl = [f1;f2;f3;f4]/4;
            end
            
            K(el,el) = K(el,el) + Kh;
            M(el,el) = M(el,el) + Mh;
            
            %             if mod(e,2) == 0
            f(el) = f(el)+fl;
            %             else
            %                 f(el) = f(el)-fl;
            %             end
            if mod(e,n-1) == 0
                ii = 0;
                jj = jj+1;
            end
            U{e} = (Kh+Mh)\fl;
        end
    end

    function [A,f] = ApplyBC(K,M,c,f,FEMmesh)
        
        boundary_elements = [1:FEMmesh.n-1];
        jj = FEMmesh.n;
        k = FEMmesh.n;
        for i = 1:FEMmesh.n-1
            boundary_elements(jj) = k;
            boundary_elements(jj+1) = k+FEMmesh.n-1;
            k = k+2*FEMmesh.n-1;
            jj = jj+2;
        end
        Bmax = max(max(FEMmesh.NNode));
        boundary_elements = [boundary_elements,boundary_elements(end)+1:Bmax];
        
        A =K+c*M;
        zero = sparse(length(f),1);
        for i = 1:length(boundary_elements)
            A(boundary_elements(i),:) = zero';
            A(:,boundary_elements(i)) = zero;
            A(boundary_elements(i),boundary_elements(i)) = 1;
        end
        f(boundary_elements) = 0;
        
    end

    function [approx,exact] = DispSolution(U,FEMmesh,Exact,graph)
        a = FEMmesh.a; b = FEMmesh.b;
        n = FEMmesh.n;
        
        % Number of cells
        NumCells = (n-1)^2;
        
        h = (b-a)/(n-1);
        
        approxsol = @(u,x,y) u(1)*[1-y 0] + u(2)*[0 x] + u(3)*[y 0] + u(4)*[0 1-x];
        %         approxsolhat = @(u,x,y,i,j) u(1)*[1-(y-(j-1)*h)/h 0] + u(2)*[0 (x-(i-1)*h)/h] + u(3)*[-(y-(j-1)*h)/h 0] + u(4)*[0 (x-(i-1)*h)/h-1];
        ii = 0;
        jj = 1;
        uu = cell(n-1,n-1);
        uExact = cell(n-1,n-1);
        for ee = 1:NumCells
            ii = ii+1;
            ele=FEMmesh.NNode(ee,:);
            ul = U(ele);
            %             ul = [1;1;-1;-1];
            x = FEMmesh.X(ele(1));
            y = FEMmesh.Y(ele(1))+.5*h;
            xx(jj,ii) = x;
            yy(jj,ii) = y;
            uu{jj,ii} = approxsol(ul,0.5,0.5);
            %             uuhat{jj,ii} = approxsolhat(ul,FEMmesh.X(ele(1)),FEMmesh.Y(ele(1))+.5*h,ii,jj);
            uExact{jj,ii} = Exact(x,y);
            xe(jj,ii) = uExact{jj,ii}(1);
            ye(jj,ii) = uExact{jj,ii}(2);
            %             xahat(jj,ii) = uuhat{jj,ii}(1);
            %             yahat(jj,ii) = uuhat{jj,ii}(2);
            xa(jj,ii) = uu{jj,ii}(1);
            ya(jj,ii) = uu{jj,ii}(2);
            if mod(ee,n-1) == 0
                ii = 0;
                jj = jj+1;
            end
            
        end
        
        TF = strcmp('yes',graph);
        if TF ==1
            figure('Position', [100, 100, 1049, 895]);
            subplot(2,1,1); quiver(xx,yy,xe,ye);
            subplot(2,1,2); quiver(xx,yy,xa,ya);
        end
        
        approx = struct('x',full(xa),'y',full(ya));
        exact = struct('x',xe,'y',ye);
        
    end

end