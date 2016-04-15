from dolfin import *
from numpy import *
from block import *
from block.iterative import *
from block.algebraic.trilinos import *
import scipy as Sci
#import scipy.linalg
from math import pi,sin,cos,sqrt
import scipy.sparse as sps
import scipy.io as save
import scipy
import ipdb
import os

class FMaxwell(object):

    """docstring for Maxwell"""
    def __init__(self, n):
        self.n = n
        self.mesh = self.MeshGenerator(n)

    def MeshGenerator(self, n):
        mesh = UnitSquareMesh(n,n)
        return mesh

    def CreateTrialTestFuncs(self, mesh):
        V = FunctionSpace(mesh, "N1curl", 1)
        Q = FunctionSpace(mesh, "CG", 1)
        W = V * Q
        v, u = TestFunction(V), TrialFunction(V)
        q, p = TestFunction(Q), TrialFunction(Q)
        return V,Q,u,p,v,q



    def AssembleSystem(self, V,Q,u,p,v,q,f,u0,p0,k):
        # parameters.linear_algebra_backend = 'Epetra'
        def u0_boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, u0, u0_boundary)
        bc1 = DirichletBC(Q , p0, u0_boundary)

        a = dolfin.inner(curl(u),curl(v))*dx
        m = dolfin.inner(u,v)*dx
        b = dolfin.inner(v,grad(p))*dx
        bt = dolfin.inner(u,grad(q))*dx
        l = dolfin.inner(grad(p),grad(q))*dx
        rhs = dolfin.inner(f,v)*dx

        bcs = [bc,bc1]

        gamma = 1-k**2
        p11 = a+gamma*m
        p22 = l


        # GRAD = assemble((q*grad(u)))
        # bc.apply(GRAD)

        AA, AArhs = block_symmetric_assemble([[a-k*k*m, b],[bt,  0 ]], bcs=bcs)
        bb  = block_assemble([rhs, 0], bcs=bcs, symmetric_mod=AArhs)
        P,  Prhs= block_symmetric_assemble([[p11,0],[0,p22]],bcs=bcs)
        return AA,bb, P


    def StoreMatrix(self,A,name):
        Aarray = A.array()
        sA = sps.csr_matrix(Aarray)
        test ="".join([name,".mat"])
        scipy.io.savemat( test, {name: sA},oned_as='row')

    def Error(self, u,ue):
        err = ue - u
        L2normerr = sqrt(assemble(dolfin.inner(err,err)*dx))
        return L2normerr

    def SaveEpertaMatrix(self,A,name):
        from PyTrilinos import EpetraExt
        from numpy import array
        import scipy.sparse as sps
        import scipy.io
        test ="".join([name,".txt"])
        EpetraExt.RowMatrixToMatlabFile(test,A)
        data = loadtxt(test)
        col,row,values = data[:,0]-1,data[:,1]-1,data[:,2]
        Asparse = sps.csr_matrix((values, (row, col)))
        testmat ="".join([name,".mat"])
        scipy.io.savemat( testmat, {name: Asparse},oned_as='row')

    def LabelStucture(self, Values,Label):
        for x,y in zip(Values,Label):
            d.setdefault(y, []).append(x)
        return d

    # def minres(self,B, A, x, b, tolerance, maxiter, progress, relativeconv=False, shift=0, callback=None):
    # def minres(self,A,b,x,B,tolerance,maxiter,innertol):
    #     from block import *
    #     from block.iterative import *
    #     from block.algebraic.trilinos import *
    #     msg = [' beta1 = 0.  The exact solution is  x = 0          ',  # 0
    #            ' A solution to Ax = b was found, given rtol        ',  # 1
    #            ' A least-squares solution was found, given rtol    ',  # 2
    #            ' Reasonable accuracy achieved, given eps           ',  # 3
    #            ' x has converged to an eigenvector                 ',  # 4
    #            ' acond has exceeded 0.1/eps                        ',  # 5
    #            ' The iteration limit was reached                   ',  # 6
    #            ' Aname  does not define a symmetric matrix         ',  # 7
    #            ' Mname  does not define a symmetric matrix         ',  # 8
    #            ' Mname  does not define a pos-def preconditioner   ',  # 9
    #            ' Userdefined callback function returned true       ',  #10
    #            ' beta2 = 0.  If M = I, b and x are eigenvectors    ']  #-1

    #     callback_converged = False
    #     istop = 0;   itn = 0;     Anorm = 0.0;    Acond = 0.0;
    #     rnorm = 0.0; ynorm = 0.0; done  = False;
    #     callback=None
    #     eps = 1e-10

    #     #------------------------------------------------------------------
    #     # Set up y and v for the first Lanczos vector v1.
    #     # y  =  beta1 P' v1,  where  P = C**(-1).
    #     # v is really P' v1.
    #     #------------------------------------------------------------------

    #     [[P11,_],[_,P22]] =B
    #     print A
    #     r1 = b - A*x

    #     [r11,r12] = r1
    #     y1 = MinRes(P11,precond=ML(P11),tolerance=innertol,maxiter=50000,show=1)*r11
    #     y2 = MinRes(P22,precond=ML(P22),tolerance=innertol,maxiter=50000,show=1)*r12
    #     # y = block_vec([y1,y2])
    #     # y = B*r1

    #     # y =block_mat([[MinRes(P11,precond=ML(P11),tolerance=1e-6,maxiter=50000,show=1),0],[0,MinRes(P22,precond=ML(P22),tolerance=1e-6,maxiter=50000,show=1)]])*r1
    #     # print r1
    #     beta1 = inner(r11,y1)+inner(r12,y2)

    #     #  Test for an indefinite preconditioner.
    #     #  If b = 0 exactly, stop with x = 0.
    #     if beta1 < 0:
    #         raise ValueError('B does not define a pos-def preconditioner')
    #     if beta1 == 0:
    #         x *= 0
    #         return x, [0], [], []

    #     beta1 = sqrt(beta1);       # Normalize y to get v1 later.
    #     residuals = [beta1]


    #     # -------------------------------------------------------------------
    #     # Initialize other quantities.
    #     # ------------------------------------------------------------------
    #     oldb   = 0.0;     beta   = beta1;   dbar   = 0.0;     epsln  = 0.0
    #     qrnorm = beta1;   phibar = beta1;   rhs1   = beta1;   Arnorm = 0.0
    #     rhs2   = 0.0;     tnorm2 = 0.0;     ynorm2 = 0.0
    #     cs     = -1.0;    sn     = 0.0
    #     w  = 0*b
    #     w2 = 0*b
    #     r2 = r1.copy()

    #     # ---------------------------------------------------------------------
    #     # Main iteration loop.
    #     # --------------------------------------------------------------------
    #     while itn < maxiter:
    #         itn    += 1
    #         # progress += 1

    #         # -------------------------------------------------------------
    #         # Obtain quantities for the next Lanczos vector vk+1, k=1,2,...
    #         # The general iteration is similar to the case k=1 with v0 = 0:
    #         #
    #         #   p1      = Operator * v1  -  beta1 * v0,
    #         #   alpha1  = v1'p1,
    #         #   q2      = p2  -  alpha1 * v1,
    #         #   beta2^2 = q2'q2,
    #         #   v2      = (1/beta2) q2.
    #         #
    #         # Again, y = betak P vk,  where  P = C**(-1).
    #         # .... more description needed.
    #         # -------------------------------------------------------------
    #         s = 1.0/beta                # Normalize previous vector (in y).
    #         v = block_vec([s*y1,s*y2])                     # v = vk if P = I

    #         y = A*v
    #         # if shift:
    #         #     y -= shift*v

    #         if itn >= 2:
    #             y -= (beta/oldb)*r1
    #         y =block_vec([y1,y2]).blocks.all()
    #         # print v,y
    #         [v1,v2] = v
    #         alfa = inner(v1,y1 )+inner(v2,y2)
    #         [r21,r22] = r2      # alphak
    #         y1   -= alfa/beta*r21
    #         y2   -= alfa/beta*r22
    #         r11   = r21
    #         r12   = r21
    #         r21   = y1
    #         r22   = y2
    #         # y    = B*r2
    #         [r21,r22] = r2
    #         y1 = MinRes(P11,precond=ML(P11),tolerance=innertol,maxiter=50000,show=1)*r21
    #         y2 = MinRes(P22,precond=ML(P22),tolerance=innertol,maxiter=50000,show=1)*r22
    #         # y = block_vec([y1,y2])
    #         oldb   = beta               # oldb = betak
    #         beta   = inner(r21,y1)+inner(r22,y2)          # beta = betak+1^2
    #         if beta < 0:
    #             raise ValueError('B does not define a pos-def preconditioner')
    #         beta   = sqrt(beta)
    #         tnorm2 += alfa**2 + oldb**2 + beta**2

    #         if itn==1:                  # Initialize a few things.
    #             if beta/beta1 <= 10*eps:  # beta2 = 0 or ~ 0.
    #                 istop = -1            # Terminate later.

    #             # tnorm2 = alfa**2  ??
    #             gmax   = abs(alfa)      # alpha1
    #             gmin   = gmax             # alpha1

    #         # Apply previous rotation Qk-1 to get
    #         #   [deltak epslnk+1] = [cs  sn][dbark    0   ]
    #         #   [gbar k dbar k+1]   [sn -cs][alfak betak+1].

    #         oldeps = epsln
    #         delta  = cs * dbar  +  sn * alfa  # delta1 = 0         deltak
    #         gbar   = sn * dbar  -  cs * alfa  # gbar 1 = alfa1     gbar k

    #         # Note: There is severe cancellation in the computation of gbar
    #         #print ' sn = %21.15e\n dbar = %21.15e\n cs = %21.15e\n alfa = %21.15e\n sn*dbar-cs*alfa = %21.15e\n gbar =%21.15e' % (sn, dbar, cs, alfa, sn*dbar-cs*alfa, gbar)

    #         epsln  =               sn * beta  # epsln2 = 0         epslnk+1
    #         dbar   =            -  cs * beta  # dbar 2 = beta2     dbar k+1
    #         root   = sqrt(gbar**2 + dbar**2)
    #         Arnorm = phibar * root

    #         # Compute the next plane rotation Qk

    #         gamma  = sqrt(gbar**2 + beta**2)
    #         gamma  = max(gamma, eps)
    #         cs     = gbar / gamma             # ck
    #         sn     = beta / gamma             # sk
    #         phi    = cs * phibar              # phik
    #         phibar = sn * phibar              # phibark+1

    #         # Update  x.

    #         denom = 1.0/gamma
    #         w1    = w2
    #         w2    = w
    #         w     = (v - oldeps*w1 - delta*w2) * denom
    #         x    += phi*w

    #         # Go round again.

    #         gmax   = max(gmax, gamma)
    #         gmin   = min(gmin, gamma)
    #         z      = rhs1 / gamma
    #         ynorm2 = z**2  +  ynorm2
    #         rhs1   = rhs2 -  delta*z
    #         rhs2   =      -  epsln*z

    #         # Estimate various norms and test for convergence.

    #         Anorm  = sqrt(tnorm2)
    #         ynorm  = sqrt(ynorm2)
    #         epsa   = Anorm * eps
    #         epsx   = Anorm * ynorm * eps
    #         #epsr   = Anorm * ynorm * tolerance
    #         diag   = gbar
    #         if diag==0: diag = epsa

    #         qrnorm = phibar
    #         rnorm  = qrnorm
    #         test1  = rnorm / (Anorm*ynorm)     #  ||r|| / (||A|| ||x||)
    #         test2  = root  /  Anorm            # ||Ar|| / (||A|| ||r||)

    #         # Estimate  cond(A).
    #         # In this version we look at the diagonals of  R  in the
    #         # factorization of the lower Hessenberg matrix,  Q * H = R,
    #         # where H is the tridiagonal matrix from Lanczos with one
    #         # extra row, beta(k+1) e_k^T.

    #         Acond  = gmax/gmin

    #         # Call user provided callback with solution
    #         if callable(callback):
    #             callback_converged = callback(k=itn, x=x, r=test1)

    #         # See if any of the stopping criteria are satisfied.
    #         # In rare cases istop is already -1 from above (Abar = const*I)

    #         if istop == 0:
    #             t1 = 1 + test1      # These tests work if rtol < eps
    #             t2 = 1 + test2
    #             if t2 <= 1: istop = 2
    #             if t1 <= 1: istop = 1

    #             if itn >= maxiter: istop = 6
    #             if Acond >= 0.1/eps: istop = 4
    #             if epsx >= beta1: istop = 3
    #             # if rnorm <= epsx: istop = 2
    #             # if rnorm <= epsr: istop = 1
    #             if test2 <= tolerance: istop = 2
    #             if test1 <= tolerance: istop = 1

    #             if callback_converged: istop = 10

    #         residuals.append(test1)

    #         if istop != 0:
    #             break

    #     if istop != 1:
    #         print 'MinRes:', msg[istop]

    #     return x, residuals, [], []

