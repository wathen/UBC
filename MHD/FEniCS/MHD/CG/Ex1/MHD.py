#!/usr/bin/python
import petsc4py
import slepc4py
import sys

petsc4py.init(sys.argv)
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
Print = PETSc.Sys.Print
# from MatrixOperations import *
from dolfin import *
import numpy as np
import matplotlib.pylab as plt
import scipy.sparse as sps
import scipy.sparse.linalg as slinalg
import os
import scipy.io
from PyTrilinos import Epetra, EpetraExt, AztecOO, ML, Amesos
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
import PETScIO as IO

def remove_ij(x, i, j):

    # Remove the ith row
    idx = range(x.shape[0])
    idx.remove(i)
    x = x[idx,:]

    # Remove the jth column
    idx = range(x.shape[1])
    idx.remove(j)
    x = x[:,idx]

    return x

def StoreMatrix(A,name):
      test ="".join([name,".mat"])
      scipy.io.savemat( test, {name: A},oned_as='row')

m = 6
errL2u = np.zeros((m-1,1))
errL2p = np.zeros((m-1,1))
errL2b = np.zeros((m-1,1))
errL2r = np.zeros((m-1,1))
l2uorder = np.zeros((m-1,1))
l2porder = np.zeros((m-1,1))
l2border = np.zeros((m-1,1))
l2rorder = np.zeros((m-1,1))
NN = np.zeros((m-1,1))
DoF = np.zeros((m-1,1))
Velocitydim = np.zeros((m-1,1))
Electricdim = np.zeros((m-1,1))
Pressuredim = np.zeros((m-1,1))
Lagrangedim = np.zeros((m-1,1))
Wdim = np.zeros((m-1,1))
iterations = np.zeros((m-1,1))
SolTime = np.zeros((m-1,1))
udiv = np.zeros((m-1,1))
MU = np.zeros((m-1,1))
nn = 2

dim = 2
Solving = 'Direct'
ShowResultPlots = 'no'
ShowErrorPlots = 'no'
EigenProblem = 'no'
SavePrecond = 'no'
UseExactSchur = 'no'
CheckMu = 'no'
parameters['linear_algebra_backend'] = 'uBLAS'
MU[0]= 1e0
for xx in xrange(1,m):
    print xx
    nn = 2**(xx)


    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn

    mesh = RectangleMesh(-1, -1, 1, 1, nn, nn,'left')

    class Neumann(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1.0)

    class DirichletL(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], -1.0)

    class DirichletT(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1],1.0)
    class DirichletB(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1],-1.0)


    neumann = Neumann()
    dirichletL = DirichletL()
    dirichletT = DirichletT()
    dirichletB = DirichletB()

    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)

    neumann.mark(boundaries, 1)
    dirichletL.mark(boundaries, 2)
    dirichletT.mark(boundaries, 2)
    dirichletB.mark(boundaries, 2)

    parameters['reorder_dofs_serial'] = False
    Velocity = VectorFunctionSpace(mesh, "CG", 2)
    Pressure = FunctionSpace(mesh, "CG", 1)
    Electric = FunctionSpace(mesh, "N1curl", 2)
    Lagrange = FunctionSpace(mesh, "CG", 2)
    parameters['reorder_dofs_serial'] = False
    W = MixedFunctionSpace([Velocity,Pressure,Electric,Lagrange])
    # W = Velocity*Pressure*Electric*Lagrange
    Velocitydim[xx-1] = Velocity.dim()
    Pressuredim[xx-1] = Pressure.dim()
    Electricdim[xx-1] = Electric.dim()
    Lagrangedim[xx-1] = Lagrange.dim()
    Wdim[xx-1] = W.dim()
    print "\n\nW:  ",Wdim[xx-1],"Velocity:  ",Velocitydim[xx-1],"Pressure:  ",Pressuredim[xx-1],"Electric:  ",Electricdim[xx-1],"Lagrange:  ",Lagrangedim[xx-1],"\n\n"



    def boundary(x, on_boundary):
        return on_boundary
    u0 =Expression(("x[1]*x[1]","x[0]*x[0]"),cell=triangle)
    p0 = Expression("x[0]",cell=triangle)
    b0 = Expression(("1-x[1]*x[1]","1-x[0]*x[0]"),cell=triangle)
    r0 = Expression(("(1-x[0]*x[0])*(1-x[1]*x[1])"),cell=triangle)

    # u0 = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
    # p0 = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")
    # b0 = Expression(("0","0"))
    # r0 = Expression("0")


    bcu = DirichletBC(W.sub(0),u0, boundary)
    bcp = DirichletBC(W.sub(1),p0, boundary)
    bcb = DirichletBC(W.sub(2),b0, boundary)
    bcr = DirichletBC(W.sub(3),r0, boundary)
    bc = [bcu,bcb,bcr]

    (u, p, b, r) = TrialFunctions(W)
    (v, q, c,s ) = TestFunctions(W)
    k = 1
    K = 0
    Mu_m = 1e4
    MU = 1
    fns = Expression(("-2*mu+1+2*pow(x[0],2)*x[1] ","-2*mu+2*pow(x[1],2)*x[0] "),mu = MU, k = K)
    fns = Expression(("-2*mu+1+2*pow(x[0],2)*x[1] +k*(-2*x[0]+2*x[1])*(1-pow(x[0],2))","-2+2*pow(x[1],2)*x[0]-k*(-2*x[0]+2*x[1])*(1-pow(x[1],2))"),mu = MU, k = K)
    fm = Expression(("2*k*mu_m+2*x[0]*(pow(x[1],2)-1)-k*4*x[1]","2*k*mu_m+2*x[1]*(pow(x[0],2)-1)-k*4*x[0]"),k = K,mu_m = Mu_m)
    # fns = Expression(("120*x[0]*x[1]*(1-mu)+ 400*x[0]*pow(x[1],6)+(5*pow(x[0],4)-5*pow(x[1],4))*60*x[0]*x[1]*x[1]","60*(pow(x[0],2)-pow(x[1],2))*(1-mu)+400*pow(x[0],4)*pow(x[1],3)-(5*pow(x[0],4)-5*pow(x[1],4))*20*x[1]*x[1]*x[1]"), mu = MU)
    # fm= Expression(("(8*pow(pi,2)-C)*sin(2*pi*x[1])*cos(2*pi*x[0])","-(8*pow(pi,2)-C)*sin(2*pi*x[0])*cos(2*pi*x[1])"),C = k*k)

    "'Maxwell Setup'"
    a11 = Mu_m*inner(curl(c),curl(b))*dx-k*k*inner(b,c)*dx
    a12 = inner(c,grad(r))*dx
    a21 = inner(b,grad(s))*dx
    Lmaxwell  = inner(c, fm)*dx
    maxwell = a11+a12+a21


    "'NS Setup'"
    u_k = Function(Velocity)
    u_k.vector()[:] = u_k.vector()[:]*0
    n =FacetNormal(mesh)
    # g = -grad(inner(u0,n))+p0*n
    g = -grad(u0)*n+p0*n
    ds = Measure("ds")[boundaries]
    n = FacetNormal(mesh)
    a11 = MU*inner(grad(v), grad(u))*dx+inner((grad(u)*u_k),v)*dx+(1/2)*div(u_k)*inner(u,v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds(0)
    a12 = -div(v)*p*dx
    a21 = -div(u)*q*dx
    Lns  = inner(v, fns)*dx - inner(v,g)*ds(1)
    ns = a11+a12+a21


    "'Coupling term Setup'"
    b_k = Function(Electric)
    b_k.vector()[:] = b_k.vector()[:]*0+1
    CoupleTerm = K*inner(v[0]*b_k[1]-v[1]*b_k[0],curl(b))*dx - K*inner(u[0]*b_k[1]-u[1],b_k[0]*curl(c))*dx



    if (Solving == 'Direct'):

        eps = 1.0           # error measure ||u-u_k||
        # epsb = 1.0
        tol = 1.0E-10       # tolerance
        iter = 0            # iteration counter
        maxiter = 20        # max no of iterations allowed
        SolutionTime = 0
        while eps > tol and iter < maxiter:
                iter += 1
                uu = Function(W)
                tic()
                AA, bb = assemble_system(maxwell+ns, Lmaxwell + Lns, bc)
                As = AA.sparray()
                StoreMatrix(As,"A")
                VelPres = Velocitydim[xx-1][0] +Pressuredim[xx-1][0]
                Adelete = remove_ij(As,VelPres-1,VelPres-1)
                A = PETSc.Mat().createAIJ(size=Adelete.shape,csr=(Adelete.indptr, Adelete.indices, Adelete.data))
                print toc()
                b = np.delete(bb,VelPres-1,0)
                zeros = 0*b
                bb = IO.arrayToVec(b)
                x = IO.arrayToVec(zeros)
                ksp = PETSc.KSP().create()
                pc = PETSc.PC().create()
                ksp.setOperators(A)

                ksp.setFromOptions()
                print 'Solving with:', ksp.getType()

                # Solve!
                tic()
                # start = time.time()
                ksp.solve(bb, x)
                # %timit ksp.solve(bb, x)
                # print time.time() - start
                time = toc()
                print time
                SolutionTime = SolutionTime +time
                # print ksp.its

                X = IO.vecToArray(x)
                uu = X[0:Velocitydim[xx-1][0]]
                bb1 = X[VelPres-1:VelPres+Electricdim[xx-1][0]-1]

                u1 = Function(Velocity)
                u1.vector()[:] = u1.vector()[:] + uu
                diff = u1.vector().array() - u_k.vector().array()
                epsu = np.linalg.norm(diff, ord=np.Inf)

                b1 = Function(Electric)
                b1.vector()[:] = b1.vector()[:] + bb1
                diff = b1.vector().array() - b_k.vector().array()
                epsb = np.linalg.norm(diff, ord=np.Inf)
                eps = epsu+epsb
                print '\n\n\niter=%d: norm=%g' % (iter, eps)
                u_k.assign(u1)
                b_k.assign(b1)

        SolTime[xx-1] = SolutionTime/iter


        # ksp = PETSc.KSP().create()
        # ksp.setOperators(A)

        # ksp.setFromOptions()
        # print 'Solving with:', ksp.getType()

        # # Solve!
        # tic()
        # ksp.solve(bb, x)
        # SolTime[xx-1] = toc()
        # print "time to solve: ",SolTime[xx-1]
        # del AA,As


    if (Solving == 'Iterative'):

        if (UseExactSchur == 'yes'):
            Aschur = As[0:Vdim[xx-1][0],0:Vdim[xx-1][0]]
            Bschur = As[Vdim[xx-1][0]:,0:Vdim[xx-1][0]]
            Btschur = As[0:Vdim[xx-1][0],Vdim[xx-1][0]:]
            AinvB = slinalg.spsolve(Aschur,Btschur)
            schur = Bschur*AinvB
            PP = sps.block_diag((Aschur,schur))
            PP = PP.tocsr()
            P = PETSc.Mat().createAIJ(size=PP.shape,csr=(PP.indptr, PP.indices, PP.data))

        ksp = PETSc.KSP().create()
        pc = PETSc.PC().create()
        ksp.setFromOptions()
        ksp.setTolerances(1e-10)
        print 'Solving with:', ksp.setType('minres')
        # ksp.setPCSide(2)

        pc = ksp.getPC()
        pc.setOperators(P)
        pc.getType()
        ksp.setOperators(A,P)
        tic()
        ksp.solve(bb, x)
        SolTime[xx-1] = toc()
        print "time to solve: ",SolTime[xx-1]
        iterations[xx-1] =  ksp.its
        print "iterations = ", iterations[xx-1]
        del PP,Pb,Ps,AA,As




    if (Solving == 'Iterative' or Solving == 'Direct'):

        ue =Expression(("x[1]*x[1]","x[0]*x[0]"))
        pe = Expression("x[0]")
        be = Expression(("1-x[1]*x[1]","1-x[0]*x[0]"))
        re = Expression(("(1-x[0]*x[0])*(1-x[1]*x[1])"))
        # ue = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
        # pe = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")
        # be = Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])","-sin(2*pi*x[0])*cos(2*pi*x[1])"))
        # re = Expression("0")

        X = IO.vecToArray(x)
        xu = X[0:Velocitydim[xx-1][0]]
        ua = Function(Velocity)
        ua.vector()[:] = xu

        pp = X[Velocitydim[xx-1][0]:VelPres-1]
        # xp[-1] = 0
        # pa = Function(Pressure)
        # pa.vector()[:] = xp

        n = pp.shape
        pp = np.insert(pp,n,0)
        pa = Function(Pressure)
        pa.vector()[:] = pp

        pend = assemble(pa*dx)

        ones = Function(Pressure)
        ones.vector()[:]=(0*pp+1)
        pp = Function(Pressure)
        pp.vector()[:] = pa.vector().array()- assemble(pa*dx)/assemble(ones*dx)

        pInterp = interpolate(pe,Pressure)
        pe = Function(Pressure)
        pe.vector()[:] = pInterp.vector().array()
        const = - assemble(pe*dx)/assemble(ones*dx)
        pe.vector()[:] = pe.vector()[:]+const





        xb = X[VelPres-1:VelPres+Electricdim[xx-1][0]-1]
        ba = Function(Electric)
        ba.vector()[:] = xb

        xr = X[VelPres+Electricdim[xx-1][0]-1:]
        ra = Function(Lagrange)
        ra.vector()[:] = xr

        errL2u[xx-1] = errornorm(ue,ua,norm_type="L2", degree_rise=4,mesh=mesh)
        errL2p[xx-1] = errornorm(pe,pp,norm_type="L2", degree_rise=4,mesh=mesh)
        errL2b[xx-1] = errornorm(be, ba,norm_type="Hcurl", degree_rise=4,mesh=mesh)
        errL2r[xx-1] = errornorm(re,ra,norm_type="L2", degree_rise=4,mesh=mesh)

        if xx == 1:
            l2uorder[xx-1] = 0
        else:
            l2uorder[xx-1] =  np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1]))
            l2porder[xx-1] =  np.abs(np.log2(errL2p[xx-2]/errL2p[xx-1]))
            l2border[xx-1] =  np.abs(np.log2(errL2b[xx-2]/errL2b[xx-1]))
            l2rorder[xx-1] =  np.abs(np.log2(errL2r[xx-2]/errL2r[xx-1]))

        print errL2u[xx-1]
        print errL2p[xx-1]
        print errL2b[xx-1]
        print errL2r[xx-1]




if (ShowErrorPlots == 'yes'):
    plt.loglog(NN,errL2u)
    plt.title('Error plot for CG2 elements - Velocity L2 convergence = %f' % np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1]))))
    plt.xlabel('N')
    plt.ylabel('L2 error')


    plt.figure()

    plt.loglog(NN,errL2p)
    plt.title('Error plot for CG1 elements - Pressure L2 convergence = %f' % np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1]))))
    plt.xlabel('N')
    plt.ylabel('L2 error')

    plt.show()


if (Solving == 'Iterative' or Solving == 'Direct'):
    print "\n\n"
    print "          ==============================="
    print "                  Results Table"
    print "          ===============================\n\n"
    import pandas as pd
    if (Solving == 'Iterative' and CheckMu == 'no'):
        tableTitles = ["Total DoF","V DoF","Q DoF","# iters","Soln Time","V-L2","V-order","P-L2","P-order"]
        tableValues = np.concatenate((Wdim,Vdim,Qdim,iterations,SolTime,errL2u,l2uorder,errL2p,l2porder),axis=1)
    elif (Solving == 'Direct' and CheckMu == 'no'):

        tableTitlesNS = ["Total DoF","V DoF","P DoF","Soln Time","V-L2","V-order","P-L2","P-order"]
        tableValuesNS = np.concatenate((Wdim,Velocitydim,Pressuredim,SolTime,errL2u,l2uorder,errL2p,l2porder),axis=1)

        tableTitlesM = ["Total DoF","B DoF","R DoF","Soln Time","B-L2","B-order","R-L2","R-order"]
        tableValuesM = np.concatenate((Wdim,Electricdim,Lagrangedim,SolTime,errL2b,l2border,errL2r,l2rorder),axis=1)

    if (CheckMu == 'yes'):
        tableTitles = ["Total DoF","mu","# iters","Soln Time","V-L2","||div u_h||","P-L2"]
        tableValues = np.concatenate((Wdim,MU,iterations,SolTime,errL2u,udiv,errL2p),axis=1)


    dfns= pd.DataFrame(tableValuesNS, columns = tableTitlesNS)
    pd.set_option('precision',3)

    dfm = pd.DataFrame(tableValuesM, columns = tableTitlesM)
    pd.set_option('precision',3)
    print "Fluids"
    print dfns
    print "\n\n"
    print "Magnetic"
    print dfm
    print "\n\n"
    if (CheckMu == 'no'):
        print "Velocity Elements rate of convergence ", np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1])))
        print "Pressure Elements rate of convergence ", np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1])))
        print "Magnetic Elements rate of convergence ", np.log2(np.average((errL2b[0:m-2]/errL2b[1:m-1])))
        print "Lagrange Elements rate of convergence ", np.log2(np.average((errL2r[0:m-2]/errL2r[1:m-1])))
        print "\n\n"
    # print df.to_latex()

if (SavePrecond == 'yes'):
    scipy.io.savemat('eigenvalues/Wdim.mat', {'Wdim':Wdim-1},oned_as = 'row')


if (ShowResultPlots == 'yes'):

    plot(ua)
    plot(interpolate(ue,Velocity))

    plot(pp)
    plot(interpolate(pe,Pressure))

    plot(ba)
    plot(interpolate(be,Electric))

    plot(ra)
    plot(interpolate(re,Lagrange))

interactive()








