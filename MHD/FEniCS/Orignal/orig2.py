
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

def StoreMatrix(A,name):
      test ="".join([name,".mat"])
      scipy.io.savemat( test, {name: A},oned_as='row')

m = 8
errL2u = np.zeros((m-1,1))
errL2p = np.zeros((m-1,1))
l2uorder = np.zeros((m-1,1))
l2porder = np.zeros((m-1,1))
NN = np.zeros((m-1,1))
DoF = np.zeros((m-1,1))
Vdim = np.zeros((m-1,1))
Qdim = np.zeros((m-1,1))
Wdim = np.zeros((m-1,1))
iterations = np.zeros((m-1,1))
SolTime = np.zeros((m-1,1))
udiv = np.zeros((m-1,1))
MU = np.zeros((m-1,1))
nn = 2

dim = 2
Solving = 'Direct'
ShowResultPlots = 'yes'
ShowErrorPlots = 'no'
EigenProblem = 'no'
SavePrecond = 'no'
UseExactSchur = 'yes'
CheckMu = 'no'
case = 3
parameters['linear_algebra_backend'] = 'uBLAS'
MU[0]= 1e0
for xx in xrange(1,m):
    print xx
    nn = 2**(xx)

    if (CheckMu == 'yes'):
        if (xx != 1):
            MU[xx-1] = MU[xx-2]/10
    else:
        if (xx != 1):
            MU[xx-1] = MU[xx-2]
    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn

    mesh = RectangleMesh(-1,-1, 1, 1, nn, nn,'right')
    parameters['reorder_dofs_serial'] = False
    V = FunctionSpace(mesh, "N1curl", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    parameters['reorder_dofs_serial'] = False
    W = V*Q
    Vdim[xx-1] = V.dim()
    Qdim[xx-1] = Q.dim()
    Wdim[xx-1] = W.dim()
    print "\n\nV:  ",Vdim[xx-1],"Q:  ",Qdim[xx-1],"W:  ",Wdim[xx-1],"\n\n"
    def boundary(x, on_boundary):
        return on_boundary


    if case == 1:
        u0 = Expression(("0","0"))
        p0 = Expression("0")
    elif case == 2:
        u0 = Expression(("0","0"))
        p0 = Expression("0")
    elif case == 3:
        u0 = Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])","-sin(2*pi*x[0])*cos(2*pi*x[1])"))
        p0 = Expression("0")



    bc = DirichletBC(W.sub(0),u0, boundary)
    bc1 = DirichletBC(W.sub(1),p0, boundary)
    bcs = [bc,bc1]

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    c = 1
    if case == 1:
        f= Expression(("(8*pow(pi,2)-C)*sin(2*pi*x[1])*cos(2*pi*x[0])","-(8*pow(pi,2)-C)*sin(2*pi*x[0])*cos(2*pi*x[1])"),C = c)
    elif case == 2:
        f = Expression(("2-C*x[1]*(1-x[1])","2-C*x[0]*(1-x[0])"), C = c)
    elif case == 3:
        f = Expression(("(8*pow(pi,2)-C)*sin(2*pi*x[1])*cos(2*pi*x[0])","-(8*pow(pi,2)-C)*sin(2*pi*x[0])*cos(2*pi*x[1]) "),C = c)


    a11 = inner(curl(v),curl(u))*dx-c*inner(u,v)*dx
    a12 = inner(v,grad(p))*dx
    a21 = inner(u,grad(q))*dx
    L1  = inner(v, f)*dx
    a = a11+a12+a21


    tic()
    AA, bb = assemble_system(a, L1, bcs)
    As = AA.sparray()
    StoreMatrix(As,'A')
    A = PETSc.Mat().createAIJ(size=As.shape,csr=(As.indptr, As.indices, As.data))
    print toc()
    b = bb.array()
    zeros = 0*b
    del bb
    bb = IO.arrayToVec(b)
    x = IO.arrayToVec(zeros)


    if (Solving == 'Direct'):
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)

        ksp.setType(ksp.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(pc.Type.LU)
        # Solve!
        tic()
        ksp.solve(bb, x)
        SolTime[xx-1] = toc()
        print "time to solve: ",SolTime[xx-1]
        del AA,As


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
        if case == 1:
            ue = Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])","-sin(2*pi*x[0])*cos(2*pi*x[1])"))
            pe = Expression("0")
        elif case == 2:
            ue = Expression(("x[1]*(1-x[1])","x[0]*(1-x[0])"))
            pe = Expression("0")
        elif case == 3:
            ue = Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])","-sin(2*pi*x[0])*cos(2*pi*x[1])"))
            pe = Expression("0")

        u = interpolate(ue,V)
        p = interpolate(pe,Q)

        Nv  = u.vector().array().shape

        X = IO.vecToArray(x)
        x = X[0:Nv[0]]
        ua = Function(V)
        ua.vector()[:] = x

        pp = X[Nv[0]:]
        pa = Function(Q)

        pa.vector()[:] = pp

        errL2u[xx-1] = errornorm(ue,ua,norm_type="L2", degree_rise=4,mesh=mesh)
        errL2p[xx-1] = errornorm(pe,pa,norm_type="L2", degree_rise=4,mesh=mesh)

        if xx == 1:
            l2uorder[xx-1] = 0
        else:
            l2uorder[xx-1] =  np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1]))
            l2porder[xx-1] =  np.abs(np.log2(errL2p[xx-2]/errL2p[xx-1]))

        print errL2u[xx-1]
        print errL2p[xx-1]


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
        tableTitles = ["Total DoF","V DoF","Q DoF","Soln Time","V-L2","V-order","P-L2","P-order"]
        tableValues = np.concatenate((Wdim,Vdim,Qdim,SolTime,errL2u,l2uorder,errL2p,l2porder),axis=1)
    if (CheckMu == 'yes'):
        tableTitles = ["Total DoF","mu","# iters","Soln Time","V-L2","||div u_h||","P-L2"]
        tableValues = np.concatenate((Wdim,MU,iterations,SolTime,errL2u,udiv,errL2p),axis=1)


    df = pd.DataFrame(tableValues, columns = tableTitles)
    pd.set_option('precision',3)
    print df
    print "\n\n"
    if (CheckMu == 'no'):
        print "Velocity Elements rate of convergence ", np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1])))
        print "Pressure Elements rate of convergence ", np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1])))
    print df.to_latex()

if (SavePrecond == 'yes'):
    scipy.io.savemat('eigenvalues/Wdim.mat', {'Wdim':Wdim-1},oned_as = 'row')


if (ShowResultPlots == 'yes'):
    plot(ua)
    plot(interpolate(ue,V))

    plot(pa)
    plot(interpolate(pe,Q))

    interactive()







