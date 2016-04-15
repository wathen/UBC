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
import os
import scipy.io
from PyTrilinos import Epetra, EpetraExt, AztecOO, ML, Amesos
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
import PETScIO as IO

m = 6
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
nn = 2

dim = 2
Solving = 'Direct'
ShowResultPlots = 'no'
ShowErrorPlots = 'no'
EigenProblem = 'no'
SavePrecond = 'no'
case = 1
parameters['linear_algebra_backend'] = 'uBLAS'


for xx in xrange(1,m):
    print xx
    nn = 2**(xx)
    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn

    mesh = RectangleMesh(-1, -1, 1, 1, nn, nn,'right')

    parameters['reorder_dofs_serial'] = False
    V = FunctionSpace(mesh, "BDM", 2)
    Q = FunctionSpace(mesh, "DG", 1)
    parameters['reorder_dofs_serial'] = False
    W = V*Q
    Vdim[xx-1] = V.dim()
    Qdim[xx-1] = Q.dim()
    Wdim[xx-1] = W.dim()
    print "\n\nV:  ",Vdim[xx-1],"Q:  ",Qdim[xx-1],"W:  ",Wdim[xx-1],"\n\n"
    def boundary(x, on_boundary):
        return on_boundary


    if case == 1:
        u0 = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
        p0 = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")
    elif case == 2:
        u0 = Expression(("sin(pi*x[1])","sin(pi*x[0])"))
        p0 = Expression("sin(x[1]*x[0])")
    elif case == 3:
        u0 = Expression(("cos(2*pi*x[1])*sin(2*pi*x[0]) ","-cos(2*pi*x[0])*sin(2*pi*x[1]) "))
        p0 = Expression("sin(2*pi*x[0])*sin(2*pi*x[1]) ")



    bc = DirichletBC(W.sub(0),u0, boundary)
    bcs = [bc]

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    if case == 1:
        f = Expression(("120*x[0]*x[1]*(1-mu)","60*(pow(x[0],2)-pow(x[1],2))*(1-mu)"), mu = 1e0)
    elif case == 2:
        f = Expression(("pi*pi*sin(pi*x[1])+x[1]*cos(x[1]*x[0])","pi*pi*sin(pi*x[0])+x[0]*cos(x[1]*x[0])"))
    elif case == 3:
        f = Expression(("8*pi*pi*cos(2*pi*x[1])*sin(2*pi*x[0]) + 2*pi*cos(2*pi*x[0])*sin(2*pi*x[1])","2*pi*cos(2*pi*x[0])*sin(2*pi*x[1]) - 8*pi*pi*cos(2*pi*x[0])*sin(2*pi*x[1])"))

    N = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    alpha = 10.0
    gamma =10.0
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    d = 0
    a11 = inner(grad(v), grad(u))*dx \
        - inner(avg(grad(v)), outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
        - inner(outer(v('+'),N('+'))+outer(v('-'),N('-')), avg(grad(u)))*dS \
        + alpha/h_avg*inner(outer(v('+'),N('+'))+outer(v('-'),N('-')),outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
        - inner(outer(v,N), grad(u))*ds \
        - inner(grad(v), outer(u,N))*ds \
        + gamma/h*inner(v,u)*ds
    a12 = div(v)*p*dx
    a21 = div(u)*q*dx
    L1  =  inner(v,f)*dx + gamma/h*inner(u0,v)*ds - inner(grad(v),outer(u0,N))*ds
    a = a11-a12-a21
    i = p*q*dx


    tic()
    AA, bb = assemble_system(a, L1, bcs)
    As = AA.sparray()[0:-1,0:-1]
    As.eliminate_zeros()
    A = PETSc.Mat().createAIJ(size=As.shape,csr=(As.indptr, As.indices, As.data))
    print toc()
    b = bb.array()[0:-1]
    zeros = 0*b
    del bb
    bb = IO.arrayToVec(b)
    x = IO.arrayToVec(zeros)

    PP, Pb = assemble_system(a11+i,L1,bcs)
    Ps = PP.sparray()[0:-1,0:-1]
    # Ps.eliminate_zeros()
    P = PETSc.Mat().createAIJ(size=Ps.shape,csr=(Ps.indptr, Ps.indices, Ps.data))

    if (SavePrecond == 'yes'):
        Wstring = str(int(Wdim[xx-1][0]-1))
        nameA ="".join(['eigenvalues/A',Wstring,".mat"])
        scipy.io.savemat(nameA, mdict={'A': As},oned_as='row')
        nameP ="".join(['eigenvalues/P',Wstring,".mat"])
        scipy.io.savemat(nameP, mdict={'P': Ps},oned_as='row')

    del AA, As, PP, Ps

    if (EigenProblem == 'yes'):
        eigenvalues = np.zeros((Wdim[xx-1][0]-1,1))
        xr, tmp = A.getVecs()
        xi, tmp = A.getVecs()
        E = SLEPc.EPS().create()
        E.setOperators(A,P)
        E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        # E.setBalance()
        E.setDimensions(Wdim[xx-1][0])
        E.setTolerances(tol=1.e-15, max_it=500000)
        E.solve()
        Print("")
        its = E.getIterationNumber()
        Print("Number of iterations of the method: %i" % its)
        sol_type = E.getType()
        Print("Solution method: %s" % sol_type)
        nev, ncv, mpd = E.getDimensions()
        Print("Number of requested eigenvalues: %i" % nev)
        tol, maxit = E.getTolerances()
        Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
        nconv = E.getConverged()
        Print("Number of converged eigenpairs: %d" % nconv)
        if nconv > 0:
            Print("")
            Print("        k          ||Ax-kx||/||kx|| ")
            Print("----------------- ------------------")
            for i in range(nconv):
                k = E.getEigenpair(i, xr, xi)
                eigenvalues[i-1] = k.real
                error = E.computeRelativeError(i)
                if k.imag != 0.0:
                  Print(" %12f" % (k.real))
                else:
                  Print(" %12f      " % (k.real))
            Print("")
            Wstring = str(int(Wdim[xx-1][0]-1))
            name ="".join(['eigenvalues/e',Wstring,".mat"])
            scipy.io.savemat(name, mdict={'e': eigenvalues},oned_as='row')

    if (Solving == 'Direct'):
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)

        ksp.setFromOptions()
        print 'Solving with:', ksp.getType()

        tic()
        ksp.solve(bb, x)
        SolTime[xx-1] = toc()
        print "time to solve: ",SolTime[xx-1]
        A.destroy()


    if (Solving == 'Iterative'):

        ksp = PETSc.KSP().create()
        pc = PETSc.PC().create()
        ksp.setFromOptions()
        # ksp.create(PETSc.COMM_WORLD)
        # use conjugate gradients
        ksp.setTolerances(1e-10)
        ksp.setType('minres')
        pc = ksp.getPC()
        pc.setOperators(P)
        pc.getType()
        # and next solve
        ksp.setOperators(A,P)
        tic()
        ksp.solve(bb, x)
        SolTime[xx-1] = toc()
        print "time to solve: ",SolTime[xx-1]
        iterations[xx-1] =  ksp.its
        print "iterations = ", iterations[xx-1]

    if (Solving == 'Iterative' or Solving == 'Direct'):
        if case == 1:
            ue = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
            pe = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)+5")
        elif case == 2:
            ue = Expression(("sin(pi*x[1])","sin(pi*x[0])"))
            pe = Expression("sin(x[1]*x[0])")
        elif case == 3:
            ue = Expression(("cos(2*pi*x[1])*sin(2*pi*x[0]) ","-cos(2*pi*x[0])*sin(2*pi*x[1]) "))
            pe = Expression("sin(2*pi*x[0])*sin(2*pi*x[1]) ")

        u = interpolate(ue,V)
        p = interpolate(pe,Q)

        Nv  = u.vector().array().shape

        X = IO.vecToArray(x)
        x = X[0:Vdim[xx-1][0]]
        # x = x_epetra[0:Nv[0]]
        ua = Function(V)
        ua.vector()[:] = x
        udiv[xx-1] = assemble(div(ua)*dx)
        pp = X[Nv[0]:]
        n = pp.shape
        pp = np.insert(pp,n,0)
        pa = Function(Q)
        pa.vector()[:] = pp

        pend = assemble(pa*dx)

        ones = Function(Q)
        ones.vector()[:]=(0*pp+1)
        pp = Function(Q)
        pp.vector()[:] = pa.vector().array()- assemble(pa*dx)/assemble(ones*dx)

        pInterp = interpolate(pe,Q)
        pe = Function(Q)
        pe.vector()[:] = pInterp.vector().array()
        const = - assemble(pe*dx)/assemble(ones*dx)
        pe.vector()[:] = pe.vector()[:]+const

        errL2u[xx-1] = errornorm(ue,ua,norm_type="L2", degree_rise=6,mesh=mesh)
        errL2p[xx-1] = errornorm(pe,pp,norm_type="L2", degree_rise=6,mesh=mesh)

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
    if (Solving == 'Iterative'):
        tableTitles = ["Total DoF","V DoF","Q DoF","# iters","Soln Time","V-L2","V-order","||div u_h||","P-L2","P-order"]
        tableValues = np.concatenate((Wdim,Vdim,Qdim,iterations,SolTime,errL2u,l2uorder,udiv,errL2p,l2porder),axis=1)
    elif (Solving == 'Direct'):
        tableTitles = ["Total DoF","V DoF","Q DoF","Soln Time","V-L2","V-order","||div u_h||","P-L2","P-order"]
        tableValues = np.concatenate((Wdim,Vdim,Qdim,SolTime,errL2u,l2uorder,udiv,errL2p,l2porder),axis=1)


    df = pd.DataFrame(tableValues, columns = tableTitles)
    pd.set_printoptions(precision=3)
    print df
    print "\n\n"
    print "Velocity Elements rate of convergence ", np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1])))
    print "Pressure Elements rate of convergence ", np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1])))
    print df.to_latex()


if (SavePrecond == 'yes'):
    scipy.io.savemat('eigenvalues/Wdim.mat', {'Wdim':Wdim-1},)


if (ShowResultPlots == 'yes'):
    plot(ua)
    plot(interpolate(ue,V))

    plot(pp)
    plot(interpolate(pe,Q))

    interactive()

