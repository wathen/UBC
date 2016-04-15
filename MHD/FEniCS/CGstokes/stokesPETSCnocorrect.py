#!/usr/bin/python
import petsc4py
# import slepc4`py
import sys

petsc4py.init(sys.argv)
# slepc4py.init(sys.argv)

from petsc4py import PETSc
# from slepc4py import SLEPc
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

def AdaptiveRefinement(mesh,g,Refine_tol):
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    g0 = sorted(g, reverse=True)[int(len(g)*Refine_tol)]
    for c in cells(mesh):
        cell_markers[c] = g[c.index()] > g0

        # Refine mesh
    mesh = refine(mesh, cell_markers)
    return mesh

m = 7
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

dim = 3
Solving = 'Direct'
ShowResultPlots = 'No'
ShowErrorPlots = 'no'
EigenProblem = 'no'
SavePrecond = 'no'
UseExactSchur = 'no'
CheckMu = 'no'
Refine = 'no'
case = 1
parameters['linear_algebra_backend'] = 'uBLAS'
MU[0]= 1e0
if Refine == 'yes':
    Refine_tol = .4
    if dim == 2:
        mesh = RectangleMesh(-1, -1, 1, 1, 16, 16,'crossed')
for xx in xrange(1,m):
    print xx
    nn = 2**(xx)
    xx = float(xx)
    if (CheckMu == 'yes'):
        if (xx != 1):
            MU[xx-1] = MU[xx-2]/10
    else:
        if (xx != 1):
            MU[xx-1] = MU[xx-2]
    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn
    if Refine == 'no':
        if dim == 2:
            mesh = RectangleMesh(-1, -1, 1, 1, nn, nn,'crossed')
        else:
            mesh = BoxMesh(-1, -1, -1, 1, 1, 1, nn/2, nn/2,nn/2)
    # mesh = RectangleMesh(-1, -1, 1, 1, nn, nn,'crossed')


    # origin1 = Point(0.5, 0.5)
    # origin2 = Point(0.5,-0.5)
    # origin3 = Point(-0.5, 0.5)
    # origin4 = Point(-0.5, -0.5)


    # for cell in cells(mesh):
    #   p = cell.midpoint()
    #   # print p
    #   if p.distance(origin1) < 0.15:
    #       cell_markers[cell] = True

    #   if p.distance(origin2) < 0.25:
    #       cell_markers[cell] = True

    #   if p.distance(origin3) < 0.35:
    #       cell_markers[cell] = True

    #   if p.distance(origin4) < 0.45:
    #       cell_markers[cell] = True
    if dim == 2:
        cell_markers = CellFunction("bool", mesh)
        cell_markers.set_all(False)
        mesh = refine(mesh, cell_markers)
        cell_markers = CellFunction("bool", mesh)
        cell_markers.set_all(False)
        origin1 = Point(-1, 0)
        origin2 = Point(-.6,-.3)
        origin3 = Point(-.2, .2)
        origin4 = Point(.2,.2)
        origin5 = Point(.6,.3)
        origin6 = Point(1,0)


        for cell in cells(mesh):
            p = cell.midpoint()
            if p.distance(origin1) < (0.3):
                cell_markers[cell] = True
            if p.distance(origin2) < (0.4):
                cell_markers[cell] = True

            if p.distance(origin3) < (0.35):
                cell_markers[cell] = True

            if p.distance(origin4) < (0.3):
                cell_markers[cell] = True

            if p.distance(origin5) < (0.3):
                cell_markers[cell] = True

            if p.distance(origin6) < (0.3):
                cell_markers[cell] = True


        mesh = refine(mesh, cell_markers)
    else:
        cell_markers = CellFunction("bool", mesh)
        cell_markers.set_all(False)
        mesh = refine(mesh, cell_markers)
        cell_markers = CellFunction("bool", mesh)
        cell_markers.set_all(False)
        origin1 = Point(-1, 0,.5)
        origin2 = Point(-.6,-.3,.5)
        origin3 = Point(-.2, .2,.5)
        origin4 = Point(.2,.2,.5)
        origin5 = Point(.6,.3,.5)
        origin6 = Point(1,0,.5)


        for cell in cells(mesh):
            p = cell.midpoint()
            if p.distance(origin1) < (0.3):
                cell_markers[cell] = True
            if p.distance(origin2) < (0.4):
                cell_markers[cell] = True

            if p.distance(origin3) < (0.35):
                cell_markers[cell] = True

            if p.distance(origin4) < (0.3):
                cell_markers[cell] = True

            if p.distance(origin5) < (0.3):
                cell_markers[cell] = True

            if p.distance(origin6) < (0.3):
                cell_markers[cell] = True


        mesh = refine(mesh, cell_markers)
    # plot(mesh)
    # interactive()


    parameters['reorder_dofs_serial'] = False
    V = VectorFunctionSpace(mesh, "CG", 2)
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
        u0 = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)","5*pow(x[0],4)-5*pow(x[1],4)"))
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
        f = Expression(("120*x[0]*x[1]*(1-mu)","60*(pow(x[0],2)-pow(x[1],2))*(1-mu)","0"), mu = MU[xx-1][0])
    elif case == 2:
        f = Expression(("pi*pi*sin(pi*x[1])+x[1]*cos(x[1]*x[0])","pi*pi*sin(pi*x[0])+x[0]*cos(x[1]*x[0])"))
    elif case == 3:
        f = -Expression(("8*pi*pi*cos(2*pi*x[1])*sin(2*pi*x[0]) + 2*pi*cos(2*pi*x[0])*sin(2*pi*x[1])","2*pi*cos(2*pi*x[0])*sin(2*pi*x[1]) - 8*pi*pi*cos(2*pi*x[0])*sin(2*pi*x[1])"))

    mu = MU[xx-1][0]
    a11 = mu*inner(grad(v), grad(u))*dx
    a12 = -div(v)*p*dx
    a21 = -div(u)*q*dx
    L1  = inner(v, f)*dx
    a = a11+a12+a21
    i = (1/mu)*p*q*dx


    tic()
    AA, bb = assemble_system(a, L1, bcs)
    As = AA.sparray()

    A = PETSc.Mat().createAIJ(size=As.shape,csr=(As.indptr, As.indices, As.data))
    print toc()
    b = bb.array()
    zeros = 0*b
    del bb
    bb = IO.arrayToVec(b)
    x = IO.arrayToVec(zeros)

    PP, Pb = assemble_system(a11+i,L1,bcs)
    Ps = PP.sparray()
    P = PETSc.Mat().createAIJ(size=Ps.shape,csr=(Ps.indptr, Ps.indices, Ps.data))

    if (SavePrecond == 'yes'):
        PP, Pb = assemble_system(a11+i,L1,bcs)
        Ps = PP.sparray()[0:-1,0:-1]
        Wstring = str(int(Wdim[xx-1][0]-1))
        nameA ="".join(['eigenvalues/A',Wstring,".mat"])
        scipy.io.savemat(nameA, mdict={'A': As},oned_as='row')
        nameP ="".join(['eigenvalues/P',Wstring,".mat"])
        scipy.io.savemat(nameP, mdict={'P': Ps},oned_as='row')
        del PP,Pb,Ps,AA,As


    if (EigenProblem == 'yes'):
        PP, Pb = assemble_system(a11+i,L1,bcs)
        Ps = PP.sparray()[0:-1,0:-1]
        P = PETSc.Mat().createAIJ(size=Ps.shape,csr=(Ps.indptr, Ps.indices, Ps.data))
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

        del PP,Pb,Ps,AA,As

    if (Solving == 'Direct'):
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)

        ksp.setFromOptions()
        print 'Solving with:', ksp.getType()

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
            ue = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
            pe = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)+5")
        elif case == 2:
            ue = Expression(("sin(pi*x[1])","sin(pi*x[0])"))
            pe = Expression("sin(x[1]*x[0])")
        elif case == 3:
            ue = Expression(("cos(2*pi*x[1])*sin(2*pi*x[0]) ","-cos(2*pi*x[0])*sin(2*pi*x[1]) "))
            pe = Expression("sin(2*pi*x[0])*sin(2*pi*x[1]) ")

        # u = interpolate(ue,V)
        # p = interpolate(pe,Q)

        # Nv  = u.vector().array().shape

        # X = IO.vecToArray(x)
        # x = X[0:Vdim[xx-1][0]]
        # # x = x_epetra[0:Nv[0]]
        # ua = Function(V)
        # ua.vector()[:] = x
        # udiv[xx-1] = assemble(div(ua)*dx)
        # pp = X[Nv[0]:]
        # pa = Function(Q)

        # pa.vector()[:] = pp

        # pend = assemble(pa*dx)

        # ones = Function(Q)
        # ones.vector()[:]=(0*pp+1)
        # pp = Function(Q)
        # pp.vector()[:] = pa.vector().array()- assemble(pa*dx)/assemble(ones*dx)

        # pInterp = interpolate(pe,Q)
        # pe = Function(Q)
        # pe.vector()[:] = pInterp.vector().array()
        # const = - assemble(pe*dx)/assemble(ones*dx)
        # pe.vector()[:] = pe.vector()[:]+const

        # errL2u[xx-1] = errornorm(ue,ua,norm_type="L2", degree_rise=4,mesh=mesh)
        # errL2p[xx-1] = errornorm(pe,pp,norm_type="L2", degree_rise=4,mesh=mesh)

        # if Refine == 'yes':
        #     PC = FunctionSpace(mesh,"DG", 0)
        #     c  = TestFunction(PC)
        #     g  = assemble(inner(grad(ua), grad(ua))*c*dx)
        #     g  = g.array()
        #     mesh = AdaptiveRefinement(mesh,g,Refine_tol)
        #     # Plot mesh
        #     # plot(u,wireframe=True,scalarbar=False)
        #     plot(mesh)

        # if xx == 1:
        #     l2uorder[xx-1] = 0
        # else:
        #     l2uorder[xx-1] =  np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1]))
        #     l2porder[xx-1] =  np.abs(np.log2(errL2p[xx-2]/errL2p[xx-1]))

        # print errL2u[xx-1]
        # print errL2p[xx-1]


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
        tableTitles = ["Total DoF","V DoF","Q DoF","# iters","Soln Time","V-L2","V-order","||div u_h||","P-L2","P-order"]
        tableValues = np.concatenate((Wdim,Vdim,Qdim,iterations,SolTime,errL2u,l2uorder,udiv,errL2p,l2porder),axis=1)
    elif (Solving == 'Direct' and CheckMu == 'no'):
        tableTitles = ["Total DoF","V DoF","Q DoF","Soln Time","V-L2","V-order","||div u_h||","P-L2","P-order"]
        tableValues = np.concatenate((Wdim,Vdim,Qdim,SolTime,errL2u,l2uorder,udiv,errL2p,l2porder),axis=1)
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

    plot(pp)
    plot(interpolate(pe,Q))



interactive()





