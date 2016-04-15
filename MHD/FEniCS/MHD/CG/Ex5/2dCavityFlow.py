#!/usr/bin/python
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
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
import CavityDomain2d as CD

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

m = 2
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
ShowResultPlots = 'yes'
ShowErrorPlots = 'no'
EigenProblem = 'no'
SavePrecond = 'no'
UseExactSchur = 'no'
CheckMu = 'no'
parameters['linear_algebra_backend'] = 'uBLAS'
MU[0]= 1e0
for xx in xrange(1,m):
    print xx
    nn = 2**(xx+2)


    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn

    mesh, boundaries = CD.CavityMesh2d(nn)

    parameters['reorder_dofs_serial'] = False
    Velocity = VectorFunctionSpace(mesh, "CG", 2)
    Pressure = FunctionSpace(mesh, "CG", 1)
    Electric = FunctionSpace(mesh, "N2curl", 1)
    Lagrange = FunctionSpace(mesh, "CG", 1)
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

    u01 =Expression(("0","0"),cell=triangle)
    u02 =Expression(("1","0"),cell=triangle)
    b0 = Expression(("1","0"),cell=triangle)
    r0 = Expression(("0"),cell=triangle)

    # u0 = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
    # p0 = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")
    # b0 = Expression(("0","0"))
    # r0 = Expression("0")


    bcu1 = DirichletBC(W.sub(0),u01, boundaries,1)
    bcu2 = DirichletBC(W.sub(0),u02, boundaries,2)
    # bcp = DirichletBC(W.sub(1),p0, boundary)
    bcb = DirichletBC(W.sub(2),b0, boundary)
    bcr = DirichletBC(W.sub(3),r0, boundary)
    bc = [bcu1,bcu2,bcb,bcr]

    (u, p, b, r) = TrialFunctions(W)
    (v, q, c,s ) = TestFunctions(W)
    K = 1e5
    Mu_m = 1e5
    MU = 1e-2
    fns = Expression(("0","0"),mu = MU, k = K)
    fm = Expression(("0","0"),k = K,mu_m = Mu_m)

    "'Maxwell Setup'"
    a11 = K*Mu_m*inner(curl(c),curl(b))*dx
    a12 = inner(c,grad(r))*dx
    a21 = inner(b,grad(s))*dx
    Lmaxwell  = inner(c, fm)*dx
    maxwell = a11+a12+a21


    "'NS Setup'"
    u_k = Function(Velocity)
    u_k.vector()[:] = u_k.vector()[:]*0
    n = FacetNormal(mesh)
    a11 = MU*inner(grad(v), grad(u))*dx+inner((grad(u)*u_k),v)*dx+(1/2)*div(u_k)*inner(u,v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds
    a12 = -div(v)*p*dx
    a21 = -div(u)*q*dx
    Lns  = inner(v, fns)*dx
    ns = a11+a12+a21


    "'Coupling term Setup'"
    b_k = Function(Electric)
    b_k.vector()[:] = b_k.vector()[:]*0
    CoupleTerm = K*inner(v[0]*b_k[1]-v[1]*b_k[0],curl(b))*dx - K*inner(u[0]*b_k[1]-u[1]*b_k[0],curl(c))*dx


    """Maxwell preconditioner"""
    M11 = K*Mu_m*inner(curl(c),curl(b))*dx +inner(c,b)*dx
    M22 = inner(grad(s),grad(r))*dx


    """NS preconditioner"""
    NS11 = MU*inner(grad(v), grad(u))*dx+inner((grad(u)*u_k),v)*dx+(1/2)*div(u_k)*inner(u,v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds
    NS22 = inner(p, q)*dx



    if (Solving == 'Direct'):

        eps = 1.0           # error measure ||u-u_k||
        # epsb = 1.0
        tol = 1.0E-5       # tolerance
        iter = 0            # iteration counter
        maxiter = 20        # max no of iterations allowed
        SolutionTime = 0
        while eps > tol and iter < maxiter:
            iter += 1
            uu = Function(W)
            tic()
            AA, bb = assemble_system(maxwell+ns+CoupleTerm, Lmaxwell + Lns, bc)
            StoreMatrix(As,"A")
            VelPres = Velocitydim[xx-1][0] +Pressuredim[xx-1][0]
            Adelete = remove_ij(As,VelPres-1,VelPres-1)
            A = PETSc.Mat().createAIJ(size=Adelete.shape,csr=(Adelete.indptr, Adelete.indices, Adelete.data))
            print toc()

            PP,Pb = assemble_system(M11+M22+NS11+NS22,Lmaxwell + Lns, bc)
            Ps = PP.sparray()

            StoreMatrix(Ps,"A")
            VelPres = Velocitydim[xx-1][0] +Pressuredim[xx-1][0]
            Pdelete = remove_ij(Ps,VelPres-1,VelPres-1)
            P = PETSc.Mat().createAIJ(size=Pdelete.shape,csr=(Pdelete.indptr, Pdelete.indices, Pdelete.data))

            b = np.delete(bb,VelPres-1,0)
            zeros = 0*b
            bb = IO.arrayToVec(b)
            x = IO.arrayToVec(zeros)
            # ksp = PETSc.KSP().create()
            # pc = PETSc.PC().create()
            # ksp.setOperators(A)


            ksp = PETSc.KSP().create()
            pc = PETSc.PC().create()
            ksp.setFromOptions()
            ksp.setTolerances(1e-16)
            print 'Solving with:', ksp.setType('preonly')
            # ksp.setPCSide(2)

            # pc = ksp.getPC()
            # pc.setOperators(A)
            # pc.getType()
            ksp.setOperators(A)
            tic()
            ksp.solve(bb, x)
            time= toc()
            print "time to solve: ",SolTime[xx-1]
            iterations[xx-1] =  ksp.its
            print "iterations = ", iterations[xx-1]
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






        xb = X[VelPres-1:VelPres+Electricdim[xx-1][0]-1]
        ba = Function(Electric)
        ba.vector()[:] = xb

        xr = X[VelPres+Electricdim[xx-1][0]-1:]
        ra = Function(Lagrange)
        ra.vector()[:] = xr



if (ShowResultPlots == 'yes'):

    # plot(ua)

    # plot(pp)

    plot(ba)

    # plot(ra)

interactive()








