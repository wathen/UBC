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
import PETScIO as IO
import StepDomain as SD

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
    nn = 2**(xx+6)


    # Create mesh and define function space
    nn = int(nn)

    h = 1.0/nn


    mesh, boundaries = SD.StepMesh(h)



    parameters['reorder_dofs_serial'] = False
    Velocity = VectorFunctionSpace(mesh, "CG", 2)
    Pressure = FunctionSpace(mesh, "CG", 1)
    Electric = FunctionSpace(mesh, "N1curl", 1)
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

    inflow =Expression(("-25.6*x[1]*(x[1]-0.125)","0"),cell=triangle)
    noflow = Expression(("0","0"),cell=triangle)
    b0 = Expression(("0","1"),cell=triangle)
    r0 = Expression(("0"),cell=triangle)



    bcout = DirichletBC(W.sub(0),noflow, boundaries,3)
    bcin = DirichletBC(W.sub(0),inflow, boundaries,2)
    bcb = DirichletBC(W.sub(2),b0, boundary)
    bcr = DirichletBC(W.sub(3),r0, boundary)
    bc = [bcin,bcout,bcb,bcr]

    (u, p, b, r) = TrialFunctions(W)
    (v, q, c,s ) = TestFunctions(W)
    k = 1e-2
    kappa = 1e5
    Mu_m = 2.5e4
    MU = 1e-2

    fns = Expression(("0","0"))
    fm = Expression(("0","0"))

    "'Maxwell Setup'"
    a11 = kappa*Mu_m*inner(curl(c),curl(b))*dx-k*k*inner(b,c)*dx
    a12 = inner(c,grad(r))*dx
    a21 = inner(b,grad(s))*dx
    Lmaxwell  = inner(c, fm)*dx
    maxwell = a11+a12+a21


    "'NS Setup'"
    u_k = Function(Velocity)
    u_k.vector()[:] = u_k.vector()[:]*0
    n =FacetNormal(mesh)
    # g = -grad(inner(u0,n))+p0*n
    g = 0
    ds = Measure("ds")[boundaries]
    n = FacetNormal(mesh)
    a11 = MU*inner(grad(v), grad(u))*dx+inner((grad(u)*u_k),v)*dx+(1/2)*div(u_k)*inner(u,v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds(0)
    a12 = -div(v)*p*dx
    a21 = -div(u)*q*dx
    Lns  = inner(v, fns)*dx
    ns = a11+a12+a21


    "'Coupling term Setup'"
    b_k = Function(Electric)
    b_k.vector()[:] = b_k.vector()[:]*0+1
    CoupleTerm = kappa*inner(v[0]*b_k[1]-v[1]*b_k[0],curl(b))*dx - kappa*inner(u[0]*b_k[1]-u[1],b_k[0]*curl(c))*dx



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
                As = AA.sparray()
                # StoreMatrix(As,"A")
                VelPres = Velocitydim[xx-1][0] +Pressuredim[xx-1][0]
                A = PETSc.Mat().createAIJ(size=As.shape,csr=(As.indptr, As.indices, As.data))
                print toc()
                b = bb.array()
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
                bb1 = X[VelPres:VelPres+Electricdim[xx-1][0]]

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
        # scipy.io.savemat('eigenvalues/Wdim.mat', {'Approx'W:dim-1},oned_as = 'row')
        xu = X[0:Velocitydim[xx-1][0]]
        ua = Function(Velocity)
        ua.vector()[:] = xu

        pp = X[Velocitydim[xx-1][0]:VelPres]
        # xp[-1] = 0
        # pa = Function(Pressure)
        # pa.vector()[:] = xp

        n = pp.shape
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





        xb = X[VelPres:VelPres+Electricdim[xx-1][0]]
        ba = Function(Electric)
        ba.vector()[:] = xb

        xr = X[VelPres+Electricdim[xx-1][0]:]
        ra = Function(Lagrange)
        ra.vector()[:] = xr

if (ShowResultPlots == 'yes'):

    plot(ua)

    plot(pp)

    plot(ba)

    plot(ra)

interactive()



ufile_pvd = File("Results/velocity.pvd")
ufile_pvd << ua




