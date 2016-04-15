
#!/usr/bin/python
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
Print = PETSc.Sys.Print
# from MatrixOperations import *
from dolfin import *
from PETScIO import arrayToVec
import numpy as np
import os
import scipy.io
import ExactSol
import CheckPetsc4py as CP
import memory_profiler
import MHDmult
import BiLinear
# @profile
def foo():
    m = 5
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
        nn = 2**(xx+0)
        # Create mesh and define function space
        nn = int(nn)
        NN[xx-1] = nn


        # mesh = UnitSquareMesh(nn,nn)
        mesh = UnitCubeMesh(nn,nn,nn)
        parameters['reorder_dofs_serial'] = False
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh,"CG",1)
        C = FunctionSpace(mesh,"N1curl",1)
        S = FunctionSpace(mesh,"CG",1)
        W = MixedFunctionSpace([V,Q,C,S])
        def boundary(x, on_boundary):
            return on_boundary
        print "               DOFs              ", W.dim()
        u0, p0,b0, r0, Laplacian, Advection, gradPres,CurlCurl, gradR, NS_Couple, M_Couple = ExactSol.MHD3D(4,1,mesh)
        dim = Laplacian.shape()[0]

        n = FacetNormal(mesh)
        bcu = DirichletBC(V,u0, boundary)
        bcb = DirichletBC(C,b0, boundary)
        bcr = DirichletBC(S,r0, boundary)

        u_k = Function(V)
        u_k.vector()[:] = np.random.rand(V.dim())
        bcu.apply(u_k.vector())
        p_k = Function(Q)
        p_k.vector()[:] = np.random.rand(Q.dim())
        b_k = Function(C)
        b_k.vector()[:] = np.random.rand(C.dim())
        bcb.apply(b_k.vector())
        r_k = Function(S)
        r_k.vector()[:] = np.random.rand(S.dim())
        bcr.apply(r_k.vector())

        B = np.concatenate((u_k.vector().array(),p_k.vector().array(),b_k.vector().array(),r_k.vector().array()), axis=0)
        x = arrayToVec(B)


        (u, p, b, r) = TrialFunctions(W)
        (v, q, c, s) = TestFunctions(W)


        m11 = inner(curl(b),curl(c))*dx
        m22 = inner(r,s)*dx
        m21 = inner(c,grad(r))*dx
        m12 = inner(b,grad(s))*dx
        # Lmaxwell  = inner(c, F_M)*dx

        a11 = inner(grad(v), grad(u))*dx(mesh)+ inner((grad(u)*u_k),v)*dx(mesh) +(1./2)*div(u_k)*inner(u,v)*dx(mesh) - (1./2)*inner(u_k,n)*inner(u,v)*ds(mesh)
        a12 = -div(v)*p*dx
        a21 = -div(u)*q*dx
        # Lns  = inner(v, F_NS)*dx

        if dim == 2:
            CoupleT = (v[0]*b_k[1]-v[1]*b_k[0])*curl(b)*dx
            Couple = -(u[0]*b_k[1]-u[1]*b_k[0])*curl(c)*dx
        elif dim == 3:
            CoupleT = inner(cross(v,b_k),curl(b))*dx
            Couple = -inner(cross(u,b_k),curl(c))*dx


        a = m11+m12+m21+a11+a12+a21+Couple+CoupleT

        bcu = DirichletBC(W.sub(0),u0, boundary)
        bcb = DirichletBC(W.sub(2),b0, boundary)
        bcr = DirichletBC(W.sub(3),r0, boundary)
        bcs = [bcu,bcb,bcr]


        tic()

        params = [1,1,1]
        aVec, L_M, L_NS, Bt, CoupleT = BiLinear.MHDmatvec(mesh, W, Laplacian, Laplacian,u_k,b_k,u_k,b_k,p_k,r_k, params,"Full","CG", SaddlePoint = "No")
        PrecondTmult = {'Bt': Bt, 'Ct':CoupleT, 'BC': DirichletBC(V,u0, boundary)}
        FS = {'velocity': V, 'pressure': Q, 'magnetic': C, 'multiplier': S}
        P = PETSc.Mat().createPython([W.dim(), W.dim()])
        P.setType('python')
        aa = MHDmult.MatVec(FS, aVec, bcs)
        P.setPythonContext(aa)
        P.setUp()
        for i in range(50):
            v = x.duplicate()
            P.mult(x,v)
            # print A.array()
        print '                                      ', toc()


        tic()
        AA = assemble(a)
        for bc in bcs:
            bc.apply(AA)
        # bc.apply(AA)
        A = CP.Assemble(AA)
        # bb.set(1)
        for i in range(50):
            # A = CP.Assemble(A)
            for bc in bcs:
                bc.apply(AA)
            u = x.duplicate()
            A.mult(x,u)


        print '                                      ', toc()

        # print b_k.vector().array()
        # a = inner(grad(v), grad(b_k))*dx
        print np.linalg.norm(u.array- v.array, ord=np.inf)
        # print u.array, P.array

foo()