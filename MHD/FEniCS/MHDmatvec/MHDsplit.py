
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


        u  = TrialFunction(V)
        b = TrialFunction(C)
        p = TrialFunction(Q)
        r = TrialFunction(S)
        v = TestFunction(V)
        c = TestFunction(C)
        q = TestFunction(Q)
        s = TestFunction(S)

        mm11 = inner(curl(b_k),curl(c))*dx
        mm21 = inner(c,grad(r_k))*dx
        mm12 = inner(b_k,grad(s))*dx

        aa11 = inner(grad(v), grad(u_k))*dx(mesh) + inner((grad(u_k)*u_k),v)*dx(mesh) +(1./2)*div(u_k)*inner(u_k,v)*dx(mesh) - (1./2)*inner(u_k,n)*inner(u_k,v)*ds(mesh)
        aa12 = -div(v)*p_k*dx
        aa21 = -div(u_k)*q*dx

        if dim == 2:
            CCoupleT = (v[0]*b_k[1]-v[1]*b_k[0])*curl(b_k)*dx
            CCouple = -(u_k[0]*b_k[1]-u_k[1]*b_k[0])*curl(c)*dx
        elif dim == 3:
            CCoupleT = inner(cross(v,b_k),curl(b_k))*dx
            CCouple = -inner(cross(u_k,b_k),curl(c))*dx

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

        aVec = {'velocity': [aa11, aa12, CCoupleT], 'pressure': [aa21], 'magnetic': [CCouple, mm11, mm21], 'multiplier': [mm12]}
        bcs = {'velocity': bcu, 'magnetic': bcb, 'multiplier': bcr}
        tic()
        a
        P = PETSc.Mat().createPython([W.dim(), W.dim()])
        P.setType('python')
        aa = MHDmult.SplitMatVec(W, aVec, bcs)
        P.setPythonContext(aa)
        P.setUp()
        for i in range(50):
            # U = assemble(aa11)+assemble(aa12)+assemble(CCoupleT)
            # bcu.apply(U)
            # P = assemble(aa21)
            # B = assemble(CCouple)+assemble(mm11)+assemble(mm21)
            # bcb.apply(B)
            # R = assemble(mm12)
            # bcr.apply(R)

            # B = np.concatenate((U.array(),P.array(),B.array(),R.array()), axis=0)
            # P = arrayToVec(B)
            # print A.array()
            v = x.duplicate()
            P.mult(x,v)

        print '                                      ', toc()

        bcu = DirichletBC(W.sub(0),u0, boundary)
        bcb = DirichletBC(W.sub(2),b0, boundary)
        bcr = DirichletBC(W.sub(3),r0, boundary)
        bcs = [bcu,bcb,bcr]

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