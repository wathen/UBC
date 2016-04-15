
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
# @profile
def foo():
    m = 10
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


        mesh = UnitSquareMesh(nn,nn)
        parameters['reorder_dofs_serial'] = False
        V = VectorFunctionSpace(mesh, "CG", 2)
        def boundary(x, on_boundary):
            return on_boundary
        print " DOFs              ", V.dim()

        u0, p0,b0, r0, Laplacian, Advection, gradPres,CurlCurl, gradR, NS_Couple, M_Couple = ExactSol.MHD2D(4,1,mesh)


        bc = DirichletBC(V,u0, boundary)

        u = TrialFunction(V)
        v = TestFunction(V)

        N = FacetNormal(mesh)
        h = CellSize(mesh)
        h_avg =avg(h)
        alpha = 10.0
        gamma =10.0
        n = FacetNormal(mesh)
        h = CellSize(mesh)
        h_avg =avg(h)
        d = 0
        a = inner(grad(v), grad(u))*dx
        L = inner(Laplacian, v)*dx
        # A = assemble(a)

        b_k = Function(V)
        # print bb.array
        b_k.vector()[:] = np.random.rand(V.dim())
        bc.apply(b_k.vector())
        bb = arrayToVec(b_k.vector().array())


        tic()
        AA = assemble(a)
        bc.apply(AA)
        A = CP.Assemble(AA)
        # bb.set(1)
        for i in range(50):
            # A = CP.Assemble(A)
            u = A*bb
        print '                                      ', toc()

        # print b_k.vector().array()
        a = inner(grad(v), grad(b_k))*dx
        tic()
        for i in range(50):
            A = assemble(a)
            # print A.array()
            bc.apply(A)
            A = arrayToVec(A.array())
            # print A.array()
        print '                                      ', toc()
        print np.linalg.norm(u.array- A.array)
        # print u.array, A.array

        # #\
        # - inner(avg(grad(v)), outer(u('+'),n('+'))+outer(u('-'),n('-')))*dS \
        # - inner(outer(v('+'),n('+'))+outer(v('-'),n('-')), avg(grad(u)))*dS \
        # + alpha/h_avg*inner(outer(v('+'),n('+'))+outer(v('-'),n('-')),outer(u('+'),n('+'))+outer(u('-'),n('-')))*dS \
        # + gamma/h*inner(v,u)*ds
        # p = inner(grad(v), grad(u))*dx \
        # - inner(avg(grad(v)), outer(u('+'),n('+'))+outer(u('-'),n('-')))*dS \
        # - inner(outer(v('+'),n('+'))+outer(v('-'),n('-')), avg(grad(u)))*dS \
        # + alpha/h_avg*inner(outer(v('+'),n('+'))+outer(v('-'),n('-')),outer(u('+'),n('+'))+outer(u('-'),n('-')))*dS \
        # + gamma/h*inner(v,u)*ds

        # L = inner(Laplacian, v)*dx
        # AA, bb = assemble_system(a,L,bc)

        # A,b = CP.Assemble(AA,bb)
        # PP, pb = assemble_system(p,L,bc)

        # P,bb = CP.Assemble(AA,bb)
        # u = b.duplicate()

        # ksp = PETSc.KSP()
        # ksp.create(comm=PETSc.COMM_WORLD)
        # pc = ksp.getPC()
        # ksp.setType('cg')
        # pc.setType('hypre')
        # OptDB = PETSc.Options()
        # OptDB['pc_hypre_boomeramg_cycle_type']  = "W"
        # # OptDB['pc_hypre_boomeramg_strong_threshold'] = 0.7
        # # OptDB['pc_hypre_boomeramg_grid_sweeps_all'] = 3
        # # OptDB['pc_hypre_boomeramg_relax_type_all'] = 'CG'

        # # direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1
        # ksp.setFromOptions()
        # scale = b.norm()
        # b = b/scale
        # ksp.setOperators(A,P)
        # del A
        # ksp.solve(b,u)
        # # Mits +=dodim
        # u = u*scale
        # print '                             ', ksp.its
foo()