from dolfin import *

#!/usr/bin/python
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
Print = PETSc.Sys.Print

import numpy as np
import matplotlib.pylab as plt
import PETScIO as IO
import scipy
import scipy.io
import CheckPetsc4py as CP

def Stokes(V,Q,BC,f,mu,boundaries):

    parameters = CP.ParameterSetup()

    W = V*Q

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    def boundary(x, on_boundary):
        return on_boundary


    bcu1 = DirichletBC(W.sub(0),BC[0], boundaries,2)
    bcu2 = DirichletBC(W.sub(0),BC[1], boundaries,1)
    bcs = [bcu1,bcu2]
    u_k = Function(V)
    a11 = mu*inner(grad(v), grad(u))*dx
    a12 = div(v)*p*dx
    a21 = div(u)*q*dx
    L1  =  inner(v,f)*dx
    a = mu*a11-a12-a21
    i = p*q*dx

    tic()
    AA, bb = assemble_system(a, L1, bcs)
    # A = as_backend_type(AA).mat()
    A,b = CP.Assemble(AA,bb)
    print toc()
    b = bb.array()
    zeros = 0*b
    del bb
    bb = IO.arrayToVec(b)
    x = IO.arrayToVec(zeros)


    pp = inner(grad(v), grad(u))*dx + (1/mu)*p*q*dx
    PP, Pb = assemble_system(pp,L1,bcs)
    P = CP.Assemble(PP)


    u_is = PETSc.IS().createGeneral(range(V.dim()))
    p_is = PETSc.IS().createGeneral(range(V.dim(),V.dim()+Q.dim()))

    ksp = PETSc.KSP().create()
    ksp.setOperators(A,P)
    pc = ksp.getPC()
    pc.setType(pc.Type.FIELDSPLIT)
    fields = [ ("field1", u_is), ("field2", p_is)]
    pc.setFieldSplitIS(*fields)
    pc.setFieldSplitType(0)

    OptDB = PETSc.Options()

    OptDB["field_split_type"] = "multiplicative"

    OptDB["fieldsplit_field1_ksp_type"] = "preonly"
    OptDB["fieldsplit_field1_pc_type"] = "hypre"
    OptDB["fieldsplit_field2_ksp_type"] = "cg"
    OptDB["fieldsplit_field2_pc_type"] = "jacobi"
    OptDB["fieldsplit_field2_ksp_rtol"] = 1e-8



    ksp.setFromOptions()
    ksp.setTolerances(1e-8)


    ksp.solve(bb, x)


    X = IO.vecToArray(x)
    x = X[0:V.dim()]
    p =  X[V.dim():]
    # x =
    u = Function(V)
    u.vector()[:] = x

    p_k = Function(Q)
    n = p.shape
    p_k.vector()[:] = p


    u_k.assign(u)


    return u_k, p_k