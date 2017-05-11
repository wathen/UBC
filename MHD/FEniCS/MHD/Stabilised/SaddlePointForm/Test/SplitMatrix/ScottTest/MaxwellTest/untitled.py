#!/usr/bin/python

# interpolate scalar gradient onto nedelec space

import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
from dolfin import *
# from MatrixOperations import *
import numpy as np
import PETScIO as IO
import common
import scipy
import scipy.io
import time
import scipy.sparse as sp
import BiLinear as forms
import IterOperations as Iter
import MatrixOperations as MO
import CheckPetsc4py as CP
import Solver as S
import MHDmatrixPrecondSetup as PrecondSetup
import NSprecondSetup
import MHDprec as MHDpreconditioner
import gc
import MHDmulti
import MHDmatrixSetup as MHDsetup
import ExactSol
import test
# import matplotlib.pyplot as plt
#@profile
m = 8

set_log_active(False)
errL2u = np.zeros((m-1,1))
errH1u = np.zeros((m-1,1))
errL2p = np.zeros((m-1,1))
errL2b = np.zeros((m-1,1))
errCurlb = np.zeros((m-1,1))
errL2r = np.zeros((m-1,1))
errH1r = np.zeros((m-1,1))



l2uorder =  np.zeros((m-1,1))
H1uorder =np.zeros((m-1,1))
l2porder =  np.zeros((m-1,1))
l2border =  np.zeros((m-1,1))
Curlborder = np.zeros((m-1,1))
l2rorder =  np.zeros((m-1,1))
H1rorder = np.zeros((m-1,1))

NN = np.zeros((m-1,1))
DoF = np.zeros((m-1,1))
Velocitydim = np.zeros((m-1,1))
Magneticdim = np.zeros((m-1,1))
Pressuredim = np.zeros((m-1,1))
Lagrangedim = np.zeros((m-1,1))
Wdim = np.zeros((m-1,1))
iterations = np.zeros((m-1,1))
SolTime = np.zeros((m-1,1))
udiv = np.zeros((m-1,1))
MU = np.zeros((m-1,1))
level = np.zeros((m-1,1))
NSave = np.zeros((m-1,1))
Mave = np.zeros((m-1,1))
TotalTime = np.zeros((m-1,1))
DimSave = np.zeros((m-1,4))

dim = 2
ShowResultPlots = 'yes'
split = 'Linear'
MU[0] = 1e0
def PETScToScipy(A):
    data = A.getValuesCSR()
    (Istart,Iend) = A.getOwnershipRange()
    columns = A.getSize()[0]
    sparseSubMat = sp.csr_matrix(data[::-1],shape=(Iend-Istart,columns))
    return sparseSubMat

for xx in xrange(1,m):
    print xx
    level[xx-1] = xx + 1
    nn = 2**(level[xx-1])

    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn/2
    L = 10.
    y0 = 2.
    z0 = 1.
    # mesh, boundaries, domains = HartmanChannel.Domain(nn)
    mesh = UnitSquareMesh(nn, nn)

    parameters['form_compiler']['quadrature_degree'] = -1
    order = 2
    parameters['reorder_dofs_serial'] = False

    Magnetic = FiniteElement("N1curl", mesh.ufl_cell(), order-1)
    Lagrange = FiniteElement("CG", mesh.ufl_cell(), order-1)
    MagneticF = FunctionSpace(mesh, "N1curl", order-1)
    LagrangeF = FunctionSpace(mesh, "CG", order-1)

    Magneticdim[xx-1] = MagneticF.dim()

    print "Magnetic:  ",Magneticdim[xx-1]

    def boundary(x, on_boundary):
        return on_boundary
    b0 = Expression(("1.0", "0.0"), degree=4)
    (b) = TrialFunction(MagneticF)
    (c) = TestFunction(MagneticF)



    b0, r0, CurlCurl, gradPres, CurlMass = ExactSol.M2D(1, Mass=1)

    a = inner(curl(b),curl(c))*dx + inner(b, c)*dx
    t = inner(grad(b),grad(c))*dx + inner(b, c)*dx
    a1 = inner(curl(b),curl(c))*dx
    m = inner(b, c)*dx
    L = inner(CurlMass, c)*dx
    bc = DirichletBC(MagneticF, b0, boundary)

    M = assemble(m)
    bc.apply(M)
    M = CP.Assemble(M)
    AA = assemble(a1)
    bc.apply(AA)
    AA = CP.Assemble(AA)
    T = assemble(t)
    bc.apply(T)
    T = CP.Assemble(T)

    A, b = assemble_system(a, L, bc)
    A, b = CP.Assemble(A,b)
    u = b.duplicate()
    params = [1., 1., 1.]
    MO.PrintStr("Seting up initial guess matricies",2,"=","\n\n","\n")
    BCtime = time.time()
    BC = MHDsetup.BoundaryIndices(mesh)
    MO.StrTimePrint("BC index function, time: ", time.time()-BCtime)
    Hiptmairtol = 1e-6
    HiptmairMatrices = PrecondSetup.MagneticSetup(mesh, Magnetic, Lagrange, b0, r0, Hiptmairtol, params)
    G =  HiptmairMatrices[0]
    Gt = PETSc.Mat()#.createAIJ(LagrangeF.dim(),MagneticF.dim())
    G.transpose(out=Gt)
    nullVecs = []
    for i in range(0, LagrangeF.dim()):
        nullVecs.append(G.getColumnVector(i))
    TT = A + G*Gt
    null_space_final = PETSc.NullSpace()
    null_space_final.create(vectors=nullVecs)
    T.setNearNullSpace(null_space_final)
#    print (TT).view()
#    ss
    Z = M + G*Gt
    # T = PETScToScipy(T)
    # Z = PETScToScipy(Z)
    # A = PETScToScipy(A)
    # M = PETScToScipy(M)

    # invZ = sp.linalg.inv(Z)
    # invM = sp.linalg.inv(M)

    # S =  T*invZ*M
    # ss = Z*invM*A
    # S = PETSc.Mat().createAIJ(size=S.shape, csr=(S.indptr, S.indices, S.data))
    # Ss = PETSc.Mat().createAIJ(size=ss.shape, csr=(ss.indptr, ss.indices, ss.data))
    # A = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    # T = PETSc.Mat().createAIJ(size=T.shape, csr=(T.indptr, T.indices, T.data))
    # print (Ss-T).view()

    # sss
    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    pc = ksp.getPC()
    ksp.setType('cg')
    pc.setType('python')
    options = PETSc.Options()
#    options["pc_hypre_type"] = "boomeramg"
#    options["pc_hypre_boomeramg_nodal_coarsen"] = 2
#    options["pc_hypre_boomeramg_vec_interp_variant"] = 3
    pc.setPythonContext(test.ShiftedCurlCurl(Z, M, T))
    scale = b.norm()
    ksp.setOperators(A, A)
    reshist = {}
    def monitor(ksp, its, fgnorm):
        reshist[its] = fgnorm
        print its,"    OUTER:", fgnorm
    ksp.setMonitor(monitor)
    ksp.solve(b,u)
    # Mits +=dodim
    print (b-A*u).norm()


    print "                iter =", ksp.its
