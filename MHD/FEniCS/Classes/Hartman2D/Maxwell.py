#!/usr/bin/python

# interpolate scalar gradient onto nedelec space

import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
from dolfin import *
import mshr
Print = PETSc.Sys.Print
# from MatrixOperations import *
import numpy as np
import PETScIO as IO
import common
import scipy
import scipy.io
import time

import BiLinear as forms
import IterOperations as Iter
import MatrixOperations as MO
import CheckPetsc4py as CP
import Solver as S
import MHDmatrixPrecondSetup as PrecondSetup
import NSprecondSetup
import MHDprec as MHDpreconditioner
import memory_profiler
import gc
import MHDmulti
import MHDmatrixSetup as MHDsetup
import HartmanChannel
import matplotlib.pyplot as plt
#@profile
m = 5


set_log_active(False)
errL2u =np.zeros((m-1,1))
errH1u =np.zeros((m-1,1))
errL2p =np.zeros((m-1,1))
errL2b =np.zeros((m-1,1))
errCurlb =np.zeros((m-1,1))
errL2r =np.zeros((m-1,1))
errH1r =np.zeros((m-1,1))



l2uorder =  np.zeros((m-1,1))
H1uorder =np.zeros((m-1,1))
l2porder =  np.zeros((m-1,1))
l2border =  np.zeros((m-1,1))
Curlborder =np.zeros((m-1,1))
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

nn = 2

dim = 2
ShowResultPlots = 'yes'
split = 'Linear'
MU[0]= 1e0
for xx in xrange(1,m):
    print xx
    level[xx-1] = xx+1
    nn = 2**(level[xx-1])

    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn/2
    L = 10.
    y0 = 2.
    z0 = 1.
    mesh, boundaries, domains = HartmanChannel.Domain(nn)
    # mesh = UnitSquareMesh(nn,nn)
    # set_log_level(WARNING)
    # p = plot(mesh)
    # p.write_png()
    # sss
    parameters['form_compiler']['quadrature_degree'] = -1
    order = 1
    parameters['reorder_dofs_serial'] = False
    Magnetic = FunctionSpace(mesh, "N1curl", order)
    Lagrange = FunctionSpace(mesh, "CG", order)

    parameters['reorder_dofs_serial'] = False

    W = Magnetic*Lagrange
    IS = MO.IndexSet(W)

    (b, r) = TrialFunctions(W)
    (c, s) = TestFunctions(W)

    kappa = 1.0
    Mu_m = float(1e4)
    MU = 1.0

    params = [kappa,Mu_m,MU]
    a11 = params[1]*params[2]*inner(curl(b), curl(c))*dx
    a21 = inner(b,grad(s))*dx
    a12 = inner(c,grad(r))*dx
    # print F
    F = Expression(("0.0","0.0"))
    L = inner(c, F)*dx
    a = a11+a12+a21
    u0, b0, p0, r0, F_S, F_M = HartmanChannel.ExactSol22(mesh, params)

    def boundary(x, on_boundary):
        return on_boundary
    class b_in(Expression):
        def __init__(self):
            self.p = 1
        def eval_cell(self, values, x, ufc_cell):
            values[0] = 0.0
            values[1] = 1.0
        def value_shape(self):
            return (2,)

    bcb = DirichletBC(W.sub(0), b_in(), boundary)
    bcr = DirichletBC(W.sub(1), r0, boundary)
    bc = [bcb, bcr]

    A, b = assemble_system(a, L, bc)
    A, b = CP.Assemble(A, b)
    u = b.duplicate()

    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    pc = ksp.getPC()
    ksp.setType('preonly')
    pc.setType('lu')
    OptDB = PETSc.Options()
    # if __version__ != '1.6.0':
    OptDB['pc_factor_mat_solver_package']  = "mumps"
    OptDB['pc_factor_mat_ordering_type']  = "rcm"
    ksp.setFromOptions()

    # ksp = PETSc.KSP().create()
    # ksp.setTolerances(1e-8)
    # ksp.max_it = 200
    # pc = ksp.getPC()
    # pc.setType(PETSc.PC.Type.PYTHON)
    # ksp.setType('minres')
    # pc.setPythonContext(MP.Hiptmair(W, HiptmairMatrices[3], HiptmairMatrices[4], HiptmairMatrices[2], HiptmairMatrices[0], HiptmairMatrices[1], HiptmairMatrices[6],Hiptmairtol))
    scale = b.norm()
    b = b/scale
    ksp.setOperators(A,A)
    del A
    start_time = time.time()
    ksp.solve(b,u)
    print ("{:40}").format("Maxwell solve, time: "), " ==>  ",("{:4f}").format(time.time() - start_time),("{:9}").format("   Its: "), ("{:4}").format(ksp.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    u = u*scale

    b_k = Function(Magnetic)
    r_k = Function(Lagrange)
    b_k.vector()[:] = u.getSubVector(IS[0]).array
    r_k.vector()[:] = u.getSubVector(IS[1]).array

    MagneticE = FunctionSpace(mesh,"N1curl", 2)
    LagrangeE = FunctionSpace(mesh,"CG", 2)

    b = interpolate(b_in(),MagneticE)
    r = interpolate(r0,LagrangeE)

    ErrorB = Function(Magnetic)
    ErrorR = Function(Lagrange)


    ErrorB = b-b_k
    ErrorR = r-r_k

    print b_k.vector().array()
    print b.vector().array()

    tic()
    errL2b[xx-1] = sqrt(abs(assemble(inner(ErrorB, ErrorB)*dx)))
    MO.StrTimePrint("Magnetic L2 error, time: ", toc())
    tic()
    errCurlb [xx-1] = sqrt(abs(assemble(inner(curl(ErrorB),curl(ErrorB))*dx)))
    MO.StrTimePrint("Magnetic Curl error, time: ", toc())
    tic()
    errL2r[xx-1] = sqrt(abs(assemble(inner(ErrorR, ErrorR)*dx)))
    MO.StrTimePrint("Multiplier L2 error, time: ", toc())
    tic()
    errH1r[xx-1] = sqrt(abs(assemble(inner(grad(ErrorR), grad(ErrorR))*dx)))
    MO.StrTimePrint("Multiplier H1 error, time: ", toc())

import pandas as pd
print "\n\n   Magnetic convergence"
MagneticTitles = ["l","B DoF","R DoF","B-L2","L2-order","B-Curl","HCurl-order"]
MagneticValues = np.concatenate((level,Magneticdim,Lagrangedim,errL2b,l2border,errCurlb,Curlborder),axis=1)
MagneticTable= pd.DataFrame(MagneticValues, columns = MagneticTitles)
pd.set_option('precision',3)
MagneticTable = MO.PandasFormat(MagneticTable,"B-Curl","%2.4e")
MagneticTable = MO.PandasFormat(MagneticTable,'B-L2',"%2.4e")
MagneticTable = MO.PandasFormat(MagneticTable,"L2-order","%1.2f")
MagneticTable = MO.PandasFormat(MagneticTable,'HCurl-order',"%1.2f")
print MagneticTable.to_latex()

print "\n\n   Lagrange convergence"
LagrangeTitles = ["l","B DoF","R DoF","R-L2","L2-order","R-H1","H1-order"]
LagrangeValues = np.concatenate((level,Magneticdim,Lagrangedim,errL2r,l2rorder,errH1r,H1rorder),axis=1)
LagrangeTable= pd.DataFrame(LagrangeValues, columns = LagrangeTitles)
pd.set_option('precision',3)
LagrangeTable = MO.PandasFormat(LagrangeTable,"R-L2","%2.4e")
LagrangeTable = MO.PandasFormat(LagrangeTable,'R-H1',"%2.4e")
LagrangeTable = MO.PandasFormat(LagrangeTable,"L2-order","%1.2f")
LagrangeTable = MO.PandasFormat(LagrangeTable,'H1-order',"%1.2f")
print LagrangeTable.to_latex()

