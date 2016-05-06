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
m = 6


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
Velocitydim = np.zeros((m-1,1))
Pressuredim = np.zeros((m-1,1))
Pressuredim = np.zeros((m-1,1))
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
    # mesh = RectangleMesh(Point(0., -1.), Point(10., 1.),nn,nn)
    mesh = RectangleMesh(0., -1., 10., 1., 5*nn, nn)
    # mesh, boundaries, domains = HartmanChannel.Domain(nn)
    # mesh = UnitSquareMesh(nn,nn)
    # set_log_level(WARNING)
    # p = plot(mesh)
    # p.write_png()
    # sss
    parameters['form_compiler']['quadrature_degree'] = -1
    order = 1
    parameters['reorder_dofs_serial'] = False
    Velocity = VectorFunctionSpace(mesh, "CG", 2)
    Pressure = FunctionSpace(mesh, "CG", 1)

    parameters['reorder_dofs_serial'] = False

    W = Velocity*Pressure
    IS = MO.IndexSet(W)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    kappa = 1.0
    Mu_m = float(1e4)
    MU = 1.0
    print v.shape()
    params = [kappa,Mu_m,MU]
    a11 = params[2]*inner(grad(v), grad(u))*dx
    a12 = -div(v)*p*dx
    a21 = -div(u)*q*dx
    a = a11+a12+a21
    # print F
    # u0, b0, p0, r0, F_S, F_M = HartmanChannel.ExactSol22(mesh, params)

    def boundary(x, on_boundary):
        return on_boundary

    # u0 = Expression(("x[1]","x[0]"))
    # F = Expression(("1.0","0.0"))
    # p0 = Expression("x[0]")

    class u_in(Expression):
        def __init__(self):
            self.p = 1
        def eval_cell(self, values, x, ufc_cell):
            values[0] = x[1]*x[1]
            values[1] = x[0]*x[0]
        def value_shape(self):
            return (2,)

    class p_in(Expression):
        def __init__(self):
            self.p = 1
        def eval_cell(self, values, x, ufc_cell):
            values[0] = x[0]

    class f_in(Expression):
        def __init__(self):
            self.p = 1
        def eval_cell(self, values, x, ufc_cell):
            values[0] = -1.0
            values[1] = 2.0
        def value_shape(self):
            return (2,)

    F = f_in()
    u = u_in()
    p = p_in()
    print F, v
    L = inner(v, F)*dx

    bc = DirichletBC(W.sub(0), u0, boundary)


    A, b = assemble_system(a, L, bc)
    A, b = CP.Assemble(A, b)
    x = b.duplicate()


    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    pc = ksp.getPC()
    ksp.setType('preonly')
    pc.setType('lu')
    OptDB = PETSc.Options()
    # if __version__ != '1.6.0':
    OptDB['pc_factor_mat_solver_package']  = "umfpack"
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
    ksp.solve(b,x)
    print ("{:40}").format("Maxwell solve, time: "), " ==>  ",("{:4f}").format(time.time() - start_time),("{:9}").format("   Its: "), ("{:4}").format(ksp.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    x = x*scale

    b_k = Function(Velocity)
    r_k = Function(Pressure)
    r_k.vector()[:] = x.getSubVector(IS[1]).array
    ones = Function(Pressure)
    ones.vector()[:]=(0*ones.vector().array()+1)
    # pConst = - assemble(p_k*dx)/assemble(ones*dx)
    r_k.vector()[:] += - assemble(r_k*dx)/assemble(ones*dx)
    b_k.vector()[:] = x.getSubVector(IS[0]).array

    VelocityE = VectorFunctionSpace(mesh,"CG", 4)
    PressureE = FunctionSpace(mesh,"CG", 3)

    b = interpolate(u0,VelocityE)
    r = interpolate(p0,PressureE)
    ones = Function(PressureE)
    ones.vector()[:]=(0*ones.vector().array()+1)
    # pConst = - assemble(p_k*dx)/assemble(ones*dx)
    r.vector()[:] += - assemble(r*dx)/assemble(ones*dx)
    ErrorB = Function(Velocity)
    ErrorR = Function(Pressure)


    ErrorB = b-b_k
    ErrorR = r-r_k

    # print b_k.vector().array()
    # print b.vector().array()


    errL2b[xx-1] = sqrt(abs(assemble(inner(ErrorB, ErrorB)*dx)))
    errCurlb [xx-1] = errornorm(b, b_k, norm_type='H10', degree_rise=4)
    errL2r[xx-1] = sqrt(abs(assemble(inner(ErrorR, ErrorR)*dx)))

# p = plot(u_k)
# p.write_png()
# p = plot(p_k)
# p.write_png()
# p = plot(b_k)
# p.write_png()
p = plot(r_k)
# p.write_png()
# p = plot(interpolate(u0,Velocity))
# p.write_png()

p = plot(r)
# p.write_png()
# p = plot(interpolate(b0,Magnetic))
# p.write_png()
# p = plot(interpolate(r0,Lagrange))
# p.write_png()
# sss

import pandas as pd
print "\n\n   Velocity convergence"
VelocityTitles = ["l","U DoF","P DoF","U-L2","L2-order","U-Grad","HGrad-order"]
VelocityValues = np.concatenate((level,Velocitydim,Pressuredim,errL2b,l2border,errCurlb,Curlborder),axis=1)
VelocityTable= pd.DataFrame(VelocityValues, columns = VelocityTitles)
pd.set_option('precision',3)
VelocityTable = MO.PandasFormat(VelocityTable,"U-Grad","%2.4e")
VelocityTable = MO.PandasFormat(VelocityTable,'U-L2',"%2.4e")
VelocityTable = MO.PandasFormat(VelocityTable,"L2-order","%1.2f")
VelocityTable = MO.PandasFormat(VelocityTable,'HGrad-order',"%1.2f")
print VelocityTable.to_latex()

print "\n\n   Pressure convergence"
PressureTitles = ["l","B DoF","R DoF","R-L2","L2-order","R-H1","H1-order"]
PressureValues = np.concatenate((level,Velocitydim,Pressuredim,errL2r,l2rorder,errH1r,H1rorder),axis=1)
PressureTable= pd.DataFrame(PressureValues, columns = PressureTitles)
pd.set_option('precision',3)
PressureTable = MO.PandasFormat(PressureTable,"R-L2","%2.4e")
PressureTable = MO.PandasFormat(PressureTable,'R-H1',"%2.4e")
PressureTable = MO.PandasFormat(PressureTable,"L2-order","%1.2f")
PressureTable = MO.PandasFormat(PressureTable,'H1-order',"%1.2f")
print PressureTable.to_latex()
interactive()
