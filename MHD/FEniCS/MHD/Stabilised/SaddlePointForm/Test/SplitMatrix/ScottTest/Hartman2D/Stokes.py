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
import StokesPrecond
# import matplotlib.pyplot as plt
import sympy as sy
#@profile
m = 8


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

x = sy.Symbol('x[0]')
y = sy.Symbol('x[1]')

# u = sy.diff(x,x)
# v = sy.diff(x,x)
# p = sy.diff(x,x)

u = y**2
v = x**2
p = x
p = sy.sin(x)*sy.exp(y)
uu = y*x*sy.exp(x+y)
u = sy.diff(uu,y)
v = -sy.diff(uu,x)

kappa = 1.0
Mu_m = float(1e4)
MU = 1.0
params = [kappa,Mu_m,MU]

# G = 10.
# Re = 1./params[2]
# Ha = sqrt(params[0]/(params[1]*params[2]))

# p = -G*x - (G**2)/(2*params[0])*(sy.sinh(y*Ha)/sy.sinh(Ha)-y)**2
# u = (G/(params[2]*Ha*sy.tanh(Ha)))*(1-sy.cosh(y*Ha)/sy.cosh(Ha))
# v = sy.diff(x,y)

L1 = sy.diff(u,x,x) + sy.diff(u,y,y)
L2 = sy.diff(v,x,x) + sy.diff(v,y,y)

print "u=(", u,",", v,")"
print "p=(", p,")"

P1 = sy.diff(p,x)
P2 = sy.diff(p,y)

A1 = u*sy.diff(u,x)+v*sy.diff(u,y)
A2 = u*sy.diff(v,x)+v*sy.diff(v,y)

F1 = -L1 + P1 + A1
F2 = -L2 + P2 + A1

J11 = p - sy.diff(u,x)
J12 = - sy.diff(u,y)
J21 = - sy.diff(v,x)
J22 = p - sy.diff(v,y)

u0 = Expression((sy.ccode(u),sy.ccode(v)))
p0 = Expression(sy.ccode(p))
# Laplacian = Vec(L1, L2, x, y)
# Advection = Vec(A1, A2, x, y)
# gradPres = Vec(P1, P2, x, y)
Laplacian = Expression((sy.ccode(L1),sy.ccode(L2)))
Advection = Expression((sy.ccode(A1),sy.ccode(A2)))
gradPres = Expression((sy.ccode(P1),sy.ccode(P2)))

# F = f_in(F1, F2)
# u0 = u_in(u, v)
# p0 = p_in(p)
# J = J(J11, J12, J21, J22)
pN = as_matrix(((Expression(sy.ccode(J11)),Expression(sy.ccode(J12))), (Expression(sy.ccode(J21)),Expression(sy.ccode(J22)))))

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
    # mesh = RectangleMesh(0., -1., 10., 1., 5*nn, nn)
    mesh, boundaries, domains = HartmanChannel.Domain(nn)
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

    Velocitydim[xx-1] = Velocity.dim()
    Pressuredim[xx-1] = Pressure.dim()

    W = Velocity*Pressure
    IS = MO.IndexSet(W)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    kappa = 1.0
    Mu_m = float(1e4)
    MU = 1.0
    print v.shape()
    params = [kappa,Mu_m,MU]
    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)


    a11 = inner(grad(v), grad(u))*dx(0)
    a12 = -div(v)*p*dx(0)
    a21 = -div(u)*q*dx(0)
    a = a11+a12+a21
    # print F
    # u0, b0, p0, r0, F_S, F_M = HartmanChannel.ExactSol22(mesh, params)

    def boundary(x, on_boundary):
        return on_boundary

    # u0 = Expression(("x[1]","x[0]"))
    # F = Expression(("1.0","0.0"))
    # p0 = Expression("x[0]")

    F = -Laplacian + gradPres
    n = FacetNormal(mesh)

    L = inner(v, F)*dx(0)

    bc = DirichletBC(W.sub(0), u0, boundaries, 1)


    # r1 = b.array
    L = inner(v, F)*dx(0) - inner(pN*n,v)*ds(2)

    A, b = assemble_system(a, L, bc)
    A, b = CP.Assemble(A, b)
    # r2 = b.array
    # print r1
    x = b.duplicate()

    P, Pb = assemble_system(a11 + inner(p ,q)*dx, L, bc)
    P, Pb = CP.Assemble(P, Pb)

    # ksp = PETSc.KSP()
    # ksp.create(comm=PETSc.COMM_WORLD)
    # pc = ksp.getPC()
    # ksp.setType('minres')
    # pc.setType('lu')
    # OptDB = PETSc.Options()
    # # if __version__ != '1.6.0':
    # # OptDB['pc_factor_mat_solver_package']  = "umfpack"
    # # OptDB['pc_factor_mat_ordering_type']  = "rcm"
    # ksp.setFromOptions()

    ksp = PETSc.KSP().create()
    ksp.setTolerances(1e-8)
    ksp.max_it = 200
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    ksp.setType('minres')
    pc.setPythonContext(StokesPrecond.Approx(W, 1))
    scale = b.norm()
    b = b/scale
    ksp.setOperators(A,P)
    del A
    start_time = time.time()
    ksp.solve(b,x)
    print ("{:40}").format("Stokes solve, time: "), " ==>  ",("{:4f}").format(time.time() - start_time),("{:9}").format("   Its: "), ("{:4}").format(ksp.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    x = x*scale

    b_k = Function(Velocity)
    r_k = Function(Pressure)
    r_k.vector()[:] = x.getSubVector(IS[1]).array
    ones = Function(Pressure)
    ones.vector()[:]=(0*ones.vector().array()+1)
    # pConst = - assemble(p_k*dx)/assemble(ones*dx)
    r_k.vector()[:] += - assemble(r_k*dx)/assemble(ones*dx)
    b_k.vector()[:] = x.getSubVector(IS[0]).array
    print "                                     its = ", ksp.its
    # parameters['form_compiler']['quadrature_degree'] = -1

    VelocityE = VectorFunctionSpace(mesh,"CG", 3)
    PressureE = FunctionSpace(mesh,"CG", 2)

    b = interpolate(u0,VelocityE)
    r = interpolate(p0,PressureE)
    ones = Function(PressureE)
    ones.vector()[:]=(0*ones.vector().array()+1)
    # pConst = - assemble(p_k*dx)/assemble(ones*dx)
    r.vector()[:] += - assemble(r*dx)/assemble(ones*dx)
    ErrorB = Function(Velocity)
    ErrorR = Function(Pressure)
    def my_range(start, end, step):
        while start <= end:
            yield start
            start += step
    j = 0
    # Q = np.zeros((10000,2))
    # QQ = np.zeros((10000,2))
    # for x in my_range(0, 1, 0.01):
    #     for y in my_range(0, 1, 0.01):
    #         Q[j,:] = b_k(np.array([x,y])) - np.array([1.0,1.0])
    #         QQ[j] = np.max(b_k(np.array([x,y])) - np.array([1.0,1.0]))
    #         j = j+1
    # print j
    # print np.linalg.norm(Q, ord=np.inf)/10000, np.linalg.norm(Q)/10000
    # print np.linalg.norm(QQ, ord=np.inf)/10000, np.linalg.norm(QQ)/10000
    Q1 = np.zeros((mesh.num_vertices(),1))
    Q2 = np.zeros((mesh.num_vertices(),1))
    for i in xrange(0, mesh.num_vertices()):
        Q1[i] = (b_k(mesh.coordinates()[i])-b(mesh.coordinates()[i]))[0]
        Q2[i] = (b_k(mesh.coordinates()[i])-b(mesh.coordinates()[i]))[1]

    print "Normalised Pointwise 2-norm (first component)  = ", np.linalg.norm(Q1)/mesh.num_vertices()
    print "Normalised Pointwise 2-norm (Second component) = ", np.linalg.norm(Q2)/mesh.num_vertices()

    ErrorB = b-b_k
    ErrorR = r-r_k
    # print b_k.vector().array()
    # print b.vector().array()

    errL2b[xx-1] = sqrt(abs(assemble(inner(ErrorB, ErrorB)*dx)))
    errCurlb [xx-1] = errornorm(b, b_k, norm_type='H10', degree_rise=4)
    errL2r[xx-1] = sqrt(abs(assemble(inner(ErrorR, ErrorR)*dx)))

    if xx > 1:
        l2uorder[xx-1] = abs(np.log2(errL2b[xx-1]/errL2b[xx-2]))
        H1uorder[xx-1] = abs(np.log2(errCurlb[xx-1]/errCurlb[xx-2]))
        l2rorder[xx-1] = abs(np.log2(errL2r[xx-1]/errL2r[xx-2]))


# p = plot(b_k)
# p.write_png()
# p = plot(r_k)
# p.write_png()
# p = plot(b)
# p.write_png()
# p = plot(mesh)
# p.write_png()
# sss

import pandas as pd
print "\n\n   Velocity convergence"
VelocityTitles = ["l","U DoF","P DoF","U-L2","L2-order","U-Grad","HGrad-order"]
VelocityValues = np.concatenate((level,Velocitydim,Pressuredim,errL2b,l2uorder,errCurlb,H1uorder),axis=1)
VelocityTable= pd.DataFrame(VelocityValues, columns = VelocityTitles)
pd.set_option('precision',3)
VelocityTable = MO.PandasFormat(VelocityTable,"U-Grad","%2.4e")
VelocityTable = MO.PandasFormat(VelocityTable,'U-L2',"%2.4e")
VelocityTable = MO.PandasFormat(VelocityTable,"L2-order","%1.2f")
VelocityTable = MO.PandasFormat(VelocityTable,'HGrad-order',"%1.2f")
print VelocityTable.to_latex()

print "\n\n   Pressure convergence"
PressureTitles = ["l","U DoF","P DoF","P-L2","L2-order"]
PressureValues = np.concatenate((level,Velocitydim,Pressuredim,errL2r,l2rorder),axis=1)
PressureTable= pd.DataFrame(PressureValues, columns = PressureTitles)
pd.set_option('precision',3)
PressureTable = MO.PandasFormat(PressureTable,"P-L2","%2.4e")
PressureTable = MO.PandasFormat(PressureTable,"L2-order","%1.2f")
print PressureTable.to_latex()
interactive()
