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
# import common
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
# import matplotlib.pyplot as plt
import sympy as sy
import ExactSol
import NSpreconditioner

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
H1uorder = np.zeros((m-1,1))
l2porder =  np.zeros((m-1,1))
l2border =  np.zeros((m-1,1))
Curlborder = np.zeros((m-1,1))
l2rorder =  np.zeros((m-1,1))
H1rorder = np.zeros((m-1,1))

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

G = 10.
Re = 1./params[2]
Ha = sqrt(params[0]/(params[1]*params[2]))

p = -G*x - (G**2)/(2*params[0])*(sy.sinh(y*Ha)/sy.sinh(Ha)-y)**2
u = (G/(params[2]*Ha*sy.tanh(Ha)))*(1-sy.cosh(y*Ha)/sy.cosh(Ha))
v = sy.diff(x,y)

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

Laplacian = Expression((sy.ccode(L1),sy.ccode(L2)))
Advection = Expression((sy.ccode(A1),sy.ccode(A2)))
gradPres = Expression((sy.ccode(P1),sy.ccode(P2)))

pN = as_matrix(((Expression(sy.ccode(J11)),Expression(sy.ccode(J12))), (Expression(sy.ccode(J21)),Expression(sy.ccode(J22)))))
for xx in xrange(1,m):
    print xx
    level[xx-1] = xx+1
    nn = 2**(level[xx-1])

    # Create mesh and define function space
    nn = int(nn)
    L = 10.
    y0 = 2.
    z0 = 1.

    mesh, boundaries, domains = HartmanChannel.Domain(nn)

    parameters['form_compiler']['quadrature_degree'] = -1
    parameters['reorder_dofs_serial'] = False
    Velocity = VectorFunctionSpace(mesh, "CG", 2)
    Pressure = FunctionSpace(mesh, "CG", 1)

    Velocitydim[xx-1] = Velocity.dim()
    Pressuredim[xx-1] = Pressure.dim()

    W = Velocity*Pressure
    IS = MO.IndexSet(W)
    Wdim[xx-1] = W.dim()

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    kappa = 1.0
    Mu_m = float(1e4)
    MU = 1.0
    # u0, p0, Laplacian, Advection, gradPres = ExactSol.NS2D(1, mesh)

    F = -Laplacian + Advection + gradPres


    params = [kappa,Mu_m,MU]
    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    a11 = inner(grad(v), grad(u))*dx(0)
    a12 = -div(v)*p*dx(0)
    a21 = -div(u)*q*dx(0)
    a = a11+a12+a21

    def boundary(x, on_boundary):
        return on_boundary
    n = FacetNormal(mesh)

    L = inner(v, F)*dx(0) - inner(pN*n,v)*ds(2)

    bc = DirichletBC(W.sub(0), u0, boundaries, 1)

    A, b = assemble_system(a, L, bc)
    A, b = CP.Assemble(A, b)
    x = b.duplicate()
    del a, L, bc

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


    scale = b.norm()
    b = b/scale
    ksp.setOperators(A,A)
    del A
    start_time = time.time()
    ksp.solve(b,x)
    print ("{:40}").format("Stokes solve, time: "), " ==>  ",("{:4f}").format(time.time() - start_time),("{:9}").format("   Its: "), ("{:4}").format(ksp.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    x = x*scale

    u_k = Function(Velocity)
    p_k = Function(Pressure)
    p_k.vector()[:] = x.getSubVector(IS[1]).array
    ones = Function(Pressure)
    ones.vector()[:]=(0*ones.vector().array()+1)
    # pConst = - assemble(p_k*dx)/assemble(ones*dx)
    p_k.vector()[:] += - assemble(p_k*dx)/assemble(ones*dx)
    u_k.vector()[:] = x.getSubVector(IS[0]).array

    tol = 1e-4
    maxiter = 20
    eps = 1
    iter = 0
    L = inner(v, F)*dx(0) - inner(pN*n,v)*ds(2)
    a11 = inner(grad(v), grad(u))*dx(0) + inner((grad(u)*u_k),v)*dx(0) + (1./2)*div(u_k)*inner(u,v)*dx(0) - (1./2)*inner(u_k,n)*inner(u,v)*ds(0)
    a12 = -div(v)*p*dx(0)
    a21 = -div(u)*q*dx(0)
    a = a11+a12+a21

    rhs = inner(grad(v), grad(u_k))*dx(0) + inner((grad(u_k)*u_k),v)*dx(0) + (1./2)*div(u_k)*inner(u_k,v)*dx(0) - (1./2)*inner(u_k,n)*inner(u_k,v)*ds(0) - div(v)*p_k*dx(0) - div(u_k)*q*dx(0)
    u_is = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs())

    KSPlinearfluids, MatrixLinearFluids = PrecondSetup.FluidLinearSetup(Pressure, MU,mesh, boundaries, domains)
    kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k, mesh, boundaries, domains)

    NSits = 0
    SolutionTime = 0
    TotalStart = time.time()
    while eps > tol  and iter < maxiter:
        iter += 1
        bc = DirichletBC(W.sub(0), Expression(("0.0", "0.0")), boundaries, 1)
        # bc = DirichletBC(W.sub(0), Expression(("0.0", "0.0")), boundary)
        print "Iteration = ", iter
        A, b = assemble_system(a, L-rhs, bc)
        A, b = CP.Assemble(A,b)
        kspF = NSprecondSetup.LSCKSPnonlinear(A.getSubMatrix(u_is, u_is))

        x = b.duplicate()

        # ksp = PETSc.KSP()
        # ksp.create(comm=PETSc.COMM_WORLD)
        # pc = ksp.getPC()
        # ksp.setType('preonly')
        # pc.setType('lu')
        # OptDB = PETSc.Options()
        # OptDB['pc_factor_mat_solver_package']  = "umfpack"
        # OptDB['pc_factor_mat_ordering_type']  = "rcm"
        # ksp.setFromOptions()
        kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k, mesh, boundaries, domains)

        ksp = PETSc.KSP()
        ksp.create(comm=PETSc.COMM_WORLD)
        ksp.setType('fgmres')
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(NSpreconditioner.NSPCD(W, kspF, KSPlinearfluids[0], KSPlinearfluids[1],Fp))
        ksp.setOperators(A)
        OptDB = PETSc.Options()
        ksp.max_it = 1000
        ksp.setFromOptions()


        scale = b.norm()
        b = b/scale
        ksp.setOperators(A,A)

        start_time = time.time()
        ksp.solve(b,x)
        print ("{:40}").format("Navier-Stokes solve, time: "), " ==>  ",("{:4f}").format(time.time() - start_time),("{:9}").format("   Its: "), ("{:4}").format(ksp.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
        SolutionTime += time.time() - start_time
        x = x*scale
        b1 = Function(Velocity)
        r1 = Function(Pressure)
        NSits += ksp.its
        r1.vector()[:] = x.getSubVector(IS[1]).array
        r1.vector()[:] += - assemble(r1*dx)/assemble(ones*dx)
        b1.vector()[:] = x.getSubVector(IS[0]).array

        eps = np.linalg.norm(r1.vector().array()) + np.linalg.norm(b1.vector().array())
        print "Update = ", eps, '\n\n'

        u_k.assign(u_k+b1)
        p_k.assign(p_k+r1)
        del A, b, b1, r1

    SolTime[xx-1] = SolutionTime/iter
    NSave[xx-1] = (float(NSits)/iter)
    iterations[xx-1] = iter
    TotalTime[xx-1] = time.time() - TotalStart
    VelocityE = VectorFunctionSpace(mesh,"CG", 4)
    PressureE = FunctionSpace(mesh,"CG", 3)

    u = interpolate(u0,VelocityE)
    p = interpolate(p0,PressureE)
    p.vector()[:] += - assemble(p*dx)/assemble(ones*dx)
    ErrorU = Function(Velocity)
    ErrorP = Function(Pressure)
    Q1 = np.zeros((mesh.num_vertices(),1))
    Q2 = np.zeros((mesh.num_vertices(),1))
    Q3 = np.zeros((mesh.num_vertices(),1))
    for i in xrange(0, mesh.num_vertices()):
        Q1[i] = (u_k(mesh.coordinates()[i])-u(mesh.coordinates()[i]))[0]
        Q2[i] = (u_k(mesh.coordinates()[i])-u(mesh.coordinates()[i]))[1]
        Q3[i] = p_k(mesh.coordinates()[i])-p(mesh.coordinates()[i])
    print "U-Normalised Pointwise 2-norm (first component)   = ", np.linalg.norm(Q1)/mesh.num_vertices()
    print "U-Normalised Pointwise 2-norm (Second component)  = ", np.linalg.norm(Q2)/mesh.num_vertices()
    print "P-Normalised Pointwise 2-norm                     = ", np.linalg.norm(Q3)/mesh.num_vertices()

    ErrorU = u-u_k
    ErrorP = p-p_k



    errL2b[xx-1] = sqrt(abs(assemble(inner(ErrorU, ErrorU)*dx)))
    errCurlb [xx-1] = sqrt(abs(assemble(inner(grad(ErrorU), grad(ErrorU))*dx)))#errornorm(u, u_k, norm_type='H10', degree_rise=4)
    errL2r[xx-1] = sqrt(abs(assemble(inner(ErrorP, ErrorP)*dx)))

    if xx > 1:
        l2uorder[xx-1] = abs(np.log2(errL2b[xx-1]/errL2b[xx-2]))
        H1uorder[xx-1] = abs(np.log2(errCurlb[xx-1]/errCurlb[xx-2]))
        l2rorder[xx-1] = abs(np.log2(errL2r[xx-1]/errL2r[xx-2]))




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


print "\n\n   Iteration table"
IterTitles = ["l","DoF","AV solve Time","Total picard time","picard iterations","Av NS iters"]
IterValues = np.concatenate((level,Wdim,SolTime,TotalTime,iterations,NSave),axis=1)
IterTable= pd.DataFrame(IterValues, columns = IterTitles)
IterTable = MO.PandasFormat(IterTable,'Av NS iters',"%2.1f")
print IterTable

# p1 = plot(u_k)
# p1.write_png()
# p1 = plot(p_k)
# p1.write_png()
# p1 = plot(u)
# p1.write_png()
# p1 = plot(p)
# p1.write_png()
# sss
# interactive()
