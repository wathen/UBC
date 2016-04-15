#!/usr/bin/python

# interpolate scalar gradient onto nedelec space

import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
from dolfin import *
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
import ExactSol
import Solver as S
import MHDmatrixPrecondSetup as PrecondSetup
import NSprecondSetup
import MHDprec as MHDpreconditioner
import memory_profiler
import gc
import MHDmulti
import MHDmatrixSetup as MHDsetup
from scipy.sparse import coo_matrix, csr_matrix, spdiags, bmat, linalg

#@profile
m = 2


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
parameters["form_compiler"]["no-evaluate_basis_derivatives"] = False

MU[0]= 1e0
for xx in xrange(1,m):
    print xx
    level[xx-1] = xx + 1
    nn = 2**(level[xx-1])



    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn/2
    # parameters["form_compiler"]["quadrature_degree"] = 6
    # parameters = CP.ParameterSetup()
    mesh = UnitSquareMesh(nn,nn)
    # mesh = RectangleMesh(0,0,2*np.pi,2*np.pi,nn,nn)
    order = 1
    parameters['reorder_dofs_serial'] = False
    Velocity = VectorFunctionSpace(mesh, "CG", order)
    Pressure = FunctionSpace(mesh, "CG", order)
    Magnetic = FunctionSpace(mesh, "N1curl", order)
    Lagrange = FunctionSpace(mesh, "CG", order)
    W = MixedFunctionSpace([Velocity, Pressure, Magnetic,Lagrange])
    # W = Velocity*Pressure*Magnetic*Lagrange
    Velocitydim[xx-1] = Velocity.dim()
    Pressuredim[xx-1] = Pressure.dim()
    Magneticdim[xx-1] = Magnetic.dim()
    Lagrangedim[xx-1] = Lagrange.dim()
    Wdim[xx-1] = W.dim()
    print "\n\nW:  ",Wdim[xx-1],"Velocity:  ",Velocitydim[xx-1],"Pressure:  ",Pressuredim[xx-1],"Magnetic:  ",Magneticdim[xx-1],"Lagrange:  ",Lagrangedim[xx-1],"\n\n"
    dim = [Velocity.dim(), Pressure.dim(), Magnetic.dim(), Lagrange.dim()]


    def boundary(x, on_boundary):
        return on_boundary

    u0, p0,b0, r0, Laplacian, Advection, gradPres,CurlCurl, gradR, NS_Couple, M_Couple = ExactSol.MHD2D(4,1)


    bcu = DirichletBC(Velocity,u0, boundary)
    bcb = DirichletBC(Magnetic,Expression(('0','0')), boundary)
    bcr = DirichletBC(Lagrange,Expression(('0')), boundary)

    # bc = [u0,p0,b0,r0]
    bcs = [bcu,bcb,bcr]
    FSpaces = [Velocity,Pressure,Magnetic,Lagrange]


    (u, b, p, r) = TrialFunctions(W)
    (v, c, q, s) = TestFunctions(W)
    kappa = 10.0
    Mu_m =10.0
    MU = 1.0/1
    IterType = 'Full'
    Split = "No"
    Saddle = "No"
    Stokes = "No"
    SetupType = 'python-class'
    F_NS = -MU*Laplacian+Advection+gradPres-kappa*NS_Couple
    if kappa == 0:
        F_M = Mu_m*CurlCurl+gradR -kappa*M_Couple
    else:
        F_M = Mu_m*kappa*CurlCurl+gradR -kappa*M_Couple
    params = [kappa,Mu_m,MU]

    MO.PrintStr("Seting up initial guess matricies",2,"=","\n\n","\n")
    BCtime = time.time()
    BC = MHDsetup.BoundaryIndices(mesh)
    MO.StrTimePrint("BC index function, time: ", time.time()-BCtime)
    Hiptmairtol = 1e-5
    HiptmairMatrices = PrecondSetup.MagneticSetup(Magnetic, Lagrange, b0, r0, Hiptmairtol, params)
    print HiptmairMatrices
    C = HiptmairMatrices[0]
    Px = HiptmairMatrices[1][0]
    Py = HiptmairMatrices[1][1]


    VecV = VectorFunctionSpace(mesh,"CG",1)










def HiptmairAnyOrder(Magnetic,Lagrange):
    mesh = Magnetic.mesh()
    VecLagrange = VectorFunctionSpace(mesh, "CG", Magnetic.__dict__['_FunctionSpace___degree'])

    def boundary(x, on_boundary):
        return on_boundary

    dim = mesh.geometry().dim()
    u0 = []
    for i in range(dim):
        u0.append('0.0')
    u0 = Expression(u0)
    VecBC = DirichletBC(VecLagrange, u0, boundary)
    BCb = DirichletBC(Magnetic, u0, boundary)
    BCr = DirichletBC(Lagrange, Expression(('0.0')), boundary)

    p = TestFunction(Lagrange)
    q = TrialFunction(Lagrange)
    u = TestFunction(Magnetic)
    v = TrialFunction(Magnetic)
    Vu = TestFunction(VecLagrange)
    Vv = TrialFunction(VecLagrange)

    M = assemble(inner(u,v)*dx)
    # BCb.apply(M)
    B = assemble(inner(v,grad(p))*dx)
    L = assemble(inner(grad(Vu),grad(Vv))*dx + inner(Vu,Vv)*dx)
    l = assemble(inner(grad(p),grad(q))*dx)
    VecBC.apply(L)
    BCr.apply(l)
    L = CP.Scipy2PETSc(L.sparray())
    B = CP.Scipy2PETSc(B.sparray())
    M = CP.Scipy2PETSc(M.sparray())
    l = CP.Scipy2PETSc(l.sparray())

    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    pc = ksp.getPC()
    ksp.setType('cg')
    pc.setType('bjacobi')
    ksp.setOperators(M,M)
    ksp.setTolerances(1e-8)


    return VecLagrange, ksp, L, l, B, [BCb, BCr, VecBC]



VecLagrange, ksp, L, l, B, BC = HiptmairAnyOrder(Magnetic,Lagrange)
def GradOp(ksp,B,u):
    Bu = B.createVecRight()
    B.multTranspose(u,Bu)
    v = Bu.duplicate()
    ksp.solve(Bu,v)
    return v

def TransGradOp(ksp,B,u):
    Bu = u.duplicate()
    ksp.solve(u,Bu)
    v = B.createVecLeft()
    B.mult(Bu,v)
    return v


def BCapply(V,BC,x,opt = "PETSc"):
    v = Function(V)
    v.vector()[:] = x.array
    BC.apply(v.vector())
    if opt == "PETSc":
        x = IO.arrayToVec(v.vector().array())
        return x
    else:
        return v

def PETScToFunc(V,x):
    v = Function(V)
    v.vector()[:] = x.array
    return x

def FuncToPETSc(x):
    return IO.arrayToVec(x.vector().array())


Hdiv = FunctionSpace(mesh,'BDM',order)

f = Expression(('(x[0])','(x[1])'))

Ft = interpolate(f,Magnetic)
x = IO.arrayToVec(Ft.vector().array())
print x.array
Pxx = Px.createVecRight()
Px.multTranspose(x,Pxx)
Pyy  = Py.createVecRight()
Py.multTranspose(x,Pyy)
PPP = CP.PETSc2Scipy(Px)
print (PPP*PPP.T).nnz
print (PPP*PPP.T).diagonal()

MO.StoreMatrix(PPP,"P")

P = np.concatenate((Pxx.array,Pyy.array),axis=1)
# print P
f = BCapply(Magnetic,BC[0],x,"dolfin")
fVec = interpolate(f,VecLagrange)
BC[2].apply(fVec.vector())
# plot(fVec, interactive=True)
uVec = FuncToPETSc(fVec)

for i in range(len(uVec.array)):
    print uVec.array[i], '    ', P[i]
print uVec.array
print P
print np.max(abs(uVec.array-P))

print "\n\n\n\n"

# f = Expression(('(x[0])','(x[1])'))
# Ft = interpolate(f,VecLagrange)
# x = IO.arrayToVec(Ft.vector().array())
# Pxx = Px.createVecLeft()
# PP = CP.Scipy2PETSc(bmat([[CP.PETSc2Scipy(Px),CP.PETSc2Scipy(Py)]]))
# P = Px.createVecLeft()
# PP.mult(x,P)

# # P = np.concatenate((Pxx.array,Pyy.array),axis=1)

# f = BCapply(VecLagrange,BC[2],x,"dolfin")
# fVec = interpolate(f,Magnetic)
# BC[0].apply(fVec.vector())
# uVec = FuncToPETSc(fVec)

# print uVec.array
# print P.array
# print np.max(abs(uVec.array-P.array))

## Discrete gradient test!!!!!
# f = Expression(('(x[0])','(x[1])'))
# Ft = interpolate(f,Magnetic)
# x = IO.arrayToVec(Ft.vector().array())

# # BC[1].apply(Ft.vector())
# Ft = IO.arrayToVec(Ft.vector().array())
# ft = C.getVecRight()
# C.multTranspose(Ft,ft)

# xMag = BCapply(Magnetic,BC[0],x)
# uGrad = TransGradOp(ksp,B,xMag)
# xLag1 = BCapply(Lagrange,BC[1],uGrad)

# f = Expression('sin(x[0])')
# F = interpolate(f,Lagrange)
# x = IO.arrayToVec(F.vector().array())

# # BC[2].apply(F.vector())
# F = IO.arrayToVec(F.vector().array())
# f = C.getVecLeft()
# C.mult(F,f)


# # v = GradOp(ksp,B,F)
# # vFunc = Function(Magnetic)
# # vFunc.vector()[:] = v.array
# # BC[1].apply(vFunc.vector())


# xMag = BCapply(Lagrange,BC[1],x)
# uGrad = GradOp(ksp,B,xMag)
# xLag = BCapply(Magnetic,BC[0],uGrad)


# # v = TransGradOp(ksp,B,Ft)
# # vtFunc = Function(Lagrange)
# # vtFunc.vector()[:] = v.array
# # BC[2].apply(vtFunc.vector())

# print mesh.hmin(), mesh.hmax()
# print "\n\n"
# print xLag1.array
# print ft.array
# # print "\n\n"
# print xLag.array
# print f.array


# print "\n\nNORMS"
# print np.max(abs(xLag.array-f.array))
# print np.max(abs(xLag1.array-ft.array))