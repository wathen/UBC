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
import HartmanChannel
import ExactSol
# import matplotlib.pyplot as plt
#@profile
m = 6


def PETScToScipy(A):
    data = A.getValuesCSR()
    sparseSubMat = sp.csr_matrix(data[::-1], shape=A.size)
    return sparseSubMat


def savePETScMat(A, name1, name2):
    A_ = PETScToScipy(A)
    scipy.io.savemat(name1, mdict={name2: A_})

set_log_active(False)
errL2u = np.zeros((m-1, 1))
errH1u = np.zeros((m-1, 1))
errL2p = np.zeros((m-1, 1))
errL2b = np.zeros((m-1, 1))
errCurlb = np.zeros((m-1, 1))
errL2r = np.zeros((m-1, 1))
errH1r = np.zeros((m-1, 1))


l2uorder = np.zeros((m-1, 1))
H1uorder = np.zeros((m-1, 1))
l2porder = np.zeros((m-1, 1))
l2border = np.zeros((m-1, 1))
Curlborder = np.zeros((m-1, 1))
l2rorder = np.zeros((m-1, 1))
H1rorder = np.zeros((m-1, 1))

NN = np.zeros((m-1, 1))
DoF = np.zeros((m-1, 1))
Velocitydim = np.zeros((m-1, 1))
Magneticdim = np.zeros((m-1, 1))
Pressuredim = np.zeros((m-1, 1))
Lagrangedim = np.zeros((m-1, 1))
Wdim = np.zeros((m-1, 1))
iterations = np.zeros((m-1, 1))
SolTime = np.zeros((m-1, 1))
udiv = np.zeros((m-1, 1))
MU = np.zeros((m-1, 1))
level = np.zeros((m-1, 1))
NSave = np.zeros((m-1, 1))
Mave = np.zeros((m-1, 1))
TotalTime = np.zeros((m-1, 1))
DimSave = np.zeros((m-1, 4))

dim = 2
ShowResultPlots = 'yes'
MU[0] = 1e0

for xx in xrange(1, m):
    print xx
    level[xx-1] = xx + 2
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
    Velocity = VectorElement("CG", mesh.ufl_cell(), order)
    Pressure = FiniteElement("CG", mesh.ufl_cell(), order-1)
    Magnetic = FiniteElement("N1curl", mesh.ufl_cell(), order-1)
    Lagrange = FiniteElement("CG", mesh.ufl_cell(), order-1)

    VelocityF = VectorFunctionSpace(mesh, "CG", order)
    PressureF = FunctionSpace(mesh, "CG", order-1)
    MagneticF = FunctionSpace(mesh, "N1curl", order-1)
    LagrangeF = FunctionSpace(mesh, "CG", order-1)
    W = FunctionSpace(mesh, MixedElement(
        [Velocity, Pressure, Magnetic, Lagrange]))

    Velocitydim[xx-1] = W.sub(0).dim()
    Pressuredim[xx-1] = W.sub(1).dim()
    Magneticdim[xx-1] = W.sub(2).dim()
    Lagrangedim[xx-1] = W.sub(3).dim()
    Wdim[xx-1] = W.dim()

    print "\n\nW:  ", Wdim[xx-1], "Velocity:  ", Velocitydim[xx-1], "Pressure:  ", Pressuredim[xx-1], "Magnetic:  ", Magneticdim[xx-1], "Lagrange:  ", Lagrangedim[xx-1], "\n\n"

    dim = [W.sub(0).dim(), W.sub(1).dim(), W.sub(2).dim(), W.sub(3).dim()]

    def boundary(x, on_boundary):
        return on_boundary

    FSpaces = [VelocityF, PressureF, MagneticF, LagrangeF]
    DimSave[xx-1, :] = np.array(dim)

    kappa = 1.0
    Mu_m = 10.0
    MU = 1.0

    N = FacetNormal(mesh)

    IterType = 'Full'

    params = [kappa, Mu_m, MU]
    n = FacetNormal(mesh)
    u0, p0, b0, r0, Laplacian, Advection, gradPres, CurlCurl, gradR, NS_Couple, M_Couple = ExactSol.MHD2D(
        4, 1)

    MO.PrintStr("Seting up initial guess matricies", 2, "=", "\n\n", "\n")
    BCtime = time.time()
    BC = MHDsetup.BoundaryIndices(mesh)
    MO.StrTimePrint("BC index function, time: ", time.time()-BCtime)
    Hiptmairtol = 1e-6
    HiptmairMatrices = PrecondSetup.MagneticSetup(
        mesh, Magnetic, Lagrange, b0, r0, Hiptmairtol, params)

    MO.PrintStr("Setting up MHD initial guess", 5, "+", "\n\n", "\n\n")

    F_NS = -MU*Laplacian + Advection + gradPres - kappa*NS_Couple
    if kappa == 0.0:
        F_M = Mu_m*CurlCurl + gradR - kappa*M_Couple
    else:
        F_M = Mu_m*kappa*CurlCurl + gradR - kappa*M_Couple

    u_k, p_k = HartmanChannel.Stokes(
        Velocity, Pressure, F_NS, u0, 1, params, mesh)
    b_k, r_k = HartmanChannel.Maxwell(
        Magnetic, Lagrange, F_M, b0, r0, params, mesh, HiptmairMatrices, Hiptmairtol)

    du = TrialFunction(W)
    (v, q, c, s) = TestFunctions(W)
    u, p, b, r = split(du)

    if kappa == 0.0:
        m11 = params[1]*inner(curl(b), curl(c))*dx
    else:
        m11 = params[1]*params[0]*inner(curl(b), curl(c))*dx
    m21 = inner(c, grad(r))*dx
    m12 = inner(b, grad(s))*dx

    a11 = params[2]*inner(grad(v), grad(u))*dx + inner((grad(u)*u_k), v)*dx + (
        1./2)*div(u_k)*inner(u, v)*dx - (1./2)*inner(u_k, n)*inner(u, v)*ds
    a12 = -div(v)*p*dx
    a21 = -div(u)*q*dx

    CoupleT = params[0]*(v[0]*b_k[1]-v[1]*b_k[0])*curl(b)*dx
    Couple = -params[0]*(u[0]*b_k[1]-u[1]*b_k[0])*curl(c)*dx

    Ftilde = inner((grad(u_k)*u), v)*dx + (1./2)*div(u) * \
        inner(u_k, v)*dx - (1./2)*inner(u, n)*inner(u_k, v)*ds
    Mtilde = -params[0]*(u_k[0]*b[1]-u_k[1]*b[0])*curl(c)*dx
    Ctilde = params[0]*(v[0]*b[1]-v[1]*b[0])*curl(b_k)*dx

    a = m11 + m12 + m21 + a11 + a21 + a12 + \
        Couple + CoupleT + Ftilde + Mtilde + Ctilde
    aa = m11 + m12 + m21 + a11 + a21 + a12 + Couple + CoupleT

    if kappa == 0.0:
        m11 = params[1]*inner(curl(b_k), curl(c))*dx
    else:
        m11 = params[1]*params[0]*inner(curl(b_k), curl(c))*dx
    m21 = inner(c, grad(r_k))*dx
    m12 = inner(b_k, grad(s))*dx

    a11 = params[2]*inner(grad(v), grad(u_k))*dx + inner((grad(u_k)*u_k), v)*dx + (
        1./2)*div(u_k)*inner(u_k, v)*dx - (1./2)*inner(u_k, n)*inner(u_k, v)*ds
    a12 = -div(v)*p_k*dx
    a21 = -div(u_k)*q*dx

    CoupleT = params[0]*(v[0]*b_k[1]-v[1]*b_k[0])*curl(b_k)*dx
    Couple = -params[0]*(u_k[0]*b_k[1]-u_k[1]*b_k[0])*curl(c)*dx

    Lns = inner(v, F_NS)*dx
    Lmaxwell = inner(c, F_M)*dx

    L = Lns + Lmaxwell - (m11 + m12 + m21 + a11 + a21 + a12 + Couple + CoupleT)
    x = Iter.u_prev(u_k, p_k, b_k, r_k)

    KSPlinearfluids, MatrixLinearFluids = PrecondSetup.FluidLinearSetup(
        PressureF, MU, mesh)
    kspFp, Fp = PrecondSetup.FluidNonLinearSetup(PressureF, MU, u_k, mesh)

    F = Lns + Lmaxwell - aa

    Hiptmairtol = 1e-4
    HiptmairMatrices = PrecondSetup.MagneticSetup(
        mesh, Magnetic, Lagrange, b0, r0, Hiptmairtol, params)
    IS = MO.IndexSet(W, 'Blocks')

    ones = Function(PressureF)
    ones.vector()[:] = (0*ones.vector().array()+1)
    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-4         # tolerance
    iter = 0            # iteration counter
    maxiter = 1       # max no of iterations allowed
    SolutionTime = 0
    outer = 0
    # parameters['linear_algebra_backend'] = 'uBLAS'

    u_is = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs())
    p_is = PETSc.IS().createGeneral(W.sub(1).dofmap().dofs())
    b_is = PETSc.IS().createGeneral(W.sub(2).dofmap().dofs())
    r_is = PETSc.IS().createGeneral(W.sub(3).dofmap().dofs())
    NS_is = PETSc.IS().createGeneral(range(VelocityF.dim()+PressureF.dim()))
    M_is = PETSc.IS().createGeneral(range(VelocityF.dim()+PressureF.dim(), W.dim()))
    bcu = DirichletBC(W.sub(0), Expression(("0.0", "0.0"), degree=4), boundary)
    bcb = DirichletBC(W.sub(2), Expression(("0.0", "0.0"), degree=4), boundary)
    bcr = DirichletBC(W.sub(3), Expression(("0.0"), degree=4), boundary)
    bcs = [bcu, bcb, bcr]

    U = Function(W)      # the most recently computed solution
    F = action(aa, U)
    print assemble(dolfin.Jacobian(F))

    J = derivative(F, U, du)
    print J
    J = assemble(J)
    J = CP.Assemble(J)

    OuterTol = 1e-5
    InnerTol = 1e-5
    NSits = 0
    Mits = 0
    TotalStart = time.time()
    SolutionTime = 0
    errors = np.array([])
    bcu1 = DirichletBC(VelocityF, Expression(
        ("0.0", "0.0"), degree=4), boundary)
    U = x
    while eps > tol and iter < maxiter:
        iter += 1
        MO.PrintStr("Iter "+str(iter), 7, "=", "\n\n", "\n\n")

        A, b = assemble_system(aa, L)
        A, b = CP.Assemble(A, b)

        savePETScMat(J, "J", "J")
        savePETScMat(A, "A", "A")
        ss
