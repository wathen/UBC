#!/usr/bin/python

import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
from dolfin import *
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
m = 8

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
# split = 'Linear'
MU[0] = 1e0


def PETScToScipy(A):
    data = A.getValuesCSR()
    sparseSubMat = sp.csr_matrix(data[::-1], shape=A.size)
    return sparseSubMat


def savePETScMat(A, name1, name2):
    A_ = PETScToScipy(A)
    scipy.io.savemat(name1, mdict={name2: A_})

for xx in xrange(1, m):
    print xx
    level[xx-1] = xx + 0
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

    print "\n\nW:  ", Wdim[xx-1], "Velocity:  ", Velocitydim[xx-1],
    "Pressure:  ", Pressuredim[xx-1], "Magnetic:  ", Magneticdim[xx-1],
    "Lagrange:  ", Lagrangedim[xx-1], "\n\n"

    dim = [W.sub(0).dim(), W.sub(1).dim(), W.sub(2).dim(), W.sub(3).dim()]

    def boundary(x, on_boundary):
        return on_boundary

    FSpaces = [VelocityF, PressureF, MagneticF, LagrangeF]

    DimSave[xx-1, :] = np.array(dim)

    kappa = 1.0
    Mu_m = 1.0
    MU = 1.0

    N = FacetNormal(mesh)

    IterType = 'Full'

    params = [kappa, Mu_m, MU]
    n = FacetNormal(mesh)
    # u0 = Expression(("1.0", "0.0"), degree=4)
    # p0 = Expression(("1.0"), degree=4)
    # b0 = Expression(("1.0", "0.0"), degree=4)
    # r0 = Expression(("1.0"), degree=4)

    # u_k = interpolate(u0, VelocityF)
    # b_k = interpolate(b0, MagneticF)

    # F_NS = Expression(("1.0", "0.0"), degree=4)
    # F_M = Expression(("1.0", "0.0"), degree=4)
    u0, p0, b0, r0, Laplacian, Advection, gradPres, CurlCurl, gradR, NS_Couple, M_Couple = ExactSol.MHD2D(
        4, 1)

    MO.PrintStr("Seting up initial guess matricies", 2, "=", "\n\n", "\n")
    BCtime = time.time()
    BC = MHDsetup.BoundaryIndices(mesh)
    MO.StrTimePrint("BC index function, time: ", time.time()-BCtime)
    Hiptmairtol = 1e-6
    HiptmairMatrices = PrecondSetup.MagneticSetup(
        mesh, Magnetic, Lagrange, b0, r0, Hiptmairtol, params)
    [G, P, kspVector, kspScalar, kspCGScalar, diag, CurlCurlShift] =  HiptmairMatrices
    MO.PrintStr("Setting up MHD initial guess", 5, "+", "\n\n", "\n\n")

    F_NS = -MU * Laplacian + Advection + gradPres - kappa * NS_Couple
    if kappa == 0.0:
        F_M = Mu_m*CurlCurl + gradR - kappa*M_Couple
    else:
        F_M = Mu_m*kappa*CurlCurl + gradR - kappa*M_Couple

    u_k, p_k = HartmanChannel.Stokes(
        Velocity, Pressure, F_NS, u0, 1, params, mesh)
    b_k, r_k = HartmanChannel.Maxwell(
        Magnetic, Lagrange, F_M, b0, r0, params, mesh, HiptmairMatrices, Hiptmairtol)

    bcu = DirichletBC(VelocityF, Expression(
        ("0.0", "0.0"), degree=4), boundary)
    bcp = DirichletBC(PressureF, Expression(("0.0"), degree=4), boundary)

    bcb = DirichletBC(MagneticF, Expression(
        ("0.0", "0.0"), degree=4), boundary)
    bcr = DirichletBC(LagrangeF, Expression(("0.0"), degree=4), boundary)

    bcu = np.array(bcu.get_boundary_values().keys())
    bcp = np.array(bcp.get_boundary_values().keys())
    bcb = np.array(bcb.get_boundary_values().keys())
    bcr = np.array(bcr.get_boundary_values().keys())

    scipy.io.savemat(
        "Matrix/bcu_"+str(int(level[xx-1][0]))+".mat", mdict={"bcu": bcu})
    scipy.io.savemat(
        "Matrix/bcp_"+str(int(level[xx-1][0]))+".mat", mdict={"bcp": bcp})
    scipy.io.savemat(
        "Matrix/bcb_"+str(int(level[xx-1][0]))+".mat", mdict={"bcb": bcb})
    scipy.io.savemat(
        "Matrix/bcr_"+str(int(level[xx-1][0]))+".mat", mdict={"bcr": bcr})
    scipy.io.savemat("Matrix/dim_"+str(int(level[xx-1][0]))+".mat", mdict={"bcr": np.array(
        [VelocityF.dim(), PressureF.dim(), MagneticF.dim(), LagrangeF.dim()])})

    u = TrialFunction(VelocityF)
    v = TestFunction(VelocityF)

    p = TrialFunction(PressureF)
    q = TestFunction(PressureF)

    b = TrialFunction(MagneticF)
    c = TestFunction(MagneticF)

    r = TrialFunction(LagrangeF)
    s = TestFunction(LagrangeF)

    U = Function(W)
    U.vector()[:] = 1.
    u_k, p_k, b_k, r_k = split(U)

    M = assemble(inner(curl(b), curl(c))*dx)
    M = as_backend_type(M).mat()
    savePETScMat(M, "Matrix/M_"+str(int(level[xx-1][0]))+".mat", "M")

    W = assemble(inner(v, u)*dx)
    W = as_backend_type(W).mat()
    savePETScMat(W, "Matrix/W_"+str(int(level[xx-1][0]))+".mat", "W")

    # vecL = assemble(inner(grad(u), grad(r))*dx)
    # vecL = as_backend_type(vecL).mat()
    # savePETScMat(vecL, "Matrix/vecL_"+str(int(level[xx-1][0]))+".mat", "vecL")

    L = assemble(inner(grad(r), grad(s))*dx)
    L = as_backend_type(L).mat()
    savePETScMat(L, "Matrix/L_"+str(int(level[xx-1][0]))+".mat", "L")

    X = assemble(inner(b, c)*dx)
    X = as_backend_type(X).mat()
    savePETScMat(X, "Matrix/X_"+str(int(level[xx-1][0]))+".mat", "X")

    Qp = assemble(inner(p, q)*dx)
    Qp = as_backend_type(Qp).mat()
    savePETScMat(Qp, "Matrix/Qp_"+str(int(level[xx-1][0]))+".mat", "Qp")


    D = assemble(inner(c, grad(r))*dx)
    D = as_backend_type(D).mat()
    savePETScMat(D, "Matrix/D_"+str(int(level[xx-1][0]))+".mat", "D")

    F = assemble(inner(grad(v), grad(u))*dx + inner((grad(u)*u_k), v)*dx +
                 (1./2)*div(u_k)*inner(u, v)*dx - (1./2)*inner(u_k, n)*inner(u, v)*ds)
    F = as_backend_type(F).mat()
    savePETScMat(F, "Matrix/F_"+str(int(level[xx-1][0]))+".mat", "F")

    O = assemble(inner((grad(u)*u_k), v)*dx +
                 (1./2)*div(u_k)*inner(u, v)*dx - (1./2)*inner(u_k, n)*inner(u, v)*ds)
    O = as_backend_type(O).mat()
    savePETScMat(O, "Matrix/O_"+str(int(level[xx-1][0]))+".mat", "O")

    A = assemble(inner(grad(v), grad(u))*dx)
    A = as_backend_type(A).mat()
    savePETScMat(A, "Matrix/A_"+str(int(level[xx-1][0]))+".mat", "A")

    XX = assemble(inner((v), (u))*dx)
    XX = as_backend_type(XX).mat()
    savePETScMat(XX, "Matrix/XX_"+str(int(level[xx-1][0]))+".mat", "XX")

    B = assemble(-div(v)*p*dx)
    B = as_backend_type(B).mat()
    savePETScMat(B, "Matrix/B_"+str(int(level[xx-1][0]))+".mat", "B")

    C = assemble((u[0]*b_k[1]-u[1]*b_k[0])*curl(c)*dx)
    C = as_backend_type(C).mat()
    savePETScMat(C, "Matrix/C_"+str(int(level[xx-1][0]))+".mat", "C")

    Ftilde = assemble(inner((grad(u_k)*u), v)*dx + (1./2)*div(u) *
                      inner(u_k, v)*dx - (1./2)*inner(u, n)*inner(u_k, v)*ds)
    Ftilde = as_backend_type(Ftilde).mat()
    savePETScMat(Ftilde, "Matrix/Ftilde_" +
                 str(int(level[xx-1][0]))+".mat", "Ftilde")

    Mtilde = assemble(-(u_k[0]*b[1]-u_k[1]*b[0])*curl(c)*dx)
    Mtilde = as_backend_type(Mtilde).mat()
    savePETScMat(Mtilde, "Matrix/Mtilde_" +
                 str(int(level[xx-1][0]))+".mat", "Mtilde")

    Ctilde = assemble((v[0]*b[1]-v[1]*b[0])*curl(b_k)*dx)
    Ctilde = as_backend_type(Ctilde).mat()
    savePETScMat(Ctilde, "Matrix/Ctilde_" +
                 str(int(level[xx-1][0]))+".mat", "Ctilde")



    savePETScMat(G, "Matrix/G_"+str(int(level[xx-1][0]))+".mat", "G")



# G = load(?strcat('Matrix/G_',num2str(level)));
# G = G.('G');