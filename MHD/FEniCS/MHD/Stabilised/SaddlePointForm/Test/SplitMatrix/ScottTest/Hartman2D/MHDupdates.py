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
m = 5

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

for xx in xrange(1,m):
    print xx
    level[xx-1] = xx + 3
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
    W = FunctionSpace(mesh, MixedElement([Velocity, Pressure, Magnetic,Lagrange]))

    Velocitydim[xx-1] = W.sub(0).dim()
    Pressuredim[xx-1] = W.sub(1).dim()
    Magneticdim[xx-1] = W.sub(2).dim()
    Lagrangedim[xx-1] = W.sub(3).dim()
    Wdim[xx-1] = W.dim()

    print "\n\nW:  ",Wdim[xx-1],"Velocity:  ",Velocitydim[xx-1],"Pressure:  ",Pressuredim[xx-1],"Magnetic:  ",Magneticdim[xx-1],"Lagrange:  ",Lagrangedim[xx-1],"\n\n"

    dim = [W.sub(0).dim(), W.sub(1).dim(), W.sub(2).dim(), W.sub(3).dim()]

    def boundary(x, on_boundary):
        return on_boundary

    FSpaces = [VelocityF,PressureF,MagneticF,LagrangeF]
    DimSave[xx-1,:] = np.array(dim)

    kappa = 1.0
    Mu_m = 10.0
    MU = 1.0

    N = FacetNormal(mesh)

    IterType = 'Full'

    params = [kappa,Mu_m,MU]
    n = FacetNormal(mesh)
    u0, p0,b0, r0, Laplacian, Advection, gradPres,CurlCurl, gradR, NS_Couple, M_Couple = ExactSol.MHD2D(4, 1)

    MO.PrintStr("Seting up initial guess matricies",2,"=","\n\n","\n")
    BCtime = time.time()
    BC = MHDsetup.BoundaryIndices(mesh)
    MO.StrTimePrint("BC index function, time: ", time.time()-BCtime)
    Hiptmairtol = 1e-6
    HiptmairMatrices = PrecondSetup.MagneticSetup(mesh, Magnetic, Lagrange, b0, r0, Hiptmairtol, params)

    MO.PrintStr("Setting up MHD initial guess",5,"+","\n\n","\n\n")

    F_NS = -MU*Laplacian + Advection + gradPres - kappa*NS_Couple
    if kappa == 0.0:
        F_M = Mu_m*CurlCurl + gradR - kappa*M_Couple
    else:
        F_M = Mu_m*kappa*CurlCurl + gradR - kappa*M_Couple

    u_k, p_k = HartmanChannel.Stokes(Velocity, Pressure, F_NS, u0, 1, params, mesh)
    b_k, r_k = HartmanChannel.Maxwell(Magnetic, Lagrange, F_M, b0, r0, params, mesh, HiptmairMatrices, Hiptmairtol)


    (u, p, b, r) = TrialFunctions(W)
    (v, q, c, s) = TestFunctions(W)
    if kappa == 0.0:
        m11 = params[1]*inner(curl(b),curl(c))*dx
    else:
        m11 = params[1]*params[0]*inner(curl(b),curl(c))*dx
    m21 = inner(c,grad(r))*dx
    m12 = inner(b,grad(s))*dx

    a11 = params[2]*inner(grad(v), grad(u))*dx + inner((grad(u)*u_k),v)*dx + (1./2)*div(u_k)*inner(u,v)*dx - (1./2)*inner(u_k,n)*inner(u,v)*ds
    a12 = -div(v)*p*dx
    a21 = -div(u)*q*dx

    CoupleT = params[0]*(v[0]*b_k[1]-v[1]*b_k[0])*curl(b)*dx
    Couple = -params[0]*(u[0]*b_k[1]-u[1]*b_k[0])*curl(c)*dx

    a = m11 + m12 + m21 + a11 + a21 + a12 + Couple + CoupleT

    if kappa == 0.0:
        m11 = params[1]*inner(curl(b_k),curl(c))*dx
    else:
        m11 = params[1]*params[0]*inner(curl(b_k),curl(c))*dx
    m21 = inner(c,grad(r_k))*dx
    m12 = inner(b_k,grad(s))*dx

    a11 = params[2]*inner(grad(v), grad(u_k))*dx + inner((grad(u_k)*u_k),v)*dx + (1./2)*div(u_k)*inner(u_k,v)*dx - (1./2)*inner(u_k,n)*inner(u_k,v)*ds
    a12 = -div(v)*p_k*dx
    a21 = -div(u_k)*q*dx

    CoupleT = params[0]*(v[0]*b_k[1]-v[1]*b_k[0])*curl(b_k)*dx
    Couple = -params[0]*(u_k[0]*b_k[1]-u_k[1]*b_k[0])*curl(c)*dx

    Lns  = inner(v, F_NS)*dx
    Lmaxwell  = inner(c, F_M)*dx

    L = Lns + Lmaxwell - (m11 + m12 + m21 + a11 + a21 + a12 + Couple + CoupleT)
    x = Iter.u_prev(u_k,p_k,b_k,r_k)

    KSPlinearfluids, MatrixLinearFluids = PrecondSetup.FluidLinearSetup(PressureF, MU, mesh)
    kspFp, Fp = PrecondSetup.FluidNonLinearSetup(PressureF, MU, u_k, mesh)

    IS = MO.IndexSet(W, 'Blocks')

    ones = Function(PressureF)
    ones.vector()[:]=(0*ones.vector().array()+1)
    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-6         # tolerance
    iter = 0            # iteration counter
    maxiter = 20       # max no of iterations allowed
    SolutionTime = 0
    outer = 0
    # parameters['linear_algebra_backend'] = 'uBLAS'

    u_is = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs())
    p_is = PETSc.IS().createGeneral(W.sub(1).dofmap().dofs())
    b_is = PETSc.IS().createGeneral(W.sub(2).dofmap().dofs())
    r_is = PETSc.IS().createGeneral(W.sub(3).dofmap().dofs())
    NS_is = PETSc.IS().createGeneral(range(VelocityF.dim()+PressureF.dim()))
    M_is = PETSc.IS().createGeneral(range(VelocityF.dim()+PressureF.dim(),W.dim()))
    bcu = DirichletBC(W.sub(0), Expression(("0.0", "0.0"), degree=4), boundary)
    bcb = DirichletBC(W.sub(2), Expression(("0.0", "0.0"), degree=4), boundary)
    bcr = DirichletBC(W.sub(3), Expression(("0.0"), degree=4), boundary)
    bcs = [bcu, bcb, bcr]
    OuterTol = 1e-4
    InnerTol = 1e-4
    NSits = 0
    Mits = 0
    TotalStart = time.time()
    SolutionTime = 0
    errors = np.array([])
    bcu1 = DirichletBC(VelocityF,Expression(("0.0","0.0"), degree=4), boundary)
    U = x
    while eps > tol and iter < maxiter:
        iter += 1
        MO.PrintStr("Iter "+str(iter),7,"=","\n\n","\n\n")

        A, b = assemble_system(a, L, bcs)
        A, b = CP.Assemble(A,b)
        u = x.duplicate()

        print "                               Max rhs = ",np.max(b.array)

        kspFp, Fp = PrecondSetup.FluidNonLinearSetup(PressureF, MU, u_k, mesh)
        ShiftedMass = A.getSubMatrix(u_is, u_is)
        kspF = NSprecondSetup.LSCKSPnonlinear(ShiftedMass)
        Options = 'p4'
        norm = (b-A*U).norm()
        residual = b.norm()
        stime = time.time()
        u, mits,nsits = S.solve(A,b,u,params,W,'Directi',IterType,OuterTol,InnerTol,HiptmairMatrices,Hiptmairtol,KSPlinearfluids, Fp,kspF)

        U = u
        Soltime = time.time() - stime
        MO.StrTimePrint("MHD solve, time: ", Soltime)
        Mits += mits
        NSits += mits
        SolutionTime += Soltime

        u1 = Function(VelocityF)
        p1 = Function(PressureF)
        b1 = Function(MagneticF)
        r1 = Function(LagrangeF)

        u1.vector()[:] = u.getSubVector(u_is).array
        p1.vector()[:] = u.getSubVector(p_is).array
        b1.vector()[:] = u.getSubVector(b_is).array
        r1.vector()[:] = u.getSubVector(r_is).array
        p1.vector()[:] += - assemble(p1*dx)/assemble(ones*dx)
        diff = np.concatenate((u1.vector().array(),p1.vector().array(),b1.vector().array(),r1.vector().array()), axis=0)
        u1.vector()[:] += u_k.vector().array()
        p1.vector()[:] += p_k.vector().array()
        b1.vector()[:] += b_k.vector().array()
        r1.vector()[:] += r_k.vector().array()

        u_k.assign(u1)
        p_k.assign(p1)
        b_k.assign(b1)
        r_k.assign(r1)

        uOld = np.concatenate((u_k.vector().array(),p_k.vector().array(),b_k.vector().array(),r_k.vector().array()), axis=0)
        x = IO.arrayToVec(uOld)
        w = Function(W)
        w.vector()[:] = diff
        

        print np.linalg.norm(diff)/x.norm(), residual, sqrt(assemble(inner(w, w)*dx))
        eps = min(np.linalg.norm(diff)/x.norm(), residual)

        print '            ssss           ', eps


    SolTime[xx-1] = SolutionTime/iter
    NSave[xx-1] = (float(NSits)/iter)
    Mave[xx-1] = (float(Mits)/iter)
    iterations[xx-1] = iter
    TotalTime[xx-1] = time.time() - TotalStart

    XX= np.concatenate((u_k.vector().array(),p_k.vector().array(),b_k.vector().array(),r_k.vector().array()), axis=0)

    ExactSolution = [u0,p0,b0,r0]
    errL2u[xx-1], errH1u[xx-1], errL2p[xx-1], errL2b[xx-1], errCurlb[xx-1], errL2r[xx-1], errH1r[xx-1] = Iter.Errors(XX,mesh,FSpaces,ExactSolution,order,dim, "CG")
    print float(Wdim[xx-1][0])/Wdim[xx-2][0]


    if xx > 1:

      l2uorder[xx-1] = np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1])/np.log2((float(Velocitydim[xx-1][0])/Velocitydim[xx-2][0])**(1./2)))
      H1uorder[xx-1] =  np.abs(np.log2(errH1u[xx-2]/errH1u[xx-1])/np.log2((float(Velocitydim[xx-1][0])/Velocitydim[xx-2][0])**(1./2)))

      l2porder[xx-1] =  np.abs(np.log2(errL2p[xx-2]/errL2p[xx-1])/np.log2((float(Pressuredim[xx-1][0])/Pressuredim[xx-2][0])**(1./2)))

      l2border[xx-1] =  np.abs(np.log2(errL2b[xx-2]/errL2b[xx-1])/np.log2((float(Magneticdim[xx-1][0])/Magneticdim[xx-2][0])**(1./2)))
      Curlborder[xx-1] =  np.abs(np.log2(errCurlb[xx-2]/errCurlb[xx-1])/np.log2((float(Magneticdim[xx-1][0])/Magneticdim[xx-2][0])**(1./2)))

      l2rorder[xx-1] =  np.abs(np.log2(errL2r[xx-2]/errL2r[xx-1])/np.log2((float(Lagrangedim[xx-1][0])/Lagrangedim[xx-2][0])**(1./2)))
      H1rorder[xx-1] =  np.abs(np.log2(errH1r[xx-2]/errH1r[xx-1])/np.log2((float(Lagrangedim[xx-1][0])/Lagrangedim[xx-2][0])**(1./2)))



import pandas as pd



LatexTitles = ["l","DoFu","Dofp","V-L2","L2-order","V-H1","H1-order","P-L2","PL2-order"]
LatexValues = np.concatenate((level,Velocitydim,Pressuredim,errL2u,l2uorder,errH1u,H1uorder,errL2p,l2porder), axis=1)
LatexTable = pd.DataFrame(LatexValues, columns = LatexTitles)
pd.set_option('precision',3)
LatexTable = MO.PandasFormat(LatexTable,"V-L2","%2.4e")
LatexTable = MO.PandasFormat(LatexTable,'V-H1',"%2.4e")
LatexTable = MO.PandasFormat(LatexTable,"H1-order","%1.2f")
LatexTable = MO.PandasFormat(LatexTable,'L2-order',"%1.2f")
LatexTable = MO.PandasFormat(LatexTable,"P-L2","%2.4e")
LatexTable = MO.PandasFormat(LatexTable,'PL2-order',"%1.2f")
print LatexTable.to_latex()


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


print "\n\n   Iteration table"
if IterType == "Full":
    IterTitles = ["l","DoF","AV solve Time","Total picard time","picard iterations","Av Outer its","Av Inner its",]
else:
    IterTitles = ["l","DoF","AV solve Time","Total picard time","picard iterations","Av NS iters","Av M iters"]
IterValues = np.concatenate((level,Wdim,SolTime,TotalTime,iterations,Mave,NSave),axis=1)
IterTable= pd.DataFrame(IterValues, columns = IterTitles)
if IterType == "Full":
    IterTable = MO.PandasFormat(IterTable,'Av Outer its',"%2.1f")
    IterTable = MO.PandasFormat(IterTable,'Av Inner its',"%2.1f")
else:
    IterTable = MO.PandasFormat(IterTable,'Av NS iters',"%2.1f")
    IterTable = MO.PandasFormat(IterTable,'Av M iters',"%2.1f")
print IterTable.to_latex()
MO.StoreMatrix(DimSave, "dim")

#file = File("u_k.pvd")
#file << u_k
#
#file = File("p_k.pvd")
#file << p_k
#
#file = File("b_k.pvd")
#file << b_k
#
#file = File("r_k.pvd")
#file << r_k
#
#file = File("u0.pvd")
#file << interpolate(u0, VelocityF)
#
#file = File("p0.pvd")
#file << interpolate(p0, PressureF)
#
#file = File("b0.pvd")
#file << interpolate(b0, MagneticF)
#
#file = File("r0.pvd")
#file << interpolate(r0, LagrangeF)
#
#file = File("uError.pvd")
#error = Function(VelocityF)
#error.vector()[:] =  u_k.vector().array()-interpolate(u0, VelocityF).vector().array()
#file << error
#
#file = File("pError.pvd")
#error = Function(PressureF)
#error.vector()[:] =  p_k.vector().array()-interpolate(p0, PressureF).vector().array()
#file << error
#
#file = File("bError.pvd")
#error = Function(MagneticF)
#error.vector()[:] =  b_k.vector().array()-interpolate(b0, MagneticF).vector().array()
#file << error
#
#file = File("rError.pvd")
#error = Function(LagrangeF)
#error.vector()[:] =  r_k.vector().array()-interpolate(r0, LagrangeF).vector().array()
#file << error
#
interactive()

