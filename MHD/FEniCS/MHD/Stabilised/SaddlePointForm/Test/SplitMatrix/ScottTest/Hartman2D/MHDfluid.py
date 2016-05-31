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
import scipy.sparse as sp
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
import PCD
# import matplotlib.pyplot as plt
#@profile
m = 7


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
DimSave = np.zeros((m-1,4))
 2

dim = 2
ShowResultPlots = 'yes'
split = 'Linear'
MU[0]= 1e0
for xx in xrange(1,m):
    print xx
    level[xx-1] = xx + 0
    nn = 2**(level[xx-1])

    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn/2
    L = 10.
    y0 = 2.
    z0 = 1.
    mesh, boundaries, domains = HartmanChannel.Domain(nn)

    parameters['form_compiler']['quadrature_degree'] = -1
    order = 2
    parameters['reorder_dofs_serial'] = False
    Velocity = VectorFunctionSpace(mesh, "CG", order)
    Pressure = FunctionSpace(mesh, "CG", order-1)
    Magnetic = FunctionSpace(mesh, "N1curl", order-1)
    Lagrange = FunctionSpace(mesh, "CG", order-1)
    W = MixedFunctionSpace([Velocity, Pressure, Magnetic,Lagrange])

    Velocitydim[xx-1] = Velocity.dim()
    Pressuredim[xx-1] = Pressure.dim()
    Magneticdim[xx-1] = Magnetic.dim()
    Lagrangedim[xx-1] = Lagrange.dim()
    Wdim[xx-1] = W.dim()
    print "\n\nW:  ",Wdim[xx-1],"Velocity:  ",Velocitydim[xx-1],"Pressure:  ",Pressuredim[xx-1],"Magnetic:  ",Magneticdim[xx-1],"Lagrange:  ",Lagrangedim[xx-1],"\n\n"
    dim = [Velocity.dim(), Pressure.dim(), Magnetic.dim(), Lagrange.dim()]

    def boundary(x, on_boundary):
        return on_boundary

    FSpaces = [Velocity,Pressure,Magnetic,Lagrange]
    DimSave[xx-1,:] = np.array(dim)

    kappa = 1.0
    Mu_m = 1.0
    MU = 1.0

    N = FacetNormal(mesh)

    IterType = 'Full'
    Split = "No"
    Saddle = "No"
    Stokes = "No"
    SetupType = 'python-class'

    params = [kappa,Mu_m,MU]
    n = FacetNormal(mesh)
    trunc = 4
    u0, p0, b0, r0, pN, Laplacian, Advection, gradPres, NScouple, CurlCurl, gradLagr, Mcouple = HartmanChannel.ExactSolution(mesh, params)

    MO.PrintStr("Seting up initial guess matricies",2,"=","\n\n","\n")
    BCtime = time.time()
    BC = MHDsetup.BoundaryIndices(mesh)
    MO.StrTimePrint("BC index function, time: ", time.time()-BCtime)
    Hiptmairtol = 1e-6
    HiptmairMatrices = PrecondSetup.MagneticSetup(Magnetic, Lagrange, b0, r0, Hiptmairtol, params)

    MO.PrintStr("Setting up MHD initial guess",5,"+","\n\n","\n\n")

    F_NS = -MU*Laplacian + Advection + gradPres - kappa*NScouple
    F_M = Mu_m*kappa*CurlCurl + gradLagr - kappa*Mcouple
    u_k, p_k = HartmanChannel.Stokes(Velocity, Pressure, F_NS, u0, pN, params, mesh, boundaries, domains)
    b_k, r_k = HartmanChannel.Maxwell(Magnetic, Lagrange, F_M, b0, r0, params, mesh)

    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    (u, p, b, r) = TrialFunctions(W)
    (v, q, c, s) = TestFunctions(W)

    m11 = params[1]*params[0]*inner(curl(b),curl(c))*dx(0)
    m21 = inner(c,grad(r))*dx(0)
    m12 = inner(b,grad(s))*dx(0)

    a11 = params[2]*inner(grad(v), grad(u))*dx(0) + inner((grad(u)*u_k),v)*dx(0) + (1./2)*div(u_k)*inner(u,v)*dx(0) - (1./2)*inner(u_k,n)*inner(u,v)*ds(0)
    a12 = -div(v)*p*dx(0)
    a21 = -div(u)*q*dx(0)

    CoupleT = params[0]*(v[0]*b_k[1]-v[1]*b_k[0])*curl(b)*dx(0)
    Couple = -params[0]*(u[0]*b_k[1]-u[1]*b_k[0])*curl(c)*dx(0)

    a = m11 + m12 + m21 + a11 + a21 + a12 + Couple + CoupleT

    Lns  = inner(v, F_NS)*dx(0) - inner(pN*n,v)*ds(2)
    Lmaxwell  = inner(c, F_M)*dx(0)

    m11 = params[1]*params[0]*inner(curl(b_k),curl(c))*dx(0)
    m21 = inner(c,grad(r_k))*dx(0)
    m12 = inner(b_k,grad(s))*dx(0)

    a11 = params[2]*inner(grad(v), grad(u_k))*dx(0) + inner((grad(u_k)*u_k),v)*dx(0) + (1./2)*div(u_k)*inner(u_k,v)*dx(0) - (1./2)*inner(u_k,n)*inner(u_k,v)*ds(0)
    a12 = -div(v)*p_k*dx(0)
    a21 = -div(u_k)*q*dx(0)

    CoupleT = params[0]*(v[0]*b_k[1]-v[1]*b_k[0])*curl(b_k)*dx(0)
    Couple = -params[0]*(u_k[0]*b_k[1]-u_k[1]*b_k[0])*curl(c)*dx(0)

    L = Lns + Lmaxwell - (m11 + m12 + m21 + a11 + a21 + a12+ Couple + CoupleT)

    ones = Function(Pressure)
    ones.vector()[:]=(0*ones.vector().array()+1)
    pConst = - assemble(p_k*dx(0))/assemble(ones*dx(0))
    p_k.vector()[:] += - assemble(p_k*dx(0))/assemble(ones*dx(0))
    x = Iter.u_prev(u_k,p_k,b_k,r_k)

    KSPlinearfluids, MatrixLinearFluids = PrecondSetup.FluidLinearSetup(Pressure, MU, mesh, boundaries, domains)
    kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k, mesh, boundaries, domains)

    IS = MO.IndexSet(W, 'Blocks')

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-3     # tolerance
    iter = 0            # iteration counter
    maxiter = 20       # max no of iterations allowed
    SolutionTime = 0
    outer = 0
    # parameters['linear_algebra_backend'] = 'uBLAS'

    u_is = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs())
    b_is = PETSc.IS().createGeneral(W.sub(2).dofmap().dofs())
    NS_is = PETSc.IS().createGeneral(range(Velocity.dim()+Pressure.dim()))
    M_is = PETSc.IS().createGeneral(range(Velocity.dim()+Pressure.dim(),W.dim()))

    OuterTol = 1e-5
    InnerTol = 1e-5
    NSits = 0
    Mits = 0
    TotalStart =time.time()
    SolutionTime = 0

    while eps > tol  and iter < maxiter:
        iter += 1
        MO.PrintStr("Iter "+str(iter),7,"=","\n\n","\n\n")

        bcu = DirichletBC(W.sub(0),Expression(("0.0","0.0")), boundaries, 1)
        # bcu = DirichletBC(W.sub(0),Expression(("0.0","0.0")), boundary)
        bcb = DirichletBC(W.sub(2),Expression(("0.0","0.0")), boundary)
        bcr = DirichletBC(W.sub(3),Expression("0.0"), boundary)
        bcs = [bcu,bcb,bcr]

        A, b = assemble_system(a, L, bcs)
        A, b = CP.Assemble(A,b)
        u = b.duplicate()
        print "                               Max rhs = ",np.max(b.array)
        # A = IO.matToSparse(A)
        # b = b.array
        # j = scipy.sparse.csgraph.reverse_cuthill_mckee(A)
        # A = A[j,j]
        # b = b[j]
        # print j
        kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k, mesh, boundaries, domains)
        b_t = TrialFunction(Velocity)
        c_t = TestFunction(Velocity)
        n = FacetNormal(mesh)
        mat =  as_matrix([[b_k[1]*b_k[1],-b_k[1]*b_k[0]],[-b_k[1]*b_k[0],b_k[0]*b_k[0]]])
        aa = params[2]*inner(grad(b_t), grad(c_t))*dx(W.mesh()) + inner((grad(b_t)*u_k),c_t)*dx(W.mesh()) +(1./2)*div(u_k)*inner(c_t,b_t)*dx(W.mesh()) - (1./2)*inner(u_k,n)*inner(c_t,b_t)*ds(W.mesh())+kappa/Mu_m*inner(mat*b_t,c_t)*dx(W.mesh())
        ShiftedMass = assemble(aa)
        bcu.apply(ShiftedMass)
        ShiftedMass = CP.Assemble(ShiftedMass)
        kspF = NSprecondSetup.LSCKSPnonlinear(ShiftedMass)
        Options = 'p4'
        PCD.check(MU, u_k, p_k, mesh, boundaries, domains)
        stime = time.time()
        # MO.StoreMatrix(PETSc2Scipy(A), "A_"+str(int(level[xx-1][0])))
        # MO.StoreMatrix(b.array, "b_"+str(int(level[xx-1][0])))

        # u1, mits,nsits = S.solve(A.getSubMatrix(NS_is,NS_is),b.getSubVector(NS_is), u.getSubVector(NS_is),params,W,'Direct',IterType,OuterTol,InnerTol,1,1,1, 1,1)
        # u2, mits,nsits = S.solve(A.getSubMatrix(M_is,M_is),b.getSubVector(M_is), u.getSubVector(M_is),params,W,'Direct',IterType,OuterTol,InnerTol,1,1,1, 1,1)
        # u = IO.arrayToVec(np.concatenate((u1.array,u2.array), axis=0))
        # u, mits,nsits = S.solve(A, b, u, params, W, 'Direct', IterType, OuterTol, InnerTol, 1, 1, 1, 1, 1)
        # u = scipy.sparse.linalg.spsolve(A,b)
        u, mits,nsits = S.solve(A,b,u,params,W,'Directs',IterType,OuterTol,InnerTol,HiptmairMatrices,Hiptmairtol,KSPlinearfluids, Fp,kspF)

        Soltime = time.time() - stime
        MO.StrTimePrint("MHD solve, time: ", Soltime)
        Mits += mits
        NSits += nsits
        SolutionTime += Soltime
        # u = IO.arrayToVec(  u)
        u1, p1, b1, r1, eps = Iter.PicardToleranceDecouple(u,x,FSpaces,dim,"2",iter)
        p1.vector()[:] += - assemble(p1*dx(0))/assemble(ones*dx(0))
        u_k.assign(u1)
        p_k.assign(p1)
        b_k.assign(b1)
        r_k.assign(r1)
        uOld = np.concatenate((u_k.vector().array(),p_k.vector().array(),b_k.vector().array(),r_k.vector().array()), axis=0)
        x = IO.arrayToVec(uOld)
    # iter = 1

    SolTime[xx-1] = SolutionTime/iter
    NSave[xx-1] = (float(NSits)/iter)
    Mave[xx-1] = (float(Mits)/iter)
    iterations[xx-1] = iter
    TotalTime[xx-1] = time.time() - TotalStart

    XX= np.concatenate((u_k.vector().array(),p_k.vector().array(),b_k.vector().array(),r_k.vector().array()), axis=0)
    dim = [Velocity.dim(), Pressure.dim(), Magnetic.dim(),Lagrange.dim()]

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




import pandas as pd



# p = plot(u_k)
# p.write_png()
# p = plot(p_k)
# p.write_png()
# # p = plot(b_k)
# # p.write_png()
# # p = plot(r_k)
# # p.write_png()
# p = plot(interpolate(u0,Velocity))
# p.write_png()
# p = plot(interpolate(p0,Pressure))
# p.write_png()
# # p = plot(interpolate(b0,Magnetic))
# # p.write_png()
# # p = plot(interpolate(r0,Lagrange))
# # p.write_png()
# sss

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
print IterTable
MO.StoreMatrix(DimSave, "dim")
# print " \n  Outer Tol:  ",OuterTol, "Inner Tol:   ", InnerTol

# tableName = "2d_Lshaped_nu="+str(MU)+"_nu_m="+str(Mu_m)+"_kappa="+str(kappa)+"_l="+str(np.min(level))+"-"+str(np.max(level))+"Approx.tex"
# IterTable.to_latex(tableName)

# # # if (ShowResultPlots == 'yes'):

#    plot(interpolate(u0,Velocity))
#
# u = plot(interpolate(u0,Velocity))
# p = plot(interpolate(pN2,Pressure))
# b = plot(interpolate(b0,Magnetic))
# u.write_png()
# p.write_png()
# b.write_png()

# u = plot(u_k)
# p = plot(p_k)
# b = plot(b_k)
# u.write_png()
# p.write_png()
# b.write_png()

#
#    plot(interpolate(p0,Pressure))
#
#    plot(interpolate(b0,Magnetic))
#
#    plot(r_k)
#    plot(interpolate(r0,Lagrange))
#
#    interactive()

interactive()
