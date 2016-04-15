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
import ExactSol
import Solver as S
import MHDmatrixPrecondSetup as PrecondSetup
import NSprecondSetup
import MHDprec as MHDpreconditioner
import memory_profiler
import gc
import MHDmulti
import MHDmatrixSetup as MHDsetup
import Lshaped
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
uu0, ub0, pu0, pb0, bu0, bb0, ru0, Laplacian, Advection, gradPres, CurlCurl, gradR, NS_Couple, M_Couple = Lshaped.SolutionSetUp()
MU[0]= 1e0
for xx in xrange(1,m):
    print xx
    level[xx-1] = xx
    nn = 2**(level[xx-1])



    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn/2
    parameters["form_compiler"]["quadrature_degree"] = -1

    mesh, boundaries, domains = Lshaped.Domain(nn)

    order = 2
    parameters['reorder_dofs_serial'] = False
    Velocity = VectorFunctionSpace(mesh, "CG", order)
    Pressure = FunctionSpace(mesh, "CG", order-1)

    W = MixedFunctionSpace([Velocity, Pressure])

    Velocitydim[xx-1] = Velocity.dim()
    Pressuredim[xx-1] = Pressure.dim()

    Wdim[xx-1] = W.dim()
    print "\n\nW:  ",Wdim[xx-1],"Velocity:  ",Velocitydim[xx-1],"Pressure:  ",Pressuredim[xx-1],"\n\n"
    dim = [Velocity.dim(), Pressure.dim()]


    def boundary(x, on_boundary):
        return on_boundary


    FSpaces = [Velocity,Pressure]

    kappa = 1.0
    Mu_m =10.0
    MU = 1.0

    N = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    # g = inner(p0*N - MU*grad(u0)*N,v)*dx

    IterType = 'Full'
    Split = "No"
    Saddle = "No"
    Stokes = "No"
    SetupType = 'python-class'

    params = [kappa,Mu_m,MU]
    u0, p0, b0, r0, F_NS, F_M, F_MX, F_S, gradu0, Neumann, p0vec, bNone = Lshaped.SolutionMeshSetup(mesh, params, uu0, ub0, pu0, pb0, bu0, bb0, ru0, Laplacian, Advection, gradPres, CurlCurl, gradR, NS_Couple, M_Couple)
    F_M = Expression(("0.0","0.0"))
    F_S = Expression(("0.0","0.0"))
    n = FacetNormal(mesh)

    u_k, p_k = Lshaped.Stokes(Velocity, Pressure, F_S, u0, p0, Neumann, params, boundaries, domains)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    a11 = params[2]*inner(grad(v), grad(u))*dx + inner((grad(u)*u_k),v)*dx + (1./2)*div(u_k)*inner(u,v)*dx - (1./2)*inner(u_k,n)*inner(u,v)*ds
    a12 = -div(v)*p*dx
    a21 = -div(u)*q*dx

    a = a11 + a21 + a12


    Lns  = inner(v, F_NS)*dx #+ inner(Neumann,v)*ds(2)

    a11 = params[2]*inner(grad(v), grad(u_k))*dx + inner((grad(u_k)*u_k),v)*dx + (1./2)*div(u_k)*inner(u_k,v)*dx - (1./2)*inner(u_k,n)*inner(u_k,v)*ds
    a12 = -div(v)*p_k*dx
    a21 = -div(u_k)*q*dx

    L = Lns - ( a11 + a21 + a12 )



    MO.PrintStr("Setting up MHD initial guess",5,"+","\n\n","\n\n")

    ones = Function(Pressure)
    ones.vector()[:]=(0*ones.vector().array()+1)
    pConst = - assemble(p_k*dx)/assemble(ones*dx)
    p_k.vector()[:] += - assemble(p_k*dx)/assemble(ones*dx)
    x= np.concatenate((u_k.vector().array(),p_k.vector().array()), axis=0)

    KSPlinearfluids, MatrixLinearFluids = PrecondSetup.FluidLinearSetup(Pressure, MU)
    kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k)


    parameters['linear_algebra_backend'] = 'uBLAS'

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-4     # tolerance
    iter = 0            # iteration counter
    maxiter = 10       # max no of iterations allowed
    SolutionTime = 0
    outer = 0

    u_is = PETSc.IS().createGeneral(range(Velocity.dim()))
    NS_is = PETSc.IS().createGeneral(range(Velocity.dim()+Pressure.dim()))
    M_is = PETSc.IS().createGeneral(range(Velocity.dim()+Pressure.dim(),W.dim()))
    OuterTol = 1e-5
    InnerTol = 1e-5
    NSits =0
    Mits =0
    TotalStart =time.time()
    SolutionTime = 0
    while eps > tol  and iter < maxiter:
        iter += 1
        MO.PrintStr("Iter "+str(iter),7,"=","\n\n","\n\n")

        bcu = DirichletBC(W.sub(0),Expression(("0.0","0.0")), boundary)
        bcs = [bcu]

        A, b = assemble_system(a, L, bcs)

        A, b = CP.Assemble(A,b)
        u = b.duplicate()

        n = FacetNormal(mesh)
        b_t = TrialFunction(Velocity)
        c_t = TestFunction(Velocity)
        aa = params[2]*inner(grad(b_t), grad(c_t))*dx(W.mesh()) + inner((grad(b_t)*u_k),c_t)*dx(W.mesh()) +(1./2)*div(u_k)*inner(c_t,b_t)*dx(W.mesh()) - (1./2)*inner(u_k,n)*inner(c_t,b_t)*ds(W.mesh())
        ShiftedMass = assemble(aa)
        bcu.apply(ShiftedMass)
        ShiftedMass = CP.Assemble(ShiftedMass)
        kspF = NSprecondSetup.LSCKSPnonlinear(ShiftedMass)

        stime = time.time()

        u, mits,nsits = S.solve(A,b,u,params,W,'Directss',IterType,OuterTol,InnerTol,HiptmairMatrices,Hiptmairtol,KSPlinearfluids, Fp,kspF)
        Soltime = time.time()- stime
        MO.StrTimePrint("MHD solve, time: ", Soltime)
        Mits += mits
        NSits += nsits
        SolutionTime += Soltime

        u1, p1, b1, r1, eps= Iter.PicardToleranceDecouple(u,x,FSpaces,dim,"2",iter)
        p1.vector()[:] += - assemble(p1*dx)/assemble(ones*dx)
        u_k.assign(u1)
        p_k.assign(p1)
        b_k.assign(b1)
        r_k.assign(r1)
        uOld= np.concatenate((u_k.vector().array(),p_k.vector().array(),b_k.vector().array(),r_k.vector().array()), axis=0)
        x = IO.arrayToVec(uOld)


    SolTime[xx-1] = SolutionTime/iter
    NSave[xx-1] = (float(NSits)/iter)
    Mave[xx-1] = (float(Mits)/iter)
    iterations[xx-1] = iter
    TotalTime[xx-1] = time.time() - TotalStart

    XX= np.concatenate((u_k.vector().array(),p_k.vector().array(),b_k.vector().array(),r_k.vector().array()), axis=0)
    dim = [Velocity.dim(), Pressure.dim(), Magnetic.dim(),Lagrange.dim()]
    u0, p0, b0, r0, F_NS, F_M, F_MX, F_S, gradu0, Neumann, p0vec = Lshaped.Solution2(mesh, params)

    ExactSolution = [u0,p0,b0,r0]

    Vdim = dim[0]
    Pdim = dim[1]
    Mdim = dim[2]
    Rdim = dim[3]
    # k +=2
    VelocityE = VectorFunctionSpace(mesh,"CG",4)
    u = interpolate(ExactSolution[0],VelocityE)

    PressureE = FunctionSpace(mesh,FS,3)
    # parameters["form_compiler"]["quadrature_degree"] = 8
    # X = x.array()
    xu = X[0:Vdim]
    ua = Function(FSpaces[0])
    ua.vector()[:] = xu

    pp = X[Vdim:Vdim+Pdim]


    pa = Function(FSpaces[1])
    pa.vector()[:] = pp

    pend = assemble(pa*dx)

    ones = Function(FSpaces[1])
    ones.vector()[:]=(0*pp+1)
    pp = Function(FSpaces[1])
    pp.vector()[:] = pa.vector().array()- assemble(pa*dx)/assemble(ones*dx)

    pInterp = interpolate(ExactSolution[1],PressureE)
    pe = Function(PressureE)
    pe.vector()[:] = pInterp.vector().array()
    const = - assemble(pe*dx)/assemble(ones*dx)
    pe.vector()[:] = pe.vector()[:]+const

    ErrorU = Function(FSpaces[0])
    ErrorP = Function(FSpaces[1])

    ErrorU = u-ua
    ErrorP = pe-pp
    tic()
    errL2u= sqrt(abs(assemble(inner(ErrorU, ErrorU)*dx)))
    MO.StrTimePrint("Velocity L2 error, time: ", toc())
    tic()
    errH1u= sqrt(abs(assemble(inner(grad(ErrorU), grad(ErrorU))*dx)))
    MO.StrTimePrint("Velocity H1 error, time: ", toc())
    tic()
    errL2p= sqrt(abs(assemble(inner(ErrorP, ErrorP)*dx)))
    MO.StrTimePrint("Pressure L2 error, time: ", toc())

    print float(Wdim[xx-1][0])/Wdim[xx-2][0]
    if xx > 1:
       l2uorder[xx-1] =  np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1])/np.log2(sqrt(float(Velocitydim[xx-1][0])/Velocitydim[xx-2][0])))
       H1uorder[xx-1] =  np.abs(np.log2(errH1u[xx-2]/errH1u[xx-1])/np.log2(sqrt(float(Velocitydim[xx-1][0])/Velocitydim[xx-2][0])))

       l2porder[xx-1] =  np.abs(np.log2(errL2p[xx-2]/errL2p[xx-1])/np.log2(sqrt(float(Pressuredim[xx-1][0])/Pressuredim[xx-2][0])))





# import pandas as pd



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



import pandas as pd




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


interactive()
