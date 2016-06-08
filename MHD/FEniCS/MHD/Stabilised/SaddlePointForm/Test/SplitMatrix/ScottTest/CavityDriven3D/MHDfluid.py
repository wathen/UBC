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
import CavityDriven
#@profile
m = 2

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
    level[xx-1] = xx + 0
    nn = 2**(level[xx-1])



    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn/2
    parameters["form_compiler"]["quadrature_degree"] = -1
    # parameters = CP.ParameterSetup()
    # mesh = UnitSquareMesh(nn,nn)
    # domain = mshr.Rectangle(Point(0., 0.), Point(1., 2.)) + mshr.Rectangle(Point(1., 0.), Point(2., 1.))
    # mesh = mshr.generate_mesh(domain, nn)
    mesh, boundaries, domains = CavityDriven.Domain(nn)
    # set_log_level(WARNING)

    order = 2
    parameters['reorder_dofs_serial'] = False
    Velocity = VectorFunctionSpace(mesh, "CG", order)
    Pressure = FunctionSpace(mesh, "CG", order-1)
    Magnetic = FunctionSpace(mesh, "N1curl", order-1)
    Lagrange = FunctionSpace(mesh, "CG", order-1)
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


    FSpaces = [Velocity,Pressure,Magnetic,Lagrange]

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

    F_M = Expression(("0.0","0.0"))
    F_S = Expression(("0.0","0.0"))
    n = FacetNormal(mesh)
    class intial(Expression):
        def __init__(self, mesh):
            self.mesh = mesh
        def eval_cell(self, values, x, ufc_cell):
            values[0] = 1.0
            values[1] = 0
        def value_shape(self):
            return (2,)
    b0 = intial(mesh)
    u0 = intial(mesh)
    r0 = Expression(("0.0"))
    u_k, p_k = CavityDriven.Stokes(Velocity, Pressure, F_S, params, boundaries, domains)
    b_k, r_k = CavityDriven.Maxwell(Magnetic, Lagrange, F_M, params)

    (u, p, b, r) = TrialFunctions(W)
    (v, q, c, s) = TestFunctions(W)



    m11 = params[1]*params[0]*inner(curl(b),curl(c))*dx
    m21 = inner(c,grad(r))*dx
    m12 = inner(b,grad(s))*dx

    a11 = params[2]*inner(grad(v), grad(u))*dx + inner((grad(u)*u_k),v)*dx + (1./2)*div(u_k)*inner(u,v)*dx - (1./2)*inner(u_k,n)*inner(u,v)*ds
    a12 = -div(v)*p*dx
    a21 = -div(u)*q*dx

    CoupleT = params[0]*(v[0]*b_k[1]-v[1]*b_k[0])*curl(b)*dx
    Couple = -params[0]*(u[0]*b_k[1]-u[1]*b_k[0])*curl(c)*dx

    a = m11 + m12 + m21 + a11 + a21 + a12 + Couple + CoupleT


    Lns  = inner(v, F_S)*dx #+ inner(Neumann,v)*ds(2)
    Lmaxwell  = inner(c, F_M)*dx


    m11 = params[1]*params[0]*inner(curl(b_k),curl(c))*dx
    m21 = inner(c,grad(r_k))*dx
    m12 = inner(b_k,grad(s))*dx

    a11 = params[2]*inner(grad(v), grad(u_k))*dx + inner((grad(u_k)*u_k),v)*dx + (1./2)*div(u_k)*inner(u_k,v)*dx - (1./2)*inner(u_k,n)*inner(u_k,v)*ds
    a12 = -div(v)*p_k*dx
    a21 = -div(u_k)*q*dx
    CoupleT = params[0]*(v[0]*b_k[1]-v[1]*b_k[0])*curl(b_k)*dx
    Couple = -params[0]*(u_k[0]*b_k[1]-u_k[1]*b_k[0])*curl(c)*dx

    L = Lns + Lmaxwell - (m11 + m12 + m21 + a11 + a21 + a12 + Couple + CoupleT)


    MO.PrintStr("Seting up initial guess matricies",2,"=","\n\n","\n")
    BCtime = time.time()
    BC = MHDsetup.BoundaryIndices(mesh)
    MO.StrTimePrint("BC index function, time: ", time.time()-BCtime)
    Hiptmairtol = 1e-6
    HiptmairMatrices = PrecondSetup.MagneticSetup(Magnetic, Lagrange, b0, r0, Hiptmairtol, params)


    MO.PrintStr("Setting up MHD initial guess",5,"+","\n\n","\n\n")
    # u_k,p_k,b_k,r_k = common.InitialGuess(FSpaces,[u0,p0,b0,r0],[F_NS,F_M],params,HiptmairMatrices,1e-10,Neumann=None,options ="New")




    ones = Function(Pressure)
    ones.vector()[:]=(0*ones.vector().array()+1)
    # pConst = - assemble(p_k*dx)/assemble(ones*dx)
    # p_k.vector()[:] += - assemble(p_k*dx)/assemble(ones*dx)
    x = Iter.u_prev(u_k,p_k,b_k,r_k)

    KSPlinearfluids, MatrixLinearFluids = PrecondSetup.FluidLinearSetup(Pressure, MU)
    kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k)
    #plot(b_k)

    # ns,maxwell,CoupleTerm,Lmaxwell,Lns = forms.MHD2D(mesh, W,F_M,F_NS, u_k,b_k,params,IterType,"CG",Saddle,Stokes)
    # RHSform = forms.PicardRHS(mesh, W, u_k, p_k, b_k, r_k, params,"CG",Saddle,Stokes)

    # bcu = DirichletBC(W.sub(0),Expression(("0.0","0.0")), boundaries, 1)
    # bcb = DirichletBC(W.sub(2),Expression(("0.0","0.0")), boundaries, 1)
    # bcr = DirichletBC(W.sub(3),Expression(("0.0")), boundaries, 1)
    # bcs = [bcu,bcb,bcr]
    IS = MO.IndexSet(W, 'Blocks')

    parameters['linear_algebra_backend'] = 'uBLAS'

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-4     # tolerance
    iter = 0            # iteration counter
    maxiter = 10       # max no of iterations allowed
    SolutionTime = 0
    outer = 0
    # parameters['linear_algebra_backend'] = 'uBLAS'

    # FSpaces = [Velocity,Magnetic,Pressure,Lagrange]

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

        # if iter == 1:
        #     bcu = DirichletBC(W.sub(0),u0, boundaries, 1)
        #     bcb = DirichletBC(W.sub(2),b0, boundaries, 1)
        #     bcr = DirichletBC(W.sub(3),r0, boundaries, 1)
        #     bcs = [bcu,bcb,bcr]
        # else:
        bcu = DirichletBC(W.sub(0),Expression(("0.0","0.0")), boundary)
        bcb = DirichletBC(W.sub(2),Expression(("0.0","0.0")), boundary)
        bcr = DirichletBC(W.sub(3),Expression("0.0"), boundary)
        bcs = [bcu,bcb,bcr]
        # if iter == 1:
        # , L
        A, b = assemble_system(a, L, bcs)

        # AA = assemble(a)

        # bb = assemble(L)

        # for bc in bcs:
        #     bc.apply(AA,bb)


        # print A.sparray().todense()
        # MO.StoreMatrix(A.sparray(),'name')
        A, b = CP.Assemble(A,b)
        u = b.duplicate()
        # print b.array
        # ssss
        # L = assemble(L)
        # print L.array()
        # for bc in bcs:
        #     bc.apply(L)

        # print L.array()
        # MO.StrTimePrint("MHD total assemble, time: ", time.time()-AssembleTime)

        # u = b.duplicate()
        # kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k)
        # print "Inititial guess norm: ",  u.norm(PETSc.NormType.NORM_INFINITY)
        # #A,Q
        n = FacetNormal(mesh)
        b_t = TrialFunction(Velocity)
        c_t = TestFunction(Velocity)
        mat =  as_matrix([[b_k[1]*b_k[1],-b_k[1]*b_k[0]],[-b_k[1]*b_k[0],b_k[0]*b_k[0]]])
        aa = params[2]*inner(grad(b_t), grad(c_t))*dx(W.mesh()) + inner((grad(b_t)*u_k),c_t)*dx(W.mesh()) +(1./2)*div(u_k)*inner(c_t,b_t)*dx(W.mesh()) - (1./2)*inner(u_k,n)*inner(c_t,b_t)*ds(W.mesh())+kappa/Mu_m*inner(mat*b_t,c_t)*dx(W.mesh())
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
print IterTable.to_latex()

b = b_k.vector().array()
b = b/np.linalg.norm(b)
B = Function(Magnetic)
B.vector()[:] = b

# p = plot(u_k)
# p.write_png()

# p = plot(p_k)
# p.write_png()

# p = plot(B)
# p.write_png()

# p = plot(r_k)
# p.write_png()


interactive()
