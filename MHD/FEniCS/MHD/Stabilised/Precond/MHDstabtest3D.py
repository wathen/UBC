#!/usr/bin/python

# interpolate scalar gradient onto nedelec space
from dolfin import *

import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
Print = PETSc.Sys.Print
# from MatrixOperations import *
import numpy as np
#import matplotlib.pylab as plt
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
m = 7


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

dim = 3
ShowResultPlots = 'yes'
split = 'Linear'

MU[0]= 1e0
for xx in xrange(1,m):
    print xx
    level[xx-1] = xx
    nn = 2**(level[xx-1])


    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn/2
    # parameters["form_compiler"]["quadrature_degree"] = 6
    # parameters = CP.ParameterSetup()
    mesh = UnitCubeMesh(nn,nn,nn)

    order = 1
    parameters['reorder_dofs_serial'] = False
    Velocity = VectorFunctionSpace(mesh, "CG", order)
    Pressure = FunctionSpace(mesh, "DG", order-1)
    Magnetic = FunctionSpace(mesh, "N1curl", order)
    Lagrange = FunctionSpace(mesh, "CG", order)
    W = MixedFunctionSpace([Velocity,Pressure,Magnetic,Lagrange])
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

    u0, p0,b0, r0, Laplacian, Advection, gradPres,CurlCurl, gradR, NS_Couple, M_Couple = ExactSol.MHD3D(1,1)

    bcu = DirichletBC(W.sub(0),u0, boundary)
    bcb = DirichletBC(W.sub(2),b0, boundary)
    bcr = DirichletBC(W.sub(3),r0, boundary)

    # bc = [u0,p0,b0,r0]
    bcs = [bcu,bcb,bcr]
    FSpaces = [Velocity,Pressure,Magnetic,Lagrange]


    (u, p, b, r) = TrialFunctions(W)
    (v, q, c,s ) = TestFunctions(W)

    kappa = 1.0
    Mu_m =1e1
    MU = 1.0
    IterType = 'MD'

    F_NS = -MU*Laplacian+Advection+gradPres-kappa*NS_Couple
    if kappa == 0:
        F_M = Mu_m*CurlCurl+gradR -kappa*M_Couple
    else:
        F_M = Mu_m*kappa*CurlCurl+gradR -kappa*M_Couple
    params = [kappa,Mu_m,MU]


    # MO.PrintStr("Preconditioning MHD setup",5,"+","\n\n","\n\n")
    HiptmairMatrices = PrecondSetup.MagneticSetup(Magnetic, Lagrange, b0, r0, 1e-6)


    MO.PrintStr("Setting up MHD initial guess",5,"+","\n\n","\n\n")
    u_k,p_k,b_k,r_k = common.InitialGuess(FSpaces,[u0,p0,b0,r0],[F_NS,F_M],params,HiptmairMatrices,1e-6,Neumann=Expression(("0","0")),options ="New", FS = "DG")

    ones = Function(Pressure)
    ones.vector()[:]=(0*ones.vector().array()+1)
    # pConst = - assemble(p_k*dx)/assemble(ones*dx)
    p_k.vector()[:] += - assemble(p_k*dx)/assemble(ones*dx)
    x = Iter.u_prev(u_k,p_k,b_k,r_k)

    KSPlinearfluids, MatrixLinearFluids = PrecondSetup.FluidLinearSetup(Pressure, MU)
    kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k)
    # plot(b_k)

    ns,maxwell,CoupleTerm,Lmaxwell,Lns = forms.MHD2D(mesh, W,F_M,F_NS, u_k,b_k,params,IterType,"DG")
    RHSform = forms.PicardRHS(mesh, W, u_k, p_k, b_k, r_k, params,"DG")


    bcu = DirichletBC(W.sub(0),Expression(("0.0","0.0","0.0")), boundary)
    bcb = DirichletBC(W.sub(2),Expression(("0.0","0.0","0.0")), boundary)
    bcr = DirichletBC(W.sub(3),Expression(("0.0")), boundary)
    bcs = [bcu,bcb,bcr]

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-4     # tolerance
    iter = 0            # iteration counter
    maxiter = 40       # max no of iterations allowed
    SolutionTime = 0
    outer = 0
    # parameters['linear_algebra_backend'] = 'uBLAS'


    if IterType == "CD":
        AA, bb = assemble_system(maxwell+ns, (Lmaxwell + Lns) - RHSform,  bcs)
        A,b = CP.Assemble(AA,bb)
        # u = b.duplicate()
        # P = CP.Assemble(PP)


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
        tic()
        if IterType == "CD":
            bb = assemble((Lmaxwell + Lns) - RHSform)
            for bc in bcs:
                bc.apply(bb)
            A,b = CP.Assemble(AA,bb)
            # if iter == 1
            if iter == 1:
                u = b.duplicate()
                F = A.getSubMatrix(u_is,u_is)
                kspF = NSprecondSetup.LSCKSPnonlinear(F)
        else:
            AA, bb = assemble_system(maxwell+ns+CoupleTerm, (Lmaxwell + Lns) - RHSform,  bcs)
            A,b = CP.Assemble(AA,bb)
            F = A.getSubMatrix(u_is,u_is)
            kspF = NSprecondSetup.LSCKSPnonlinear(F)
        # if iter == 1:
            if iter == 1:
                u = b.duplicate()
        print ("{:40}").format("MHD assemble, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

        kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k)

        print "Inititial guess norm: ", u.norm()
        stime = time.time()
        # ksp.solve(b, u)
        u,it1,it2 = S.solve(A,b,u,[NS_is,M_is],FSpaces,IterType,OuterTol,InnerTol,HiptmairMatrices,KSPlinearfluids,kspF,Fp)
        Soltime = time.time()- stime
        NSits += it1
        Mits +=it2
        SolutionTime = SolutionTime +Soltime

        u1, p1, b1, r1, eps= Iter.PicardToleranceDecouple(u,x,FSpaces,dim,"2",iter)
        p1.vector()[:] += - assemble(p1*dx)/assemble(ones*dx)
        u_k.assign(u1)
        p_k.assign(p1)
        b_k.assign(b1)
        r_k.assign(r1)
        # if eps > 100 and iter > 3:
        #     print 22222
        #     break
        uOld= np.concatenate((u_k.vector().array(),p_k.vector().array(),b_k.vector().array(),r_k.vector().array()), axis=0)
        x = IO.arrayToVec(uOld)
        # iter = 10000


        # u_k,b_k,epsu,epsb=Iter.PicardTolerance(x,u_k,b_k,FSpaces,dim,"inf",iter)


    SolTime[xx-1] = SolutionTime/iter
    NSave[xx-1] = (float(NSits)/iter)
    Mave[xx-1] = (float(Mits)/iter)
    iterations[xx-1] = iter
    TotalTime[xx-1] = time.time() - TotalStart
    ue =u0
    pe = p0
    be = b0
    re = r0




    # ExactSolution = [ue,pe,be,re]
    # errL2u[xx-1], errH1u[xx-1], errL2p[xx-1], errL2b[xx-1], errCurlb[xx-1], errL2r[xx-1], errH1r[xx-1] = Iter.Errors(x,mesh,FSpaces,ExactSolution,order,dim, "DG")

    # if xx > 1:
    #     l2uorder[xx-1] =  np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1]))
    #     H1uorder[xx-1] =  np.abs(np.log2(errH1u[xx-2]/errH1u[xx-1]))

    #     l2porder[xx-1] =  np.abs(np.log2(errL2p[xx-2]/errL2p[xx-1]))

    #     l2border[xx-1] =  np.abs(np.log2(errL2b[xx-2]/errL2b[xx-1]))
    #     Curlborder[xx-1] =  np.abs(np.log2(errCurlb[xx-2]/errCurlb[xx-1]))

    #     l2rorder[xx-1] =  np.abs(np.log2(errL2r[xx-2]/errL2r[xx-1]))
    #     H1rorder[xx-1] =  np.abs(np.log2(errH1r[xx-2]/errH1r[xx-1]))




import pandas as pd



# LatexTitles = ["l","DoFu","Dofp","V-L2","L2-order","V-H1","H1-order","P-L2","PL2-order"]
# LatexValues = np.concatenate((level,Velocitydim,Pressuredim,errL2u,l2uorder,errH1u,H1uorder,errL2p,l2porder), axis=1)
# LatexTable = pd.DataFrame(LatexValues, columns = LatexTitles)
# pd.set_option('precision',3)
# LatexTable = MO.PandasFormat(LatexTable,"V-L2","%2.4e")
# LatexTable = MO.PandasFormat(LatexTable,'V-H1',"%2.4e")
# LatexTable = MO.PandasFormat(LatexTable,"H1-order","%1.2f")
# LatexTable = MO.PandasFormat(LatexTable,'L2-order',"%1.2f")
# LatexTable = MO.PandasFormat(LatexTable,"P-L2","%2.4e")
# LatexTable = MO.PandasFormat(LatexTable,'PL2-order',"%1.2f")
# print LatexTable


# print "\n\n   Magnetic convergence"
# MagneticTitles = ["l","B DoF","R DoF","B-L2","L2-order","B-Curl","HCurl-order"]
# MagneticValues = np.concatenate((level,Magneticdim,Lagrangedim,errL2b,l2border,errCurlb,Curlborder),axis=1)
# MagneticTable= pd.DataFrame(MagneticValues, columns = MagneticTitles)
# pd.set_option('precision',3)
# MagneticTable = MO.PandasFormat(MagneticTable,"B-Curl","%2.4e")
# MagneticTable = MO.PandasFormat(MagneticTable,'B-L2',"%2.4e")
# MagneticTable = MO.PandasFormat(MagneticTable,"L2-order","%1.2f")
# MagneticTable = MO.PandasFormat(MagneticTable,'HCurl-order',"%1.2f")
# print MagneticTable

# print "\n\n   Lagrange convergence"
# LagrangeTitles = ["l","SolTime","B DoF","R DoF","R-L2","L2-order","R-H1","H1-order"]
# LagrangeValues = np.concatenate((level,SolTime,Magneticdim,Lagrangedim,errL2r,l2rorder,errH1r,H1rorder),axis=1)
# LagrangeTable= pd.DataFrame(LagrangeValues, columns = LagrangeTitles)
# pd.set_option('precision',3)
# LagrangeTable = MO.PandasFormat(LagrangeTable,"R-L2","%2.4e")
# LagrangeTable = MO.PandasFormat(LagrangeTable,'R-H1',"%2.4e")
# LagrangeTable = MO.PandasFormat(LagrangeTable,"H1-order","%1.2f")
# LagrangeTable = MO.PandasFormat(LagrangeTable,'L2-order',"%1.2f")
# print LagrangeTable

print "\n\n   Iteration table"
if IterType == "Full":
    IterTitles = ["l","DoF","AV solve Time","Total picard time","picard iterations","Av Outer its","Av Inner its",]
else:
    IterTitles = ["l","DoF","AV solve Time","Total picard time","picard iterations","Av NS iters","Av M iters"]
IterValues = np.concatenate((level,Wdim,SolTime,TotalTime,iterations,NSave,Mave),axis=1)
IterTable= pd.DataFrame(IterValues, columns = IterTitles)
if IterType == "Full":
    IterTable = MO.PandasFormat(IterTable,'Av Outer its',"%2.1f")
    IterTable = MO.PandasFormat(IterTable,'Av Inner its',"%2.1f")
else:
    IterTable = MO.PandasFormat(IterTable,'Av NS iters',"%2.1f")
    IterTable = MO.PandasFormat(IterTable,'Av M iters',"%2.1f")
print IterTable.to_latex()
print " \n  Outer Tol:  ",OuterTol, "Inner Tol:   ", InnerTol




# # # if (ShowResultPlots == 'yes'):

# plot(u_k)
# plot(interpolate(ue,Velocity))

# plot(p_k)

# plot(interpolate(pe,Pressure))

# plot(b_k)
# plot(interpolate(be,Magnetic))

# plot(r_k)
# plot(interpolate(re,Lagrange))

# interactive()

interactive()
