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
#@profile
m = 6


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

def polarr(u, x, y):
    return (1./sqrt(x**2 + y**2))*(x*sy.diff(u,x)+y*sy.diff(u,y))

def polart(u, x, y):
    return -y*sy.diff(u,x)+x*sy.diff(u,y)


nn = 2

dim = 2
ShowResultPlots = 'yes'
split = 'Linear'
MU[0]= 1e0
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
    mesh, boundaries, domains = HartmanChannel.Domain(nn, L, y0, z0)
    # set_log_level(WARNING)

    parameters['form_compiler']['quadrature_degree'] = -1
    order = 2
    parameters['reorder_dofs_serial'] = False
    Velocity = VectorFunctionSpace(mesh, "CG", order)
    Pressure = FunctionSpace(mesh, "CG", order-1)
    VecPressure = VectorFunctionSpace(mesh, "CG", order-1)
    Magnetic = FunctionSpace(mesh, "N1curl", order-1    )
    Lagrange = FunctionSpace(mesh, "CG", order-1    )
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
    Mu_m =float(1e4)
    MU = 1.0

    N = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    # g = inner(p0*N - MU*grad(u0)*N,v)*dx

    IterType = 'Full'
    Split = "No"
    Saddle = "No"
    Stokes = "No"
    SetupType = 'python-class'
    # F_NS = -MU*Laplacian+Advection+gradPres-kappa*NS_Couple
    # if kappa == 0:
    #     F_M = Mu_m*CurlCurl+gradR -kappa*M_Couple
    # else:
    #     F_M = Mu_m*kappa*CurlCurl+gradR -kappa*M_Couple
    params = [kappa,Mu_m,MU]

    F_M = Expression(("0.0","0.0","0.0"))
    F_S = Expression(("0.0","0.0","0.0"))
    n = FacetNormal(mesh)

    trunc = 4
    u0, b0, pN, pN2 = HartmanChannel.ExactSol(mesh, params, y0, z0, trunc)

    b = Expression(("0.0","1.0","0.0"))
    r0 = Expression(("0.0"))
#    pN = -pN


    # u_k = Function(Velocity)
    # p_k = Function(Pressure)
    # b_k = Function(Magnetic)
    # r_k = Function(Lagrange)

    MO.PrintStr("Seting up initial guess matricies",2,"=","\n\n","\n")
    BCtime = time.time()
    BC = MHDsetup.BoundaryIndices(mesh)
    MO.StrTimePrint("BC index function, time: ", time.time()-BCtime)
    Hiptmairtol = 1e-6
    HiptmairMatrices = PrecondSetup.MagneticSetup(Magnetic, Lagrange, b, r0, Hiptmairtol, params)


    MO.PrintStr("Setting up MHD initial guess",5,"+","\n\n","\n\n")
    # u_k,p_k,b_k,r_k = common.InitialGuess(FSpaces,[u0,p0,b0,r0],[F_NS,F_M],params,HiptmairMatrices,1e-10,Neumann=None,options ="New")


    u_k, p_k = HartmanChannel.Stokes(Velocity, Pressure, F_S, u0, pN2, params, boundaries, domains)
    b_k, r_k = HartmanChannel.Maxwell(Magnetic, Lagrange, F_M, b0, r0, params, boundaries,HiptmairMatrices, Hiptmairtol)


    (u, p, b, r) = TrialFunctions(W)
    (v, q, c, s) = TestFunctions(W)

    m11 = params[1]*params[0]*inner(curl(b),curl(c))*dx('everywhere')
    m21 = inner(c,grad(r))*dx('everywhere')
    m12 = inner(b,grad(s))*dx('everywhere')

    a11 = params[2]*inner(grad(v), grad(u))*dx('everywhere') + inner((grad(u)*u_k),v)*dx('everywhere') + (1./2)*div(u_k)*inner(u,v)*dx('everywhere') - (1./2)*inner(u_k,n)*inner(u,v)*ds
    a12 = -div(v)*p*dx('everywhere')
    a21 = -div(u)*q*dx('everywhere')

    CoupleT = params[0]*inner(cross(v,b_k),curl(b))*dx('everywhere')
    Couple = -params[0]*inner(cross(u,b_k),curl(c))*dx('everywhere')

    a = m11 + m12 + m21 + a11 + a21 + a12 + Couple + CoupleT

    Lns  = inner(v, F_S)*dx('everywhere') #+ 0*inner(pN*n,v)*ds(1)
    Lmaxwell  = inner(c, F_M)*dx('everywhere')

    m11 = params[1]*params[0]*inner(curl(b_k),curl(c))*dx('everywhere')
    m21 = inner(c,grad(r_k))*dx('everywhere')
    m12 = inner(b_k,grad(s))*dx('everywhere')

    a11 = params[2]*inner(grad(v), grad(u_k))*dx('everywhere') + inner((grad(u_k)*u_k),v)*dx('everywhere') + (1./2)*div(u_k)*inner(u_k,v)*dx('everywhere') - (1./2)*inner(u_k,n)*inner(u_k,v)*ds
    a12 = -div(v)*p_k*dx('everywhere')
    a21 = -div(u_k)*q*dx('everywhere')
    CoupleT = params[0]*inner(cross(v,b_k),curl(b_k))*dx('everywhere')
    Couple = -params[0]*inner(cross(u_k,b_k),curl(c))*dx('everywhere')

    L = Lns + Lmaxwell - (m11 + m12 + m21 + a11 + a21 + a12 + Couple + CoupleT)



    ones = Function(Pressure)
    ones.vector()[:]=(0*ones.vector().array()+1)
    pConst = - assemble(p_k*dx('everywhere'))/assemble(ones*dx)
    p_k.vector()[:] += - assemble(p_k*dx)/assemble(ones*dx)
    x = Iter.u_prev(u_k,p_k,b_k,r_k)

    KSPlinearfluids, MatrixLinearFluids = PrecondSetup.FluidLinearSetup(Pressure, MU, mesh, boundaries, domains)
    kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k, mesh, boundaries, domains)
    #plot(b_k)

    # ns,maxwell,CoupleTerm,Lmaxwell,Lns = forms.MHD2D(mesh, W,F_M,F_NS, u_k,b_k,params,IterType,"CG",Saddle,Stokes)
    # RHSform = forms.PicardRHS(mesh, W, u_k, p_k, b_k, r_k, params,"CG",Saddle,Stokes)

    # bcu = DirichletBC(W.sub(0),Expression(("0.0","0.0")), boundaries, 1)
    # bcb = DirichletBC(W.sub(2),Expression(("0.0","0.0")), boundaries, 1)
    # bcr = DirichletBC(W.sub(3),Expression(("0.0")), boundaries, 1)
    # bcs = [bcu,bcb,bcr]
    IS = MO.IndexSet(W, 'Blocks')


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
        # bcu = DirichletBC(W.sub(0),Expression(("0.0","0.0","0.0")), boundaries, 2)
        bcu = DirichletBC(W.sub(0),Expression(("0.0","0.0","0.0")), boundary)
        bcb = DirichletBC(W.sub(2),Expression(("0.0","0.0","0.0")), boundary)
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
        print "                               Max rhs = ",np.max(b.array)
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
        mat = as_matrix([[b_k[2]*b_k[2]+b[1]*b[1],-b_k[1]*b_k[0],-b_k[0]*b_k[2]],
                         [-b_k[1]*b_k[0],b_k[0]*b_k[0]+b_k[2]*b_k[2],-b_k[2]*b_k[1]],
                       [-b_k[0]*b_k[2],-b_k[1]*b_k[2],b_k[0]*b_k[0]+b_k[1]*b_k[1]]])
        aa = params[2]*inner(grad(b_t), grad(c_t))*dx(W.mesh()) + inner((grad(b_t)*u_k),c_t)*dx(W.mesh()) +(1./2)*div(u_k)*inner(c_t,b_t)*dx(W.mesh()) - (1./2)*inner(u_k,n)*inner(c_t,b_t)*ds(W.mesh())+kappa/Mu_m*inner(mat*b_t,c_t)*dx(W.mesh())
        ShiftedMass = assemble(aa)
        bcu.apply(ShiftedMass)
        ShiftedMass = CP.Assemble(ShiftedMass)
        kspF = NSprecondSetup.LSCKSPnonlinear(ShiftedMass)

        stime = time.time()

        u, mits,nsits = S.solve(A,b,u,params,W,'Directss',IterType,OuterTol,InnerTol,HiptmairMatrices,Hiptmairtol,KSPlinearfluids, Fp,kspF)
        Soltime = time.time() - stime
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
    # u0, p0, b0, r0, F_NS, F_M, F_MX, F_S, gradu0, Neumann, p0vec = Lshaped.Solution2(mesh, params)
#    Vel = plot(u_k, prefix='velocityApprox')
#    Vel.write_png()
#    Vel = plot(interpolate(u0,Velocity), prefix='velocityExact')
#    Vel.write_png()
#
#    Vel = plot(p_k, prefix='pressureApprox')
#    Vel.write_png()
#    Vel = plot(interpolate(pN,Pressure), prefix='pressureExact')
#    Vel.write_png()

    ExactSolution = [u0,pN,b0,r0]
    errL2u[xx-1], errH1u[xx-1], errL2p[xx-1], errL2b[xx-1], errCurlb[xx-1], errL2r[xx-1], errH1r[xx-1] = Iter.Errors(XX,mesh,FSpaces,ExactSolution,order,dim, "CG")
    print float(Wdim[xx-1][0])/Wdim[xx-2][0]

    if xx > 1:

       l2uorder[xx-1] = np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1])/np.log2((float(Velocitydim[xx-1][0])/Velocitydim[xx-2][0])**(1./3)))
       H1uorder[xx-1] =  np.abs(np.log2(errH1u[xx-2]/errH1u[xx-1])/np.log2((float(Velocitydim[xx-1][0])/Velocitydim[xx-2][0])**(1./3)))

       l2porder[xx-1] =  np.abs(np.log2(errL2p[xx-2]/errL2p[xx-1])/np.log2((float(Pressuredim[xx-1][0])/Pressuredim[xx-2][0])**(1./3)))

       l2border[xx-1] =  np.abs(np.log2(errL2b[xx-2]/errL2b[xx-1])/np.log2((float(Magneticdim[xx-1][0])/Magneticdim[xx-2][0])**(1./3)))
       Curlborder[xx-1] =  np.abs(np.log2(errCurlb[xx-2]/errCurlb[xx-1])/np.log2((float(Magneticdim[xx-1][0])/Magneticdim[xx-2][0])**(1./3)))

       l2rorder[xx-1] =  np.abs(np.log2(errL2r[xx-2]/errL2r[xx-1])/np.log2((float(Lagrangedim[xx-1][0])/Lagrangedim[xx-2][0])**(1./3)))
       H1rorder[xx-1] =  np.abs(np.log2(errH1r[xx-2]/errH1r[xx-1])/np.log2((float(Lagrangedim[xx-1][0])/Lagrangedim[xx-2][0])**(1./3)))




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
