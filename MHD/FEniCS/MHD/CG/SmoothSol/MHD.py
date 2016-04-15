#!/usr/bin/python
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
Print = PETSc.Sys.Print
# from MatrixOperations import *
from dolfin import *
import numpy as np
import matplotlib.pylab as plt
import PETScIO as IO
import common
import scipy
import scipy.io


import BiLinearForms as forms
import DirectOperations as Direct
import MatrixOperations as MO



m = 5

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
nn = 2

dim = 2
ShowResultPlots = 'no'
split = 'Linear'

MU[0]= 1e0
for xx in xrange(1,m):
    print xx
    nn = 2**(xx )


    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn/2

    mesh = RectangleMesh(0, 0, 1, 1, nn, nn,'crossed')

    order = 2

    parameters['reorder_dofs_serial'] = False
    Velocity = VectorFunctionSpace(mesh, "CG", order)
    Pressure = FunctionSpace(mesh, "CG", order-1)
    Magnetic = FunctionSpace(mesh, "N1curl", order)
    Lagrange = FunctionSpace(mesh, "CG", order)
    parameters['reorder_dofs_serial'] = False
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

    u0 =Expression(("sin(x[1])*exp(x[0])","cos(x[1])*exp(x[0])"))
    p0 = Expression("sin(x[0])*cos(x[1])")
    b0 = Expression(("x[1]*x[1]*(x[1]-1)","x[0]*x[0]*(x[0]-1)"))
    r0 = Expression("x[1]*(x[1]-1)*x[0]*(x[0]-1)")



    bcu = DirichletBC(W.sub(0),u0, boundary)
    bcp = DirichletBC(W.sub(1),p0, boundary)
    bcb = DirichletBC(W.sub(2),b0, boundary)
    bcr = DirichletBC(W.sub(3),r0, boundary)
    bc = [bcu,bcb,bcr]
    FSpaces = [Velocity,Pressure,Magnetic,Lagrange]


    (u, p, b, r) = TrialFunctions(W)
    (v, q, c,s ) = TestFunctions(W)
    kappa = 1e2

    Mu_m = 1e2
    MU = 1


    Laplacian = -MU*Expression(("0","0"))
    Advection = Expression(("pow(exp(x[0]),2)","0"))
    gradPres = Expression(("cos(x[1])*cos(x[0])","-sin(x[1])*sin(x[0])"))
    b_bar = Expression(("3*pow(x[0],2)-2*x[0]-3*pow(x[1],2)+2*x[1]"))
    NS_couple = -kappa*b_bar*Expression(("-x[0]*x[0]*(x[0]-1)","x[1]*x[1]*(x[1]-1)"))

    F_NS = Laplacian+Advection+gradPres+NS_couple




    CurlCurl = kappa*Mu_m *Expression(("-6*x[1]+2","-6*x[0]+2"))
    gradR = Expression(("(2*x[0]-1)*x[1]*(x[1]-1)","(2*x[1]-1)*x[0]*(x[0]-1)"))
    M_couple = -kappa*Expression(("pow(x[0],2)*exp(x[0])*cos(x[1])*(x[0] - 1) - 2*x[1]*exp(x[0])*cos(x[1])*(x[1] - 1) - pow(x[1],2)*exp(x[0])*cos(x[1]) + pow(x[1],2)*exp(x[0])*sin(x[1])*(x[1] - 1)","pow(x[1],2)*exp(x[0])*cos(x[1])*(x[1] - 1) - 2*x[0]*exp(x[0])*sin(x[1])*(x[0] - 1) - pow(x[0],2)*exp(x[0])*sin(x[1]) - pow(x[0],2)*exp(x[0])*sin(x[1])*(x[0] - 1)"))
    F_M = CurlCurl+gradR +M_couple

    params = [kappa,Mu_m,MU]
    u_k,b_k = common.InitialGuess(FSpaces,[u0,p0,b0,r0],[gradPres,F_M],params,Neumann=Expression(("0","0")))

    if (split == "Linear"):
        ns,maxwell,CoupleTerm,Lmaxwell,Lns = forms.MHD2D(mesh, W,F_M,F_NS, u_k,b_k,params,split)
        a = ns+maxwell+CoupleTerm
        L = Lmaxwell+Lns
    elif (split == "NoneLinear"):
        linear, Nlinear, RHS = MHD2D(mesh, W,F_M,F_NS, u_k,b_k,params,split)
        a = Nlinear

    parameters['linear_algebra_backend'] = 'uBLAS'



    epsu = 1.0           # error measure ||u-u_k||
    epsb = 1.0
    tol = 1.0E-8       # tolerance
    iter = 0            # iteration counter
    maxiter = 10        # max no of iterations allowed
    SolutionTime = 0
    while epsu > tol  and iter < maxiter:
        iter += 1
        uu = Function(W)
        AA, bb = assemble_system(maxwell+ns+CoupleTerm, Lmaxwell + Lns, bc)


        VelPres = Velocitydim[xx-1][0] +Pressuredim[xx-1][0]


        A,b,x = Direct.RemoveRowCol(AA,bb,VelPres)

        ksp = PETSc.KSP().create()
        pc = PETSc.PC().create()
        ksp.setOperators(A)

        ksp.setFromOptions()

        print '\n\n\nSolving with:', ksp.getType()


        tic()

        ksp.solve(b, x)

        time = toc()
        print time
        SolutionTime = SolutionTime +time

        u_k,b_k,epsu, epsb = Direct.PicardTolerance(x,u_k,b_k,FSpaces,dim,"inf",iter)


    SolTime[xx-1] = SolutionTime/iter

    ue =Expression(("sin(x[1])*exp(x[0])","cos(x[1])*exp(x[0])"))
    pe = Expression("sin(x[0])*cos(x[1])")
    be = Expression(("x[1]*x[1]*(x[1]-1)","x[0]*x[0]*(x[0]-1)"))
    re = Expression("x[1]*(x[1]-1)*x[0]*(x[0]-1)")

    ExactSolution = [ue,pe,be,re]
    errL2u[xx-1], errH1u[xx-1], errL2p[xx-1], errL2b[xx-1], errCurlb[xx-1], errL2r[xx-1], errH1r[xx-1] = Direct.Errors(x,mesh,FSpaces,ExactSolution,order,dim)

    if xx == 1:
        l2uorder[xx-1] = 0
    else:
        l2uorder[xx-1] =  np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1]))
        H1uorder[xx-1] =  np.abs(np.log2(errH1u[xx-2]/errH1u[xx-1]))

        l2porder[xx-1] =  np.abs(np.log2(errL2p[xx-2]/errL2p[xx-1]))

        l2border[xx-1] =  np.abs(np.log2(errL2b[xx-2]/errL2b[xx-1]))
        Curlborder[xx-1] =  np.abs(np.log2(errCurlb[xx-2]/errCurlb[xx-1]))

        l2rorder[xx-1] =  np.abs(np.log2(errL2r[xx-2]/errL2r[xx-1]))
        H1rorder[xx-1] =  np.abs(np.log2(errH1r[xx-2]/errH1r[xx-1]))




import pandas as pd


print "\n\n   Velocity convergence"
VelocityTitles = ["Total DoF","V DoF","Soln Time","V-L2","L2-order","V-H1","H1-order"]
VelocityValues = np.concatenate((Wdim,Velocitydim,SolTime,errL2u,l2uorder,errH1u,H1uorder),axis=1)
VelocityTable= pd.DataFrame(VelocityValues, columns = VelocityTitles)
pd.set_option('precision',3)
VelocityTable = MO.PandasFormat(VelocityTable,"V-L2","%2.4e")
VelocityTable = MO.PandasFormat(VelocityTable,'V-H1',"%2.4e")
VelocityTable = MO.PandasFormat(VelocityTable,"H1-order","%1.2f")
VelocityTable = MO.PandasFormat(VelocityTable,'L2-order',"%1.2f")
print VelocityTable

print "\n\n   Pressure convergence"
PressureTitles = ["Total DoF","P DoF","Soln Time","P-L2","L2-order"]
PressureValues = np.concatenate((Wdim,Pressuredim,SolTime,errL2p,l2porder),axis=1)
PressureTable= pd.DataFrame(PressureValues, columns = PressureTitles)
pd.set_option('precision',3)
PressureTable = MO.PandasFormat(PressureTable,"P-L2","%2.4e")
PressureTable = MO.PandasFormat(PressureTable,'L2-order',"%1.2f")
print PressureTable


print "\n\n   Magnetic convergence"
MagneticTitles = ["Total DoF","B DoF","Soln Time","B-L2","L2-order","B-Curl","HCurl-order"]
MagneticValues = np.concatenate((Wdim,Magneticdim,SolTime,errL2b,l2border,errCurlb,Curlborder),axis=1)
MagneticTable= pd.DataFrame(MagneticValues, columns = MagneticTitles)
pd.set_option('precision',3)
MagneticTable = MO.PandasFormat(MagneticTable,"B-Curl","%2.4e")
MagneticTable = MO.PandasFormat(MagneticTable,'B-L2',"%2.4e")
MagneticTable = MO.PandasFormat(MagneticTable,"L2-order","%1.2f")
MagneticTable = MO.PandasFormat(MagneticTable,'HCurl-order',"%1.2f")
print MagneticTable

print "\n\n   Lagrange convergence"
LagrangeTitles = ["Total DoF","R DoF","Soln Time","R-L2","L2-order","R-H1","H1-order"]
LagrangeValues = np.concatenate((Wdim,Lagrangedim,SolTime,errL2r,l2rorder,errH1r,H1rorder),axis=1)
LagrangeTable= pd.DataFrame(LagrangeValues, columns = LagrangeTitles)
pd.set_option('precision',3)
LagrangeTable = MO.PandasFormat(LagrangeTable,"R-L2","%2.4e")
LagrangeTable = MO.PandasFormat(LagrangeTable,'R-H1',"%2.4e")
LagrangeTable = MO.PandasFormat(LagrangeTable,"H1-order","%1.2f")
LagrangeTable = MO.PandasFormat(LagrangeTable,'L2-order',"%1.2f")
print LagrangeTable




if (ShowResultPlots == 'yes'):

    plot(ua)
    plot(interpolate(ue,Velocity))

    plot(pp)
    plot(interpolate(pe,Pressure))

    plot(ba)
    plot(interpolate(be,Magnetic))

    plot(ra)
    plot(interpolate(re,Lagrange))

interactive()








