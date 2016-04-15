#!/usr/bin/python
from dolfin import *

import numpy as np
import matplotlib.pylab as plt
import scipy.sparse as sps
import os
import scipy.io

import PETScIO as IO
import MatrixOperations as MO
import PyTrilinos.ML as ml
from PyTrilinos import AztecOO, Epetra

def StoreMatrix(A,name):
      test ="".join([name,".mat"])
      scipy.io.savemat( test, {name: A},oned_as='row')

parameters['num_threads'] = 10

m = 5
errL2b =np.zeros((m-1,1))
errCurlb =np.zeros((m-1,1))
errL2r =np.zeros((m-1,1))
errH1r =np.zeros((m-1,1))


l2border =  np.zeros((m-1,1))
Curlborder =np.zeros((m-1,1))
# set_log_level(DEBUG)


NN = np.zeros((m-1,1))
DoF = np.zeros((m-1,1))
Vdim = np.zeros((m-1,1))
Qdim = np.zeros((m-1,1))
Wdim = np.zeros((m-1,1))
iterations = np.zeros((m-1,1))
SolTime = np.zeros((m-1,1))
udiv = np.zeros((m-1,1))
MU = np.zeros((m-1,1))
nn = 2

dim = 2
Solving = 'Direct'
ShowResultPlots = 'no'
ShowErrorPlots = 'no'
EigenProblem = 'no'
SavePrecond = 'no'

CheckMu = 'no'
case =2
parameters['linear_algebra_backend'] = 'Epetra'
MU[0]= 1e0
for xx in xrange(1,m):

    # parameters["form_compiler"]["quadrature_degree"] = 3
    # parameters["form_compiler"]["optimize"] = True
    print xx
    nn = 2**(xx)

    if (CheckMu == 'yes'):
        if (xx != 1):
            MU[xx-1] = MU[xx-2]/10
    else:
        if (xx != 1):
            MU[xx-1] = MU[xx-2]
    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn

    mesh = RectangleMesh(0,0, 1, 1, nn, nn,'crossed')
    # mesh = UnitSquareMesh(nn,nn)
    parameters['reorder_dofs_serial'] = False
    V = FunctionSpace(mesh, "N1curl",2)
    Q = FunctionSpace(mesh, "CG", 2)
    Vdim[xx-1] = V.dim()
    def boundary(x, on_boundary):
        return on_boundary


    if case == 1:
        u0 = Expression(("x[0]*x[0]*(x[0]-1)","x[1]*x[1]*(x[1]-1)"))
    elif case == 2:
        u0 = Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])","-sin(2*pi*x[0])*cos(2*pi*x[1])"))
    elif case == 3:
        u0 = Expression(("-sin(2*pi*x[0])*cos(2*pi*x[1]) ","sin(2*pi*x[1])*cos(2*pi*x[0]) "))




    bcs = DirichletBC(V,u0, boundary)


    # (u1) = TrialFunctions(V)
    # (v1) = TestFunctions(V)
    c = .5
    if case == 1:
        # f= Expression(("(8*pow(pi,2)-C)*sin(2*pi*x[1])*cos(2*pi*x[0])","-(8*pow(pi,2)-C)*sin(2*pi*x[0])*cos(2*pi*x[1])"),C = c)
        f = Expression(("-6*x[1]+2","-6*x[0]+2"))+c*u0
        f = c*u0
    elif case == 2:
        f = 8*pow(pi,2)*u0+c*u0
    elif case == 3:
        f = Expression(("(4*pow(pi,2)-C)*sin(2*pi*x[1])*cos(2*pi*x[0])","-(4*pow(pi,2)-C)*sin(2*pi*x[0])*cos(2*pi*x[1])"),C = c)
        f = c*u0


    MLList = {
        "default values" : "maxwell",
        "max levels" : 10,
        "prec type" : "MGV",
        "increasing or decreasing" : "decreasing",
        "aggregation: type" : "Uncoupled-MIS",
        "aggregation: damping factor" : 1.333,
        "eigen-analysis: type" : "cg",
        "eigen-analysis: iterations" : 10,
        "aggregation: edge prolongator drop threshold" : 0.0,
        "smoother: sweeps" : 1,
        "smoother: damping factor" : 1.0,
        "smoother: pre or post" : "both",
        "smoother: type" : "Hiptmair",
        "smoother: Hiptmair efficient symmetric" : True,
        "subsmoother: type" : "Chebyshev",
        "subsmoother: Chebyshev alpha" : 20.0,
        "subsmoother: node sweeps" : 4,
        "subsmoother: edge sweeps" : 4,
        "coarse: type" : "Amesos-KLU",
        "coarse: max size" : 128
    }

    (u) = TrialFunction(V)
    (v) = TestFunction(V)
    (p) = TrialFunction(Q)
    (q) = TestFunction(Q)

    a = inner(curl(u),curl(v))*dx - 0*inner(u, v)*dx
    pp = inner(curl(u),curl(v))*dx+c*inner(u, v)*dx
    b = inner(u,grad(q))*dx
    L1  = inner(v, f)*dx


    tic()
    AA, bb = assemble_system(a, L1, bcs)
    A = as_backend_type(AA).mat()
    BB = assemble(b)
    LL = assemble(inner(grad(p),grad(q))*dx)
    PP = assemble(pp)
    bcs.apply(PP)
    # bcs.apply(BB)make
    P = as_backend_type(PP).mat()
    B = as_backend_type(BB).mat()
    L = as_backend_type(LL).mat()
    b = as_backend_type(bb).vec()
    x = Epetra.Vector(0*bb.array())
    # exit
    # A = as_backend_type(AA).mat()
    print toc()






    ML_Hiptmair = ml.MultiLevelPreconditioner(P,B,L,MLList)

    ML_Hiptmair.ComputePreconditioner()
    # ML_Hiptmair.ComputePreconditioner()
    solver = AztecOO.AztecOO(A, x, b)
    solver.SetPrecOperator(ML_Hiptmair)
    solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres);
    solver.SetAztecOption(AztecOO.AZ_output, 16);
    err = solver.Iterate(1550, 1e-5)

    # if (Solving == 'Direct'):
    #     ksp = PETSc.KSP().create()
    #     ksp.setOperators(A)

    #     ksp.setFromOptions()


    #     ksp.setType(ksp.Type.PREONLY)
    #     pc = ksp.getPC()
    #     pc.setFromOptions()
    #     pc.setType("lu")
    #     pc.setFactorSolverPackage("petsc")
    #     ksp.setOptionsPrefix("ksp_monitor")

    #     # ksp.view()

    #     # print 'Solving with:', ksp.getType()

    #     # Solve!
    #     tic()
    #     ksp.solve(bb, x)
    #     SolTime[xx-1] = toc()
    #     print "time to solve: ",SolTime[xx-1]
    #     del AA


    if (Solving == 'Iterative' or Solving == 'Direct'):
        if case == 1:
            ue = Expression(("x[0]*x[0]*(x[0]-1)","x[1]*x[1]*(x[1]-1)"))
        elif case == 2:
            ue = Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])","-sin(2*pi*x[0])*cos(2*pi*x[1])"))
        elif case == 3:
            ue = Expression(("-sin(2*pi*x[0])*cos(2*pi*x[1]) ","sin(2*pi*x[1])*cos(2*pi*x[0]) "))

        Ve = FunctionSpace(mesh,"N1curl",4)
        u = interpolate(ue,Ve)

        Nv= u.vector().array().shape
        print Nv
        X = IO.vecToArray(x)
        x = X[0:Nv[0]]
        ua = Function(V)
        ua.vector()[:] = x


        # parameters["form_compiler"]["quadrature_degree"] = 4
        # parameters["form_compiler"]["optimize"] = True


        ErrorB = Function(V)

        ErrorB=  u-ua  #interpolate(ue,Ve).vector().array()-ua.vector().array()


        errL2b[xx-1] = sqrt(assemble(inner(ErrorB, ErrorB)*dx))
        errCurlb[xx-1] = sqrt(assemble(inner(curl(ErrorB), curl(ErrorB))*dx))




        if xx == 1:
            a = 1
        else:

            l2border[xx-1] =  np.abs(np.log2(errL2b[xx-2]/errL2b[xx-1]))
            Curlborder[xx-1] =  np.abs(np.log2(errCurlb[xx-2]/errCurlb[xx-1]))

        print errL2b[xx-1]
        print errCurlb[xx-1]





import pandas as pd

print "\n\n   Magnetic convergence"
MagneticTitles = ["Total DoF","Soln Time","B-L2","B-order","B-Curl","Curl-order"]
MagneticValues = np.concatenate((Vdim,SolTime,errL2b,l2border,errCurlb,Curlborder),axis=1)
MagneticTable= pd.DataFrame(MagneticValues, columns = MagneticTitles)
pd.set_option('precision',3)
MagneticTable = MO.PandasFormat(MagneticTable,"B-Curl","%2.4e")
MagneticTable = MO.PandasFormat(MagneticTable,'B-L2',"%2.4e")
print MagneticTable

print MagneticTable.to_latex()




if (SavePrecond == 'yes'):
    scipy.io.savemat('eigenvalues/Wdim.mat', {'Wdim':Wdim-1},oned_as = 'row')


if (ShowResultPlots == 'yes'):
    plot(ua)
    plot(interpolate(ue,V))

    interactive()







