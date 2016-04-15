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
import scipy.sparse as sps
import scipy.sparse.linalg as slinalg
import os
import scipy.io

import PETScIO as IO
import MatrixOperations as MO

def StoreMatrix(A,name):
      test ="".join([name,".mat"])
      scipy.io.savemat( test, {name: A},oned_as='row')

parameters['num_threads'] = 10

m = 6
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
ShowResultPlots = 'yes'
ShowErrorPlots = 'no'
EigenProblem = 'no'
SavePrecond = 'no'

CheckMu = 'no'
case = 4
parameters['linear_algebra_backend'] = 'uBLAS'
MU[0]= 1e0
for xx in xrange(1,m):
    print xx
    nn = 2**(xx)/2

    if (CheckMu == 'yes'):
        if (xx != 1):
            MU[xx-1] = MU[xx-2]/10
    else:
        if (xx != 1):
            MU[xx-1] = MU[xx-2]
    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn
    parameters["form_compiler"]["quadrature_degree"] = 3
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["representation"] = 'quadrature'


    # mesh = BoxMesh(-1,-1,-1,1, 1, 1, nn, nn, nn)
    mesh = UnitCubeMesh(nn,nn,nn)
    parameters['reorder_dofs_serial'] = False
    V = FunctionSpace(mesh, "N1curl",2)
    Q = FunctionSpace(mesh, "CG",2)
    Vdim[xx-1] = V.dim()
    print "\n\n\n V-dim", V.dim()
    def boundary(x, on_boundary):
        return on_boundary


    if case == 1:
        u0 = Expression(("x[1]*x[1]*(x[1]-1)","x[0]*x[0]*(x[0]-1)","0"))
    elif case == 2:
        u0 = Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])","-sin(2*pi*x[0])*cos(2*pi*x[1])"))
    elif case == 3:
        u0 = Expression(("x[0]*x[0]*(x[0]-1)","x[1]*x[1]*(x[1]-1)","0"))
    elif  case == 4:
        u0 = Expression(("x[0]*x[1]*x[2]*(x[0]-1)","x[0]*x[1]*x[2]*(x[1]-1)","x[0]*x[1]*x[2]*(x[2]-1)"))



    bcs = DirichletBC(V,u0, boundary)


    # (u1) = TrialFunctions(V)
    # (v1) = TestFunctions(V)
    c = .5
    if case == 1:
        # f= Expression(("(8*pow(pi,2)-C)*sin(2*pi*x[1])*cos(2*pi*x[0])","-(8*pow(pi,2)-C)*sin(2*pi*x[0])*cos(2*pi*x[1])"),C = c)
        f = Expression(("-6*x[1]+2","-6*x[0]+2"))+c*u0
    elif case == 2:
        f = 8*pow(pi,2)*u0+c*u0
    elif case == 3:
        f = Expression(("0","0","0"),C = c)
        f = c*u0
    elif  case == 4:
        f = Expression(("x[2]*(2*x[1]-1)+x[1]*(2*x[2]-1)","x[0]*(2*x[2]-1)+x[2]*(2*x[0]-1)","x[1]*(2*x[0]-1)+x[0]*(2*x[1]-1)"))+c*u0


    (u) = TrialFunction(V)
    (v) = TestFunction(V)

    a = dot(curl(u),curl(v))*dx+c*inner(u, v)*dx

    L1  = inner(v, f)*dx


    tic()
    AA, bb = assemble_system(a, L1, bcs)
    As = AA.sparray()
    StoreMatrix(As,'A')
    A = PETSc.Mat().createAIJ(size=As.shape,csr=(As.indptr, As.indices, As.data))
    # exit
    # A = as_backend_type(AA).mat()
    print toc()
    b = bb.array()
    zeros = 0*b
    x = IO.arrayToVec(zeros)
    bb = IO.arrayToVec(b)

    if (Solving == 'Direct'):
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)

        ksp.setFromOptions()
        ksp.setType(ksp.Type.PREONLY)
        ksp.pc.setType(ksp.pc.Type.LU)
        # print 'Solving with:', ksp.getType()

        # Solve!
        tic()
        ksp.solve(bb, x)
        SolTime[xx-1] = toc()
        print "time to solve: ",SolTime[xx-1]
        del AA


    if (Solving == 'Iterative' or Solving == 'Direct'):
        if case == 1:
            ue = Expression(("x[1]*x[1]*(x[1]-1)","x[0]*x[0]*(x[0]-1)"))
        elif case == 2:
            ue = Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])","-sin(2*pi*x[0])*cos(2*pi*x[1])"))
        elif case == 3:
            ue=u0
        elif  case == 4:
            ue=u0

        Ve = FunctionSpace(mesh, "N1curl",4)
        u = interpolate(ue,Ve)


        Nv  = u.vector().array().shape

        X = IO.vecToArray(x)
        x = X[0:Nv[0]]
        ua = Function(V)
        ua.vector()[:] = x


        parameters["form_compiler"]["quadrature_degree"] = 4
        parameters["form_compiler"]["optimize"] = True


        ErrorB = Function(V)

        ErrorB.vector()[:] = interpolate(ue,V).vector().array()-ua.vector().array()


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





if (SavePrecond == 'yes'):
    scipy.io.savemat('eigenvalues/Wdim.mat', {'Wdim':Wdim-1},oned_as = 'row')


if (ShowResultPlots == 'yes'):
    plot(ua)
    plot(interpolate(ue,V))

    interactive()







