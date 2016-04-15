#!/usr/bin/python
from dolfin import *
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc

Print = PETSc.Sys.Print
# from MatrixOperations import *

import numpy as np
import matplotlib.pylab as plt
import scipy.sparse as sps
import scipy.sparse.linalg as slinalg
import os
import scipy.io
# from PyTrilinos import Epetra, EpetraExt, AztecOO, ML, Amesos
# from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
import PETScIO as IO
import MatrixOperations as MO

def StoreMatrix(A,name):
      test ="".join([name,".mat"])
      scipy.io.savemat( test, {name: A},oned_as='row')

parameters['num_threads'] = 1


m = 6
errL2b =np.zeros((m-1,1))
errCurlb =np.zeros((m-1,1))
errL2r =np.zeros((m-1,1))
errH1r =np.zeros((m-1,1))


l2border =  np.zeros((m-1,1))
Curlborder =np.zeros((m-1,1))
l2rorder =  np.zeros((m-1,1))
H1rorder = np.zeros((m-1,1))


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
case = 1
parameters['linear_algebra_backend'] = 'uBLAS'
MU[0]= 1e0
for xx in xrange(1,m):
    print xx
    nn = 2**(xx)
    parameters["form_compiler"]["quadrature_degree"] = 3
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
    parameters['reorder_dofs_serial'] = False
    V = FunctionSpace(mesh, "N1curl",2)
    Q = FunctionSpace(mesh, "CG",2)
    parameters['reorder_dofs_serial'] = False
    W = V*Q
    Vdim[xx-1] = V.dim()
    Qdim[xx-1] = Q.dim()
    Wdim[xx-1] = W.dim()
    print "\n\nV:  ",Vdim[xx-1],"Q:  ",Qdim[xx-1],"W:  ",Wdim[xx-1],"\n\n"
    def boundary(x, on_boundary):
        return on_boundary


    if case == 1:
        u0 = Expression(("x[1]*x[1]*(x[1]-1)","x[0]*x[0]*(x[0]-1)"))
        p0 = Expression("x[1]*(x[1]-1)*x[0]*(x[0]-1)")
    elif case == 2:
        u0 = Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])","-sin(2*pi*x[0])*cos(2*pi*x[1])"))
        p0 = Expression(("sin(2*pi*x[0])*sin(2*pi*x[1])"))
    elif case == 3:
        u0 = Expression(("cos(2*pi*x[1])*sin(2*pi*x[0]) ","-cos(2*pi*x[0])*sin(2*pi*x[1]) "))
        p0 = Expression("0")




    bc = DirichletBC(W.sub(0),u0, boundary)
    bc1 = DirichletBC(W.sub(1),p0, boundary)
    bcs = [bc,bc1]

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    c = 0
    if case == 1:
        CurlCurl = Expression(("-6*x[1]+2","-6*x[0]+2"))
        gradR = Expression(("(2*x[0]-1)*x[1]*(x[1]-1)","(2*x[1]-1)*x[0]*(x[0]-1)"))
        f = CurlCurl + gradR
    elif case == 2:
        CurlCurl = 8*pow(pi,2)*u0
        gradR = Expression(("2*pi*cos(x[0])*sin(2*pi*x[1])","2*pi*sin(2*pi*x[0])*cos(2*pi*x[1])"))
        f = CurlCurl+gradR
    elif case == 3:
        f = Expression(("(4*pow(pi,2)-C)*sin(2*pi*x[1])*cos(2*pi*x[0])","-(4*pow(pi,2)-C)*sin(2*pi*x[0])*cos(2*pi*x[1])"),C = c)


    a11 = inner(curl(v),curl(u))*dx
    a12 = inner(v,grad(p))*dx
    a21 = inner(u,grad(q))*dx
    L1  = inner(v, f)*dx
    a = a11+a12+a21

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
    del bb
    bb = IO.arrayToVec(b)
    x = IO.arrayToVec(zeros)


    if (Solving == 'Direct'):
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)

        ksp.setFromOptions()
        ksp.setType(ksp.Type.PREONLY)
        ksp.pc.setType(ksp.pc.Type.LU)
        ksp.pc.setFactorShift(1)
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
            pe = Expression("x[1]*(x[1]-1)*x[0]*(x[0]-1)")
        elif case == 2:
            ue = Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])","-sin(2*pi*x[0])*cos(2*pi*x[1])"))
            pe = Expression(("sin(2*pi*x[0])*sin(2*pi*x[1])"))
        elif case == 3:
            ue = Expression(("cos(2*pi*x[1])*sin(2*pi*x[0]) ","-cos(2*pi*x[0])*sin(2*pi*x[1]) "))
            pe = Expression("sin(2*pi*x[0])*sin(2*pi*x[1]) ")


        Ve = FunctionSpace(mesh,"N1curl",4)
        u = interpolate(ue,Ve)
        Qe = FunctionSpace(mesh,"CG",4)
        p = interpolate(pe,Qe)



        X = IO.vecToArray(x)
        x = X[0:V.dim()]
        ua = Function(V)
        ua.vector()[:] = x

        pp = X[V.dim():]
        pa = Function(Q)

        pa.vector()[:] = pp

        parameters["form_compiler"]["quadrature_degree"] = 4

        ErrorB = Function(V)
        ErrorR = Function(Q)

        ErrorB = u-ua
        ErrorR = p-pa


        errL2b[xx-1] = sqrt(assemble(inner(ErrorB, ErrorB)*dx))
        errCurlb[xx-1] = sqrt(assemble(inner(curl(ErrorB), curl(ErrorB))*dx))

        errL2r[xx-1] = sqrt(assemble(inner(ErrorR, ErrorR)*dx))
        errH1r[xx-1] = sqrt(assemble(inner(grad(ErrorR), grad(ErrorR))*dx))



        if xx == 1:
            a = 1
        else:

            l2border[xx-1] =  np.abs(np.log2(errL2b[xx-2]/errL2b[xx-1]))
            Curlborder[xx-1] =  np.abs(np.log2(errCurlb[xx-2]/errCurlb[xx-1]))

            l2rorder[xx-1] =  np.abs(np.log2(errL2r[xx-2]/errL2r[xx-1]))
            H1rorder[xx-1] =  np.abs(np.log2(errH1r[xx-2]/errH1r[xx-1]))

        print errL2b[xx-1]
        print errCurlb[xx-1]

        print errL2r[xx-1]
        print errH1r[xx-1]


if (ShowErrorPlots == 'yes'):
    plt.loglog(NN,errL2u)
    plt.title('Error plot for CG2 elements - Velocity L2 convergence = %f' % np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1]))))
    plt.xlabel('N')
    plt.ylabel('L2 error')


    plt.figure()

    plt.loglog(NN,errL2p)
    plt.title('Error plot for CG1 elements - Pressure L2 convergence = %f' % np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1]))))
    plt.xlabel('N')
    plt.ylabel('L2 error')

    plt.show()

import pandas as pd

print "\n\n   Magnetic convergence"
MagneticTitles = ["Total DoF","B DoF","Soln Time","B-L2","B-order","B-Curl","Curl-order"]
MagneticValues = np.concatenate((Wdim,Vdim,SolTime,errL2b,l2border,errCurlb,Curlborder),axis=1)
MagneticTable= pd.DataFrame(MagneticValues, columns = MagneticTitles)
pd.set_option('precision',3)
MagneticTable = MO.PandasFormat(MagneticTable,"B-Curl","%2.4e")
MagneticTable = MO.PandasFormat(MagneticTable,'B-L2',"%2.4e")
print MagneticTable

print "\n\n   Lagrange convergence"
LagrangeTitles = ["Total DoF","R DoF","Soln Time","R-L2","R-order","R-H1","H1-order"]
LagrangeValues = np.concatenate((Wdim,Qdim,SolTime,errL2r,l2rorder,errH1r,H1rorder),axis=1)
LagrangeTable= pd.DataFrame(LagrangeValues, columns = LagrangeTitles)
pd.set_option('precision',3)
LagrangeTable = MO.PandasFormat(LagrangeTable,'R-L2',"%2.4e")
LagrangeTable = MO.PandasFormat(LagrangeTable,'R-H1',"%2.4e")
print LagrangeTable


print MagneticTable.to_latex()
print LagrangeTable.to_latex()



if (SavePrecond == 'yes'):
    scipy.io.savemat('eigenvalues/Wdim.mat', {'Wdim':Wdim-1},oned_as = 'row')


if (ShowResultPlots == 'yes'):
    plot(ua)
    plot(interpolate(ue,V))

    plot(pa)
    plot(interpolate(pe,Q))

    interactive()







