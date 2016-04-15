import os, inspect
from dolfin import *
import numpy
import ExactSol
import MatrixOperations as MO
import CheckPetsc4py as CP
import petsc4py
import sys
import HiptmairPrecond
import HiptmairSetup
petsc4py.init(sys.argv)
from petsc4py import PETSc
import matplotlib.pylab as plt
from timeit import default_timer as timer
m = 4
errL2b =numpy.zeros((m-1,1))
errCurlb =numpy.zeros((m-1,1))

l2border =  numpy.zeros((m-1,1))
Curlborder =numpy.zeros((m-1,1))

ItsSave = numpy.zeros((m-1,1))
DimSave = numpy.zeros((m-1,1))
TimeSave = numpy.zeros((m-1,1))
NN = numpy.zeros((m-1,1))
Curlgrad = numpy.zeros((m-1,1))
Massgrad = numpy.zeros((m-1,1))
Laplgrad = numpy.zeros((m-1,1))
dim =3

for xx in xrange(1,m):
    NN[xx-1] = xx+3
    nn = int(2**(NN[xx-1][0]))
    # nn = 1
    omega = 1
    if dim == 2:
        # mesh = UnitSquareMesh(int(nn),int(nn))
        mesh =  RectangleMesh(0.0, 0.0, 1.0, 1.5, int(nn), int(nn), 'left')
        u0, p0, CurlCurl, gradPres, CurlMass = ExactSol.M2D(2,Show="yes", Mass = omega)
    else:
        mesh = UnitCubeMesh(int(nn),int(nn),int(nn))
        u0, p0, CurlCurl, gradPres, CurlMass = ExactSol.M3D(1,Show="yes", Mass = omega)

    order = 1
    parameters['reorder_dofs_serial'] = False
    Magnetic = FunctionSpace(mesh, "N1curl", order)
    Lagrange = FunctionSpace(mesh, "CG", order)
    parameters['reorder_dofs_serial'] = False

    DimSave[xx-1] = Magnetic.dim()
    print Magnetic.dim()
    parameters['linear_algebra_backend'] = 'uBLAS'

    # tic()
    C, P = HiptmairSetup.HiptmairMatrixSetupBoundary(mesh, Magnetic.dim(), Lagrange.dim(),dim)
    G, P = HiptmairSetup.HiptmairBCsetupBoundary(C,P,mesh)
    # endTimeB = toc()
    # print endTimeB
    print "\n"
    # tic()
    # C, P = HiptmairSetup.HiptmairMatrixSetup(mesh, Magnetic.dim(), Lagrange.dim())
    # G, P = HiptmairSetup.HiptmairBCsetup(C,P, mesh, [Magnetic,Lagrange])
    # endTime = toc()
    # print endTime

    # ataaa
    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(Magnetic,u0, boundary)
    bcu = DirichletBC(Lagrange, Expression(("0.0")), boundary)


    (v) = TestFunction(Magnetic)
    (u) = TrialFunction(Magnetic)

    (p) = TrialFunction(Lagrange)
    (q) = TestFunction(Lagrange)

    a = inner(curl(u),curl(v))*dx + inner(u,v)*dx
    L1  = inner(v, CurlMass)*dx
    tic()
    Acurl,b = assemble_system(a,L1,bc, form_compiler_parameters={"eliminate_zeros": True})
    print "System assembled, time: ", toc()

    tic()
    A,b = CP.Assemble(Acurl,b)
    x = b.duplicate()
    print "PETSc system assembled, time: ", toc()

    tic()
    ScalarLaplacian, b1 = assemble_system(inner(grad(p),grad(q))*dx,inner(p0,q)*dx,bcu)
    VectorLaplacian, b2 = assemble_system(inner(grad(p),grad(q))*dx+inner(p,q)*dx,inner(p0,q)*dx,bcu)
    del b1, b2
    print "Hiptmair Laplacians BC assembled, time: ", toc()

    tic()
    VectorLaplacian = PETSc.Mat().createAIJ(size=VectorLaplacian.sparray().shape,csr=(VectorLaplacian.sparray().indptr, VectorLaplacian.sparray().indices, VectorLaplacian.sparray().data))
    ScalarLaplacian = PETSc.Mat().createAIJ(size=ScalarLaplacian.sparray().shape,csr=(ScalarLaplacian.sparray().indptr, ScalarLaplacian.sparray().indices, ScalarLaplacian.sparray().data))
    print "PETSc Laplacians assembled, time: ", toc()

    ksp = PETSc.KSP().create()
    ksp.setTolerances(1e-6)
    ksp.setType('cg')
    ksp.setOperators(A,A)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    kspVector, kspScalar, diag = HiptmairSetup.HiptmairKSPsetup(VectorLaplacian, ScalarLaplacian,A)
    del A, VectorLaplacian, ScalarLaplacian
    pc.setPythonContext(HiptmairPrecond.GSvector(G, P, kspVector, kspScalar, diag))
    scale = b.norm()
    b = b/scale
    tic()
    ksp.solve(b, x)
    TimeSave[xx-1] = toc()
    x = x*scale
    print ksp.its
    print TimeSave[xx-1]
    ItsSave[xx-1] = ksp.its
    print " \n\n\n\n"

import pandas as pd


print "\n\n\n"
ItsTitlesB = ["l","B DoF","Time","Iterations"]
ItsValuesB = numpy.concatenate((NN,DimSave,TimeSave,ItsSave),axis=1)
ItsTableB= pd.DataFrame(ItsValuesB, columns = ItsTitlesB)
pd.set_option('precision',5)
print ItsTableB.to_latex()

if m !=2:
    print numpy.abs((TimeSave[1:]/TimeSave[:-1]))/(2*dim)
