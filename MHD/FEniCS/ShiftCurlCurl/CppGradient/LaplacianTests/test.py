import os, inspect
from dolfin import *
import numpy
from scipy.sparse import coo_matrix, block_diag, hstack, tril
import ExactSol
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
from PyTrilinos import Epetra, ML, AztecOO, Teuchos
import MatrixOperations as MO
import matplotlib.pylab as plt
import CheckPetsc4py as CP
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc



m = 9
errL2b =numpy.zeros((m-1,1))
errCurlb =numpy.zeros((m-1,1))

l2border =  numpy.zeros((m-1,1))
Curlborder =numpy.zeros((m-1,1))

ItsSave = numpy.zeros((m-1,2))
DimSave = numpy.zeros((m-1,1))
TimeSave = numpy.zeros((m-1,1))
NN = numpy.zeros((m-1,1))
dim = 2

for xx in xrange(1,m):
    NN[xx-1] = xx+0
    nn = 2**(NN[xx-1][0])
    omega = 1
    if dim == 2:
        mesh = UnitSquareMesh(int(nn),int(nn))
        # mesh =  RectangleMesh(0,0, 1, 1, int(nn), int(nn),'left')
        u0, p0, CurlCurl, gradPres, CurlMass = ExactSol.M2D(1,Show="yes", Mass = omega)
    else:
        mesh = UnitCubeMesh(int(nn),int(nn),int(nn))
        u0, p0, CurlCurl, gradPres, CurlMass = ExactSol.M3D(1,Show="yes", Mass = omega)

    order = 1
    parameters['reorder_dofs_serial'] = False
    Magnetic = FunctionSpace(mesh, "N1curl", order)
    Lagrange = FunctionSpace(mesh, "CG", order)
    VLagrange = VectorFunctionSpace(mesh, "CG", order)

    DimSave[xx-1] = VLagrange.dim()
    print VLagrange.dim()
    parameters['linear_algebra_backend'] = 'uBLAS'


    def boundary(x, on_boundary):
        return on_boundary

    (p) = TrialFunction(Lagrange)
    (q) = TestFunction(Lagrange)
    (Vp) = TrialFunction(VLagrange)
    (Vq) = TestFunction(VLagrange)


    ScalarLaplacian = assemble(inner(grad(p),grad(q))*dx)
    VectorLaplacian = assemble(inner(grad(Vp),grad(Vq))*dx+10*inner(Vp,Vq)*dx)

    bcVu = DirichletBC(VLagrange, Expression(("0.0","0.0")), boundary)
    bcu = DirichletBC(Lagrange, Expression(("0.0")), boundary)
    bcVu.apply(VectorLaplacian)
    bcu.apply(ScalarLaplacian)

    VectorLaplacian = PETSc.Mat().createAIJ(size=VectorLaplacian.sparray().shape,csr=(VectorLaplacian.sparray().indptr, VectorLaplacian.sparray().indices, VectorLaplacian.sparray().data))

    ScalarLaplacian = PETSc.Mat().createAIJ(size=ScalarLaplacian.sparray().shape,csr=(ScalarLaplacian.sparray().indptr, ScalarLaplacian.sparray().indices, ScalarLaplacian.sparray().data))

    x, b = VectorLaplacian.getVecs()
    x.set(0.0)
    b.set(1.0)

    ksp = PETSc.KSP().create()
    ksp.setTolerances(1e-6)
    ksp.setType('cg')
    ksp.setOperators(VectorLaplacian,VectorLaplacian)

    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    # pc.setPythonContext(HiptmairPrecond.Direct(G, Gt, P, Pt, VectorLaplacian, ScalarLaplacian))

    scale = b.norm()
    b = b/scale
    ksp.solve(b, x)
    x = x*scale
    print ksp.its
    ItsSave[xx-1,0] = ksp.its

    x, b = ScalarLaplacian.getVecs()
    x.set(0.0)
    b.set(1.0)

    ksp = PETSc.KSP().create()
    ksp.setTolerances(1e-6)
    ksp.setType('cg')
    ksp.setOperators(ScalarLaplacian,ScalarLaplacian)

    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)

    scale = b.norm()
    b = b/scale
    ksp.solve(b, x)
    x = x*scale
    print ksp.its
    ItsSave[xx-1,1] = ksp.its
    # xa = Function(Magnetic)
    # xa.vector()[:] = x.array

    # ue = u0
    # pe = p0
    # # parameters["form_compiler"]["quadrature_degree"] = 15

    # Ve = FunctionSpace(mesh,"N1curl",3)
    # u = interpolate(ue,Ve)





    # ErrorB = Function(Magnetic)
    # ErrorB = u-xa


    # errL2b[xx-1] = sqrt(abs(assemble(inner(ErrorB, ErrorB)*dx)))
    # errCurlb[xx-1] = sqrt(abs(assemble(inner(curl(ErrorB), curl(ErrorB))*dx)))


    # if xx == 1:
    #     a = 1
    # else:

    #     l2border[xx-1] =  numpy.abs(numpy.log2(errL2b[xx-2]/errL2b[xx-1]))
    #     Curlborder[xx-1] =  numpy.abs(numpy.log2(errCurlb[xx-2]/errCurlb[xx-1]))

    # print errL2b[xx-1]
    # print errCurlb[xx-1]

import pandas as pd

print DimSave
print ItsSave
# plot(xa)
# plot(u)

# LatexTitlesB = ["l","B DoF","BB-L2","B-order","BB-Curl","Curl-order"]
# LatexValuesB = numpy.concatenate((NN,DimSave,errL2b,l2border,errCurlb,Curlborder),axis=1)
# LatexTableB= pd.DataFrame(LatexValuesB, columns = LatexTitlesB)
# pd.set_option('precision',3)
# LatexTableB = MO.PandasFormat(LatexTableB,'BB-Curl',"%2.4e")
# LatexTableB = MO.PandasFormat(LatexTableB,'BB-L2',"%2.4e")
# LatexTableB = MO.PandasFormat(LatexTableB,'Curl-order',"%2.2f")
# LatexTableB = MO.PandasFormat(LatexTableB,'B-order',"%2.2f")
# print LatexTableB#.to_latex()


# print "\n\n\n"
# ItsTitlesB = ["l","B DoF","Time","Iterations"]
# ItsValuesB = numpy.concatenate((NN,DimSave,TimeSave,ItsSave),axis=1)
# ItsTableB= pd.DataFrame(ItsValuesB, columns = ItsTitlesB)
# pd.set_option('precision',5)
# print ItsTableB.to_latex()

interactive()

