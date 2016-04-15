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
import HiptmairPrecond

petsc4py.init(sys.argv)

from petsc4py import PETSc

path = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), ".."))
gradient_code = open(os.path.join(path, 'DiscreteGradient.cpp'), 'r').read()
compiled_gradient_module = compile_extension_module(code=gradient_code)


m = 5

errL2b =numpy.zeros((m-1,1))
errCurlb =numpy.zeros((m-1,1))

l2border =  numpy.zeros((m-1,1))
Curlborder =numpy.zeros((m-1,1))

ItsSave = numpy.zeros((m-1,1))
DimSave = numpy.zeros((m-1,1))
TimeSave = numpy.zeros((m-1,1))
NN = numpy.zeros((m-1,1))
dim = 3

for xx in xrange(1,m):
    NN[xx-1] = xx+0
    nn = 2**(NN[xx-1][0])
    omega = 1
    if dim == 2:
        mesh = UnitSquareMesh(int(nn),int(nn))
        # mesh =  RectangleMesh(0,0, 1, 1, int(nn), int(nn),'crossed')
        u0, p0, CurlCurl, gradPres, CurlMass = ExactSol.M2D(1,Show="yes", Mass = omega)
    else:
        mesh = UnitCubeMesh(int(nn),int(nn),int(nn))
        u0, p0, CurlCurl, gradPres, CurlMass = ExactSol.M3D(1,Show="yes", Mass = omega)

    order = 1
    parameters['reorder_dofs_serial'] = False
    Magnetic = FunctionSpace(mesh, "N1curl", order)
    Lagrange = FunctionSpace(mesh, "CG", order)
    VLagrange = VectorFunctionSpace(mesh, "CG", order)

    DimSave[xx-1] = Magnetic.dim()
    print Magnetic.dim()
    parameters['linear_algebra_backend'] = 'uBLAS'


    column =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
    row =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
    data =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")

    Mapping = dof_to_vertex_map(Lagrange)
    tic()
    c = compiled_gradient_module.Gradient(Magnetic, Mapping.astype("intc"),column,row,data)
    print "C++ time:", toc()
    dataX =  numpy.zeros(2*mesh.num_edges(), order="C")
    dataY =  numpy.zeros(2*mesh.num_edges(), order="C")
    dataZ =  numpy.zeros(2*mesh.num_edges(), order="C")

    X = mesh.coordinates()[:,0].astype("float_")
    Y = mesh.coordinates()[:,1].astype("float_")
    if dim == 3:
        Z = mesh.coordinates()[:,2].astype("float_")
    else:
        Z = X
    c = compiled_gradient_module.ProlongationP(Magnetic,Mapping.astype("intc"),X,Y,Z,dataX,dataY,dataZ)


    C = coo_matrix((data,(column,row)), shape=(Magnetic.dim(),Lagrange.dim())).tocsr()
    Ct = C.transpose().tocsr()
    # print Ct.shape
    # print dataX
    Cx = C.dot(X)
    Cy = C.dot(Y)
    if dim == 3:
        Cz = C.dot(Z)
    else:
        Cz = Cx
    # print mesh.coordinates()[:,1].shape
    # Cz = C*mesh.coordinates()[:,2]

    Gt = PETSc.Mat().createAIJ(size=Ct.shape,csr=(Ct.indptr, Ct.indices, Ct.data))
    G = PETSc.Mat().createAIJ(size=C.shape,csr=(C.indptr, C.indices, C.data))
     #, dtype="intc")
    dataXX =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
    dataYY =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
    dataZZ =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
    c = compiled_gradient_module.Prolongation(Magnetic, Cx, Cy, Cz, dataXX, dataYY, dataZZ)
    # print coo_matrix((dataXX,(column,row)), shape=(Magnetic.dim(),Lagrange.dim())).todense()

    # print dataX
    # print dataY
    # # print column
    # P = coo_matrix((numpy.concatenate((dataX,dataY,dataZ),axis=0),(numpy.concatenate((column,column,column),axis=0),numpy.concatenate((row,row+Lagrange.dim(), row + 2*Lagrange.dim()),axis=0))), shape=(Magnetic.dim(),3*Lagrange.dim())).tocsr()
    # P = coo_matrix((numpy.concatenate((dataX,dataY),axis=0),(numpy.concatenate((column,column),axis=0),numpy.concatenate((row,row+Lagrange.dim()),axis=0))), shape=(Magnetic.dim(),2*Lagrange.dim())).tocsr()

    Px = coo_matrix((dataX,(column,row)), shape=(Magnetic.dim(),Lagrange.dim())).tocsr()
    Py = coo_matrix((dataY,(column,row)), shape=(Magnetic.dim(),Lagrange.dim())).tocsr()
    Pz = coo_matrix((dataZ,(column,row)), shape=(Magnetic.dim(),Lagrange.dim())).tocsr()
    print " \n\n\n\n"
    print  "X: ",(Px-coo_matrix((dataXX,(column,row)), shape=(Magnetic.dim(),Lagrange.dim())).tocsr()).max()
    # # print coo_matrix((dataYY,(column,row)), shape=(Magnetic.dim(),Lagrange.dim())).tocsr
    print "Y: ",(Py-coo_matrix((dataYY,(column,row)), shape=(Magnetic.dim(),Lagrange.dim())).tocsr()).max()
    print " \n\n\n\n"
    c = compiled_gradient_module.ProlongationGrad(Magnetic, Cx, Cy, Cz, dataXX, dataYY, dataZZ, data, row, column)
    #plt.spy(coo_matrix((dataYY,(column,row)), shape=(Magnetic.dim(),Lagrange.dim())))
    #plt.figure()
    #plt.spy(Py)
    # plt.show()


    # Pz = coo_matrix((dataZ,(column,row)), shape=(Magnetic.dim(),Lagrange.dim())).tocsr()
    # print Px.shape
    # X = numpy.sin( numpy.linspace(0, 2*numpy.pi, 3*len(mesh.coordinates()[:,0])))
    # plt.plot(X)
    # plt.plot(P*X)
    # plt.show()

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(Magnetic,u0, boundary)

    (v) = TestFunction(Magnetic)
    (u) = TrialFunction(Magnetic)
    (p) = TrialFunction(Lagrange)
    (q) = TestFunction(Lagrange)
    (Vp) = TrialFunction(VLagrange)
    (Vq) = TestFunction(VLagrange)
    Curl = assemble(inner(curl(u),curl(v))*dx).sparray()
    print  "=========================="
    print " Curl-gradient test: ", (Curl*C).max()
    print "=========================="

    a = inner(curl(u),curl(v))*dx + omega*inner(u,v)*dx
    L1  = inner(v, CurlMass)*dx
    # parameters['linear_algebra_backend'] = 'Epetra'
    Acurl,b = assemble_system(a,L1,bc)

    ScalarLaplacian = assemble(inner(grad(p),grad(q))*dx)
    VectorLaplacian = assemble(inner(grad(Vp),grad(Vq))*dx+inner(Vp,Vq)*dx)
    if dim == 2:
        # VectorLaplacian = block_diag((VectorLaplacian,VectorLaplacian)).tocsr()
        bcVu = DirichletBC(VLagrange, Expression(("0.0","0.0")), boundary)
        P = hstack([Px,Py]).tocsr()
    else:
        # VectorLaplacian = block_diag((VectorLaplacian,VectorLaplacian,VectorLaplacian)).tocsr()
        P = hstack([Px,Py,Pz]).tocsr()
        bcVu = DirichletBC(VLagrange, Expression(("0.0","0.0","0.0")), boundary)

    bcu = DirichletBC(Lagrange, Expression(("0.0")), boundary)

    # bcVu.apply(VectorLaplacian)
    # bcu.apply(ScalarLaplacian)
    Pt = P.transpose().tocsr()
    # print P.todense()
    # print C.todense()
    # print dataX
    Pt = PETSc.Mat().createAIJ(size=Pt.shape,csr=(Pt.indptr, Pt.indices, Pt.data))
    P = PETSc.Mat().createAIJ(size=P.shape,csr=(P.indptr, P.indices, P.data))
    # print VectorLaplacian
    VectorLaplacian = PETSc.Mat().createAIJ(size=VectorLaplacian.sparray().shape,csr=(VectorLaplacian.sparray().indptr, VectorLaplacian.sparray().indices, VectorLaplacian.sparray().data))

    ScalarLaplacian = PETSc.Mat().createAIJ(size=ScalarLaplacian.sparray().shape,csr=(ScalarLaplacian.sparray().indptr, ScalarLaplacian.sparray().indices, ScalarLaplacian.sparray().data))



    A,b = CP.Assemble(Acurl,b)
    x = b.duplicate()

    CurlCurl = assemble(a)
    CurlCurlPetsc = CP.Assemble(CurlCurl)
    ksp = PETSc.KSP().create()
    ksp.setTolerances(1e-6)
    ksp.setType('cg')
    ksp.setOperators(A,A)

    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    # pc.setPythonContext(HiptmairPrecond.Direct(G, Gt, P, Pt, VectorLaplacian, ScalarLaplacian))
    Lower = tril(Acurl.sparray()).tocsr()
    Upper = Lower.transpose().tocsr()
    Lower = PETSc.Mat().createAIJ(size=Lower.shape,csr=(Lower.indptr, Lower.indices, Lower.data))
    Upper = PETSc.Mat().createAIJ(size=Upper.shape,csr=(Upper.indptr, Upper.indices, Upper.data))
    pc.setPythonContext(HiptmairPrecond.GS(G, Gt, P, Pt, VectorLaplacian, ScalarLaplacian, Lower, Upper))
    scale = b.norm()
    b = b/scale
    ksp.solve(b, x)
    x = x*scale
    print ksp.its
    ItsSave[xx-1] = ksp.its
    xa = Function(Magnetic)
    xa.vector()[:] = x.array

    ue = u0
    pe = p0
    # parameters["form_compiler"]["quadrature_degree"] = 15

    Ve = FunctionSpace(mesh,"N1curl",3)
    u = interpolate(ue,Ve)





    ErrorB = Function(Magnetic)
    ErrorB = u-xa


    errL2b[xx-1] = sqrt(abs(assemble(inner(ErrorB, ErrorB)*dx)))
    errCurlb[xx-1] = sqrt(abs(assemble(inner(curl(ErrorB), curl(ErrorB))*dx)))


    if xx == 1:
        a = 1
    else:

        l2border[xx-1] =  numpy.abs(numpy.log2(errL2b[xx-2]/errL2b[xx-1]))
        Curlborder[xx-1] =  numpy.abs(numpy.log2(errCurlb[xx-2]/errCurlb[xx-1]))

    print errL2b[xx-1]
    print errCurlb[xx-1]

import pandas as pd

# plot(xa)
# plot(u)

LatexTitlesB = ["l","B DoF","BB-L2","B-order","BB-Curl","Curl-order"]
LatexValuesB = numpy.concatenate((NN,DimSave,errL2b,l2border,errCurlb,Curlborder),axis=1)
LatexTableB= pd.DataFrame(LatexValuesB, columns = LatexTitlesB)
pd.set_option('precision',3)
LatexTableB = MO.PandasFormat(LatexTableB,'BB-Curl',"%2.4e")
LatexTableB = MO.PandasFormat(LatexTableB,'BB-L2',"%2.4e")
LatexTableB = MO.PandasFormat(LatexTableB,'Curl-order',"%2.2f")
LatexTableB = MO.PandasFormat(LatexTableB,'B-order',"%2.2f")
print LatexTableB#.to_latex()


print "\n\n\n"
ItsTitlesB = ["l","B DoF","Time","Iterations"]
ItsValuesB = numpy.concatenate((NN,DimSave,TimeSave,ItsSave),axis=1)
ItsTableB= pd.DataFrame(ItsValuesB, columns = ItsTitlesB)
pd.set_option('precision',5)
print ItsTableB.to_latex()

interactive()

