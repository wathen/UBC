import os, inspect
from dolfin import *
import numpy
from scipy.sparse import coo_matrix, block_diag, hstack, tril, spdiags, csr_matrix
import ExactSol
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
from PyTrilinos import Epetra, ML, AztecOO, Teuchos
import MatrixOperations as MO
import matplotlib.pylab as plt
import CheckPetsc4py as CP
import petsc4py
import sys
import HiptmairPrecond
import HiptmairApply
petsc4py.init(sys.argv)

from petsc4py import PETSc

path = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), ".."))
gradient_code = open(os.path.join(path, 'DiscreteGradient.cpp'), 'r').read()
compiled_gradient_module = compile_extension_module(code=gradient_code)


m = 8
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
dim = 3

for xx in xrange(1,m):
    NN[xx-1] = xx+0
    nn = int(2**(NN[xx-1][0]))
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
    VLagrange = VectorFunctionSpace(mesh, "CG", order)
    parameters['reorder_dofs_serial'] = False

    W = Magnetic*Lagrange
    DimSave[xx-1] = Magnetic.dim()
    print Magnetic.dim()
    parameters['linear_algebra_backend'] = 'uBLAS'


    column =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
    row =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
    data =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")

    # Mapping = dof_to_vertex_map(Lagrange)
    # c = compiled_gradient_module.Gradient(Magnetic, Mapping.astype("intc"),column,row,data)
    dataX =  numpy.zeros(2*mesh.num_edges(), order="C")
    dataY =  numpy.zeros(2*mesh.num_edges(), order="C")
    dataZ =  numpy.zeros(2*mesh.num_edges(), order="C")

    tic()
    c = compiled_gradient_module.ProlongationGradsecond(mesh, dataX,dataY,dataZ, data, row, column)
    print "C++ time:", toc()

    C = coo_matrix((data,(row,column)), shape=(Magnetic.dim(),Lagrange.dim())).tocsr()
    # AA = Matrix()
    # AA.set(data.astype("float_"),row.astype("intc"),column.astype("intc"))
    # print AA.sparray().todense()
    Px = coo_matrix((dataX,(row,column)), shape=(Magnetic.dim(),Lagrange.dim())).tocsr()
    Py = coo_matrix((dataY,(row,column)), shape=(Magnetic.dim(),Lagrange.dim())).tocsr()
    Pz = coo_matrix((dataZ,(row,column)), shape=(Magnetic.dim(),Lagrange.dim())).tocsr()

    Px.eliminate_zeros()
    Py.eliminate_zeros()
    Pz.eliminate_zeros()

    if Magnetic.dim() == 8001:

        VertexDoF = numpy.sin(numpy.linspace(0.0, 2*numpy.pi, num=mesh.num_vertices()))
        EdgeDoFX =  Px*VertexDoF
        EdgeDoFY =  Py*VertexDoF
        EEX = Function(Magnetic)
        EEY = Function(Magnetic)
        VV = Function(Lagrange)
        VV.vector()[:] = VertexDoF
        EEX.vector()[:] = EdgeDoFX
        EEY.vector()[:] = EdgeDoFY

        # file = File("Plots/MagneticXdirection_"+str(Magnetic.dim())+".pvd")
        # file << EEX
        # file = File("Plots/MagneticYdirection_"+str(Magnetic.dim())+".pvd")
        # file << EEY
        # file = File("Plots/Nodal_"+str(Magnetic.dim())+".pvd")
        # file << VV

        plot(EEX,tite="Magnetic interpolation X-direction")
        plot(EEY,tite="Magnetic interpolation Y-direction")
        plot(VV,tite="Nodal represetation")



    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(Magnetic,u0, boundary)
    bcu = DirichletBC(Lagrange, Expression(("0.0")), boundary)

    if dim == 3:
        bcW = DirichletBC(W.sub(0),Expression(("1.0","1.0","1.0")), boundary)
    else:
        bcW = DirichletBC(W.sub(0),Expression(("1.0","1.0")), boundary)

    bcuW = DirichletBC(W.sub(1), Expression(("1.0")), boundary)

    (v) = TestFunction(Magnetic)
    (u) = TrialFunction(Magnetic)

    (p) = TrialFunction(Lagrange)
    (q) = TestFunction(Lagrange)

    (Vp) = TrialFunction(VLagrange)
    (Vq) = TestFunction(VLagrange)

    Wv,Wq=TestFunctions(W)
    Wu,Wp=TrialFunctions(W)

    NS_is = PETSc.IS().createGeneral(range(Magnetic.dim()))
    M_is = PETSc.IS().createGeneral(range(Magnetic.dim()+Lagrange.dim(),W.dim()))
    # AA = assemble(inner(grad(Wp),Wv)*dx+inner(grad(Wq),Wu)*dx)

    tic()
    AA = Function(W).vector()
    AA[:] = 0
    # AA.zero()
   #  plt.subplot(1, 3, 1)
   #  plt.spy(AA.sparray())
    bcW.apply(AA)
   #  plt.subplot(1, 3, 2)
   #  plt.spy(AA.sparray())
   #  # plt.spy(AA.sparray())
   # # # plt.figure()
   # #  # # AA.zero()
    bcuW.apply(AA)
   #  plt.subplot(1, 3, 3)
   #  plt.spy(AA.sparray())
    # plt.show()


    # aa
    diagAA = AA.array()
    print toc()
    # print diagAAdd
    # diagAA = AA.diagonal()
    BClagrange = numpy.abs(diagAA[Magnetic.dim():]) > 1e-3
    BCmagnetic = numpy.abs(diagAA[:Magnetic.dim()]) > 1e-3
    onelagrange = numpy.ones(Lagrange.dim())
    onemagnetic = numpy.ones(Magnetic.dim())
    onelagrange[BClagrange] = 0
    onemagnetic[BCmagnetic] = 0
    Diaglagrange = spdiags(onelagrange,0,Lagrange.dim(),Lagrange.dim())
    Diagmagnetic = spdiags(onemagnetic,0,Magnetic.dim(),Magnetic.dim())
    # plt.spy(C*Diag)
    # plt.figure()
    plt.spy(Diagmagnetic)
    # plt.show()

    tic()
    C = Diagmagnetic*C*Diaglagrange
    print "BC applied to gradient, time: ", toc()
    # print C
    # BC = ~numpy.array(BC)
    # tic
    # CC = C
    # ii,jj = C[:,BC].nonzero()
    # C[ii,jj] = 0
    # print (CC-C).todense()
    # # print C
    # Curl = assemble(inner(curl(u),curl(v))*dx)
    # Mass = assemble(inner(u,v)*dx)
    # Grad = assemble(inner(grad(p),v)*dx)
    # Laplacian = assemble(inner(grad(q),grad(p))*dx)
    # # plt.spy(C)
    # # plt.show()
    # bc.apply(Curl)
    # bc.apply(Mass)
    # bcu.apply(Laplacian)



    # print  "========================================"
    # Curlgrad[xx-1]=(Curl.sparray()*C).max()
    # print " Curl-gradient test: ", Curlgrad[xx-1]
    # Massgrad[xx-1]=(Mass.sparray()*C-Diagmagnetic*Grad.sparray()*Diaglagrange).max()
    # print " Mass-gradient test: ", Massgrad[xx-1]
    # Laplgrad[xx-1]=(Diaglagrange*Grad.sparray().transpose()*Diagmagnetic*C-Laplacian.sparray()).max()
    # print " Lapl-gradient test: ", Laplgrad[xx-1]
    # print "========================================"
    # # # Grad = AA.

    # Grad.zero()
    # bcu.apply(Grad)


    G = PETSc.Mat().createAIJ(size=C.shape,csr=(C.indptr, C.indices, C.data))

    a = inner(curl(u),curl(v))*dx + inner(u,v)*dx
    L1  = inner(v, CurlMass)*dx
    tic()
    Acurl,b = assemble_system(a,L1,bc)
    print "System assembled, time: ", toc()
    tic()
    A,b = CP.Assemble(Acurl,b)
    x = b.duplicate()
    print "PETSc system assembled, time: ", toc()

    tic()
    ScalarLaplacian = assemble(inner(grad(p),grad(q))*dx)
    VectorLaplacian = assemble(inner(grad(p),grad(q))*dx+inner(p,q)*dx)
    print "Hiptmair Laplacians assembled, time: ", toc()
    # tic()
    # Pxx = Px
    # Pyy = Py
    # ii,jj = Px[:,BC].nonzero()
    # Px[ii,jj] = 0
    # ii,jj = Px[:,BC].nonzero()
    # Py[ii,jj] = 0
    # print toc()

    # print (Px).todense()
    # print (Pxx).todense()
    if dim == 2:
        tic()
        Px = Diagmagnetic*Px*Diaglagrange
        Py = Diagmagnetic*Py*Diaglagrange
        print "BC applied to Prolongation, time: ", toc()
        # P = hstack([Px,Py]).tocsr()
        bcVu = DirichletBC(VLagrange, Expression(("0.0","0.0")), boundary)
        P = [PETSc.Mat().createAIJ(size=Px.shape,csr=(Px.indptr, Px.indices, Px.data)),PETSc.Mat().createAIJ(size=Py.shape,csr=(Py.indptr, Py.indices, Py.data))]
    else:
        tic()
        Px = Diagmagnetic*Px*Diaglagrange
        Py = Diagmagnetic*Py*Diaglagrange
        Pz = Diagmagnetic*Pz*Diaglagrange
        print "BC applied to Prolongation, time: ", toc()
        bcVu = DirichletBC(VLagrange, Expression(("0.0","0.0","0.0")), boundary)
        P = [PETSc.Mat().createAIJ(size=Px.shape,csr=(Px.indptr, Px.indices, Px.data)),PETSc.Mat().createAIJ(size=Py.shape,csr=(Py.indptr, Py.indices, Py.data)),PETSc.Mat().createAIJ(size=Pz.shape,csr=(Pz.indptr, Pz.indices, Pz.data))]
    tic()
    bcu.apply(VectorLaplacian)
    bcu.apply(ScalarLaplacian)
    print "Hiptmair Laplacians BC applied, time: ", toc()
    # Pt = P.transpose().tocsr()

    # Pt = PETSc.Mat().createAIJ(size=Pt.shape,csr=(Pt.indptr, Pt.indices, Pt.data))
    # P = PETSc.Mat().createAIJ(size=P.shape,csr=(P.indptr, P.indices, P.data))

    VectorLaplacian = PETSc.Mat().createAIJ(size=VectorLaplacian.sparray().shape,csr=(VectorLaplacian.sparray().indptr, VectorLaplacian.sparray().indices, VectorLaplacian.sparray().data))
    ScalarLaplacian = PETSc.Mat().createAIJ(size=ScalarLaplacian.sparray().shape,csr=(ScalarLaplacian.sparray().indptr, ScalarLaplacian.sparray().indices, ScalarLaplacian.sparray().data))




    # CurlCurl = assemble(inner(curl(u),curl(v))*dx+inner(u,v)*dx)
    # # bc.apply(CurlCurl)
    # CurlCurlPetsc = CP.Assemble(CurlCurl)
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
    pc.setPythonContext(HiptmairPrecond.GSvector(G, P, VectorLaplacian, ScalarLaplacian, Lower, Upper))
    scale = b.norm()
    b = b/scale
    tic()
    ksp.solve(b, x)
    TimeSave[xx-1] = toc()
    x = x*scale
    print ksp.its
    print TimeSave[xx-1]
    ItsSave[xx-1] = ksp.its
    xa = Function(Magnetic)
    # x = b.duplicate()
    # tic()
    # # x, i = HiptmairApply.cg(A, b, x,VectorLaplacian ,ScalarLaplacian , P, G, Magnetic, bc, 1000)
    # print toc()
    # x = x*scale
    xa.vector()[:]  = x.array

    ue = u0
    pe = p0
    # parameters["form_compiler"]["quadrature_degree"] = 15

    Ve = FunctionSpace(mesh,"N1curl",2)
    u = interpolate(ue,Ve)

    print "======================", i

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
    print " \n\n\n\n"

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

GradTests = ["l","B DoF","$AC$","$MC-B^T$","$BC-L$"]
GradValues = numpy.concatenate((NN,DimSave,Curlgrad, Massgrad, Laplgrad),axis=1)
GradTab= pd.DataFrame(GradValues, columns = GradTests)
pd.set_option('precision',3)
GradTab = MO.PandasFormat(GradTab,'$AC$',"%2.4e")
GradTab = MO.PandasFormat(GradTab,'$MC-B^T$',"%2.4e")
GradTab = MO.PandasFormat(GradTab,'$BC-L$',"%2.4e")
print GradTab.to_latex()


# Curlgrad, Massgrad, Laplgrad
