import os, inspect
from dolfin import *
import numpy
from scipy.sparse import coo_matrix
import ExactSol
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
from PyTrilinos import Epetra, ML, AztecOO, Teuchos
import MatrixOperations as MO
import matplotlib.pylab as plt

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
dim = 2

for xx in xrange(1,m):
    NN[xx-1] = xx+2
    nn = 2**(NN[xx-1][0])
    omega = 1
    if dim == 2:
        mesh = UnitSquareMesh(int(nn),int(nn))
        u0, p0, CurlCurl, gradPres, CurlMass = ExactSol.M2D(2,Show="yes", Mass = omega)
    else:
        mesh = UnitCubeMesh(int(nn),int(nn),int(nn))
        u0, p0, CurlCurl, gradPres, CurlMass = ExactSol.M3D(2,Show="yes", Mass = omega)

    order = 1
    parameters['reorder_dofs_serial'] = False
    Magnetic = FunctionSpace(mesh, "N1curl", order)
    Lagrange = FunctionSpace(mesh, "CG", order)

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

    C = coo_matrix((data,(column,row)), shape=(Magnetic.dim(),Lagrange.dim())).tocsr()


    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(Magnetic,u0, boundary)

    (v) = TestFunction(Magnetic)
    (u) = TrialFunction(Magnetic)
    (p) = TrialFunction(Lagrange)
    (q) = TestFunction(Lagrange)

    a = inner(curl(u),curl(v))*dx + omega*inner(u,v)*dx
    L1  = inner(v, CurlMass)*dx
    # parameters['linear_algebra_backend'] = 'Epetra'
    Acurl,b = assemble_system(a,L1,bc)
    Anode = assemble(inner(grad(p),grad(q))*dx+omega*inner(p,q)*dx)


    MLList = Teuchos.ParameterList()
    ML.SetDefaults("maxwell",MLList)
    MLList.set("default values","maxwell")
    MLList.set("max levels", 10)
    MLList.set("prec type", "MGV")
    MLList.set("increasing or decreasing", "decreasing")
    MLList.set("aggregation: type", "Uncoupled-MIS")
    MLList.set("aggregation: damping factor", 4.0/3.0)
    MLList.set("eigen-analysis: type", "cg")
    MLList.set("eigen-analysis: iterations", 10)
    MLList.set("smoother: sweeps", 5)
    MLList.set("smoother: damping factor", 1.0)
    MLList.set("smoother: pre or post", "both")
    MLList.set("smoother: type", "Hiptmair")
    MLList.set("subsmoother: type", "Chebyshev")
    MLList.set("aggregation: threshold", 0.2)
    MLList.set("subsmoother: Chebyshev alpha", 27.0)
    MLList.set("subsmoother: node sweeps", 10)
    MLList.set("subsmoother: edge sweeps", 5)
    MLList.set("PDE equations",1)
    MLList.set("coarse: type", "Amesos-MUMPS")
    # MLList.set("coarse: max size", 1)
    MLList.set("print unused",0)
    # MLList.set("viz: output format","vtk");
    # MLList.set("viz: print starting solution", True);


    comm = Epetra.PyComm()
    # MO.StoreMatrix(Acurl.sparray().tocsr(),"matrix/CurlCurl_n"+str(int(nn)))
    # MO.StoreMatrix(Anode.sparray().tocsr(),"matrix/NodalLaplacian_n"+str(int(nn)))
    # MO.StoreMatrix(C.tocsr(),"matrix/DiscreteGrad_n"+str(int(nn)))

    C = scipy_csr_matrix2CrsMatrix(C, comm)
    Acurl = scipy_csr_matrix2CrsMatrix(Acurl.sparray(), comm)
    Anode = scipy_csr_matrix2CrsMatrix(Anode.sparray(), comm)
    print "CurlCurl_n"+str(int(nn))
    ML_Hiptmair = ML.MultiLevelPreconditioner(Acurl,C,Anode,MLList,False)
    # ML_Hiptmair = ML.MultiLevelPreconditioner(Acurl,False)
    ML_Hiptmair.ComputePreconditioner()
    x = Function(Magnetic)

    b_epetra = Epetra.Vector(b.array())
    x_epetra = Epetra.Vector(0*b.array())
    MO.StoreMatrix(b.array(),"matrix/RHS_n"+str(int(nn)))
    scale = b_epetra.Norm2()
    b_epetra =b_epetra/scale

    import PyTrilinos.AztecOO as AztecOO
    solver = AztecOO.AztecOO(Acurl, x_epetra, b_epetra)
    solver.SetPrecOperator(ML_Hiptmair)
    solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_cg_condnum );
    solver.SetAztecOption(AztecOO.AZ_output, 1);
    err = solver.Iterate(155000, 1e-5)
    ItsSave[xx-1] = solver.NumIters()
    TimeSave[xx-1] = solver.SolveTime()

    x = Function(Magnetic)
    x.vector()[:] = x_epetra.array*scale
    # plot(x,interactive = True)

#     ue = u0
#     pe = p0
#     # parameters["form_compiler"]["quadrature_degree"] = 15


#     Ve = FunctionSpace(mesh,"N1curl",3)
#     u = interpolate(ue,Ve)





#     ErrorB = Function(Magnetic)
#     ErrorB = u-x


#     errL2b[xx-1] = sqrt(abs(assemble(inner(ErrorB, ErrorB)*dx)))
#     errCurlb[xx-1] = sqrt(abs(assemble(inner(curl(ErrorB), curl(ErrorB))*dx)))


#     if xx == 1:
#         a = 1
#     else:

#         l2border[xx-1] =  numpy.abs(numpy.log2(errL2b[xx-2]/errL2b[xx-1]))
#         Curlborder[xx-1] =  numpy.abs(numpy.log2(errCurlb[xx-2]/errCurlb[xx-1]))

#     print errL2b[xx-1]
#     print errCurlb[xx-1]

import pandas as pd


# LatexTitlesB = ["l","B DoF","BB-L2","B-order","BB-Curl","Curl-order"]
# LatexValuesB = numpy.concatenate((NN,DimSave,errL2b,l2border,errCurlb,Curlborder),axis=1)
# LatexTableB= pd.DataFrame(LatexValuesB, columns = LatexTitlesB)
# pd.set_option('precision',3)
# LatexTableB = MO.PandasFormat(LatexTableB,'BB-Curl',"%2.4e")
# LatexTableB = MO.PandasFormat(LatexTableB,'BB-L2',"%2.4e")
# LatexTableB = MO.PandasFormat(LatexTableB,'Curl-order',"%2.2f")
# LatexTableB = MO.PandasFormat(LatexTableB,'B-order',"%2.2f")
# print LatexTableB#.to_latex()


print "\n\n\n"
ItsTitlesB = ["l","B DoF","Time","Iterations"]
ItsValuesB = numpy.concatenate((NN,DimSave,TimeSave,ItsSave),axis=1)
ItsTableB= pd.DataFrame(ItsValuesB, columns = ItsTitlesB)
pd.set_option('precision',5)
print ItsTableB.to_latex()



