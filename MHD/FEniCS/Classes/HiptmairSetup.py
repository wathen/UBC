import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy
from dolfin import compile_extension_module, tic, toc, DirichletBC, Expression, TestFunctions, TrialFunctions, Function, BoundaryMesh, Face, __version__
import os, inspect
from scipy.sparse import coo_matrix, csr_matrix, spdiags
import HiptmairPrecond
import time
import MatrixOperations as MO

def HiptmairMatrixSetup(mesh, N, M):

    path = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), ".."))
    if __version__ == '1.6.0':
        gradient_code = open(os.path.join(path, 'DiscreteGradientSecond.cpp'), 'r').read()
    else:
        gradient_code = open(os.path.join(path, 'DiscreteGradient.cpp'), 'r').read()
    compiled_gradient_module = compile_extension_module(code=gradient_code)

    column =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
    row =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
    data =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")

    dataX =  numpy.zeros(2*mesh.num_edges(), order="C")
    dataY =  numpy.zeros(2*mesh.num_edges(), order="C")
    dataZ =  numpy.zeros(2*mesh.num_edges(), order="C")

    tic()
    c = compiled_gradient_module.ProlongationGradsecond(mesh, dataX,dataY,dataZ, data, row, column)
    end = toc()
    MO.StrTimePrint("Data for C and P created, time: ",end)
    # print row
    # print column
    # print  data
    C = csr_matrix((data,(row,column)), shape=(N, M)).tocsr()
    Px = csr_matrix((dataX,(row,column)), shape=(N, M)).tocsr()
    Py = csr_matrix((dataY,(row,column)), shape=(N, M)).tocsr()
    Pz = csr_matrix((dataZ,(row,column)), shape=(N, M)).tocsr()
    return C, [Px,Py,Pz]




def BoundaryEdge(mesh):
    path = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), ".."))
    # if __version__ == '1.6.0':
    gradient_code = open(os.path.join(path, 'DiscreteGradientSecond.cpp'), 'r').read()
    # else:
    #     gradient_code = open(os.path.join(path, 'DiscreteGradient.cpp'), 'r').read()
    compiled_gradient_module = compile_extension_module(code=gradient_code)
    B = BoundaryMesh(mesh,"exterior",False)
    FaceBoundary = numpy.sort(B.entity_map(2).array().astype("float_","C"))
    EdgeBoundary =  numpy.zeros(3*FaceBoundary.size, order="C")
    c = compiled_gradient_module.FaceToEdgeBoundary(mesh, FaceBoundary, FaceBoundary.size, EdgeBoundary)
    return EdgeBoundary #numpy.sort(EdgeBoundary)[::2].astype("float_","C")

def Boun(mesh):
    bdry_cells = BoundaryMesh(mesh, 'exterior').entity_map(2).array()
# Init face-edge connection
    mesh.init(2, 1)
    # Boundary edges of mesh
    bdry_edges = set(numpy.concatenate([Face(mesh, f).entities(1) for f in bdry_cells]))
    bdry_edges = list(bdry_edges)
    return bdry_edges

def HiptmairMatrixSetupBoundary(mesh, N, M,dim):
    def boundary(x, on_boundary):
        return on_boundary

    # mesh.geometry().dim()
    path = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), ".."))
    # if __version__ == '1.6.0':
    gradient_code = open(os.path.join(path, 'DiscreteGradientSecond.cpp'), 'r').read()
    # else:
        # gradient_code = open(os.path.join(path, 'DiscreteGradient.cpp'), 'r').read()
    compiled_gradient_module = compile_extension_module(code=gradient_code)
    tic()
    if dim == 3:
        EdgeBoundary = BoundaryEdge(mesh)
        EdgeBoundary = numpy.sort(EdgeBoundary)[::2].astype("float_","C")
    else:
        B = BoundaryMesh(mesh,"exterior",False)
        EdgeBoundary = numpy.sort(B.entity_map(1).array().astype("float_","C"))
    end = toc()
    MO.StrTimePrint("Compute edge boundary indices, time: ",end)


    row =  numpy.zeros(2*(mesh.num_edges()-EdgeBoundary.size), order="C") #, dtype="intc")
    column =  numpy.zeros(2*(mesh.num_edges()-EdgeBoundary.size), order="C") #, dtype="intc")
    data =  numpy.zeros(2*(mesh.num_edges()-EdgeBoundary.size), order="C") #, dtype="intc")

    dataX =  numpy.zeros(2*(mesh.num_edges()-EdgeBoundary.size), order="C")
    dataY =  numpy.zeros(2*(mesh.num_edges()-EdgeBoundary.size), order="C")
    dataZ =  numpy.zeros(2*(mesh.num_edges()-EdgeBoundary.size), order="C")
    # print 2*(mesh.num_edges()-EdgeBoundary.size)

    tic()

    # c = compiled_gradient_module.ProlongationGrad(mesh, EdgeBoundary, dataX,dataY,dataZ, data, row, column)
    c = compiled_gradient_module.ProlongationGradBoundary(mesh, EdgeBoundary, dataX,dataY,dataZ, data, row, column)
    # u, indices = numpy.unique(row, return_index=True)
    # indices = numpy.concatenate((indices,indices+1),axis=1)
    # # print VertexBoundary
    # print row
    # # print numpy.concatenate((indices,indices+1),axis=1)
    # # print  data
    # row = row[indices]
    # column = column[indices]
    # data = data[indices]
    # print row
    end = toc()
    MO.StrTimePrint("Data for C and P created, time: ",end)
    C = coo_matrix((data,(row,column)), shape=(N, M)).tocsr()
    Px = coo_matrix((dataX,(row,column)), shape=(N, M)).tocsr()
    Py = coo_matrix((dataY,(row,column)), shape=(N, M)).tocsr()
    Pz = coo_matrix((dataZ,(row,column)), shape=(N, M)).tocsr()
    del dataX, dataY, dataZ, row,column, mesh, EdgeBoundary
    return C, [Px,Py,Pz]



def HiptmairBCsetupBoundary(C, P, mesh):

    dim = mesh.geometry().dim()
    tic()
    B = BoundaryMesh(mesh,"exterior",False)
    NodalBoundary = B.entity_map(0).array()#.astype("int","C")
    onelagrange = numpy.ones(mesh.num_vertices())
    onelagrange[NodalBoundary] = 0
    Diaglagrange = spdiags(onelagrange,0,mesh.num_vertices(),mesh.num_vertices())
    end = toc()
    MO.StrTimePrint("Work out boundary matrices, time: ",end)

    del mesh
    tic()
    C = C*Diaglagrange
    G = PETSc.Mat().createAIJ(size=C.shape,csr=(C.indptr, C.indices, C.data))
    end = toc()
    MO.StrTimePrint("BC applied to gradient, time: ",end)

    if dim == 2:
        tic()
        Px = P[0]*Diaglagrange
        Py = P[1]*Diaglagrange
        end = toc()
        MO.StrTimePrint("BC applied to Prolongation, time: ",end)
        P = [PETSc.Mat().createAIJ(size=Px.shape,csr=(Px.indptr, Px.indices, Px.data)),PETSc.Mat().createAIJ(size=Py.shape,csr=(Py.indptr, Py.indices, Py.data))]
    else:
        tic()
        Px = P[0]*Diaglagrange
        Py = P[1]*Diaglagrange
        Pz = P[2]*Diaglagrange
        end = toc()
        MO.StrTimePrint("BC applied to Prolongation, time: ",end)
        P = [PETSc.Mat().createAIJ(size=Px.shape,csr=(Px.indptr, Px.indices, Px.data)),PETSc.Mat().createAIJ(size=Py.shape,csr=(Py.indptr, Py.indices, Py.data)),PETSc.Mat().createAIJ(size=Pz.shape,csr=(Pz.indptr, Pz.indices, Pz.data))]
    del Px, Py, Diaglagrange
    return  G, P


def HiptmairBCsetup(C, P, mesh, Func):
    tic()
    W = Func[0]*Func[1]
    def boundary(x, on_boundary):
        return on_boundary
    bcW = DirichletBC(W.sub(0), Expression(("1.0","1.0","1.0")), boundary)

    bcuW = DirichletBC(W.sub(1), Expression(("1.0")), boundary)
    # Wv,Wq=TestFunctions(W)
    # Wu,Wp=TrialFunctions(W)


    dim = mesh.geometry().dim()
    tic()
    if dim == 3:
        EdgeBoundary = BoundaryEdge(mesh)
    else:
        B = BoundaryMesh(Magnetic.mesh(),"exterior",False)
        EdgeBoundary = numpy.sort(B.entity_map(1).array().astype("int","C"))


    B = BoundaryMesh(mesh,"exterior",False)
    NodalBoundary = B.entity_map(0).array()#.astype("int","C")
    onelagrange = numpy.ones(mesh.num_vertices())
    onelagrange[NodalBoundary] = 0
    Diaglagrange = spdiags(onelagrange,0,mesh.num_vertices(),mesh.num_vertices())

    onemagnetiic = numpy.ones(mesh.num_edges())
    onemagnetiic[EdgeBoundary.astype("int","C")] = 0
    Diagmagnetic = spdiags(onemagnetiic,0,mesh.num_edges(),mesh.num_edges())


    MO.StrTimePrint("Work out boundary matrices, time: ",toc())

    tic()
    C = Diagmagnetic*C*Diaglagrange
    G = PETSc.Mat().createAIJ(size=C.shape,csr=(C.indptr, C.indices, C.data))
    MO.StrTimePrint("BC applied to gradient, time: ",toc())

    if dim == 2:
        tic()
        Px = Diagmagnetic*P[0]*Diaglagrange
        Py = Diagmagnetic*P[1]*Diaglagrange
        MO.StrTimePrint("BC applied to Prolongation, time: ",toc())
        P = [PETSc.Mat().createAIJ(size=Px.shape,csr=(Px.indptr, Px.indices, Px.data)),PETSc.Mat().createAIJ(size=Py.shape,csr=(Py.indptr, Py.indices, Py.data))]
    else:
        tic()
        Px = Diagmagnetic*P[0]*Diaglagrange
        Py = Diagmagnetic*P[1]*Diaglagrange
        Pz = Diagmagnetic*P[2]*Diaglagrange
        MO.StrTimePrint("BC applied to Prolongation, time: ",toc())
        P = [PETSc.Mat().createAIJ(size=Px.shape,csr=(Px.indptr, Px.indices, Px.data)),PETSc.Mat().createAIJ(size=Py.shape,csr=(Py.indptr, Py.indices, Py.data)),PETSc.Mat().createAIJ(size=Pz.shape,csr=(Pz.indptr, Pz.indices, Pz.data))]

    return  G, P


def HiptmairKSPsetup(VectorLaplacian, ScalarLaplacian, A, tol):
    OptDB = PETSc.Options()
    OptDB['pc_hypre_type'] = 'boomeramg'
    OptDB['pc_hypre_boomeramg_strong_threshold']  = 0.5
    OptDB['pc_hypre_boomeramg_grid_sweeps_all']  = 1
    # OptDB['pc_hypre_boomeramg_cycle_type']  = "W"

    kspVector = PETSc.KSP()
    kspVector.create(comm=PETSc.COMM_WORLD)
    pcVector = kspVector.getPC()
    kspVector.setType('preonly')
    pcVector.setType('hypre')
    kspVector.setFromOptions()

    kspScalar = PETSc.KSP()
    kspScalar.create(comm=PETSc.COMM_WORLD)
    pcScalar = kspScalar.getPC()
    kspScalar.setType('preonly')
    pcScalar.setType('hypre')
    kspScalar.setFromOptions()

    kspCGScalar = PETSc.KSP()
    kspCGScalar.create(comm=PETSc.COMM_WORLD)
    pcCGScalar = kspCGScalar.getPC()
    kspCGScalar.setType('preonly')
    pcCGScalar.setType('hypre')
    kspCGScalar.setTolerances(tol)
    kspCGScalar.setFromOptions()


    kspVector.setOperators(VectorLaplacian,VectorLaplacian)
    kspScalar.setOperators(ScalarLaplacian,ScalarLaplacian)
    kspCGScalar.setOperators(ScalarLaplacian,ScalarLaplacian)

    diag = A.getDiagonal()
    diag.reciprocal()

    return kspVector, kspScalar, kspCGScalar, diag





def HiptmairApply(A, b, kspVector, kspScalar, G, P,tol):
    x = b.duplicate()
    ksp = PETSc.KSP().create()
    ksp.setTolerances(tol)
    ksp.setType('cg')
    ksp.setOperators(A,A)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    diag = A.getDiagonal()
    diag.reciprocal()

    pc.setPythonContext(HiptmairPrecond.GSvector(G, P, kspVector, kspScalar, diag))
    scale = b.norm()
    b = b/scale
    tic()
    ksp.solve(b, x)
    time = toc()
    # print "Hiptmair, its  ", ksp.its," time ", toc()
    x = x*scale
    return x, ksp.its, time
