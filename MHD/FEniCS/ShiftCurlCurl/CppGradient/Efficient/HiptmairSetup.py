import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy
from dolfin import *
import os, inspect
from scipy.sparse import coo_matrix, spdiags
import time
import CheckPetsc4py as CP
import PETScIO as IO


def HiptmairMatrixSetup(mesh, N, M):

    path = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), ".."))
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
    print ("{:40}").format("Data for C and P created, time: "), " ==>  ",("{:4f}").format(end)
    # print row
    # print column
    # print  data
    C = coo_matrix((data,(row,column)), shape=(N, M)).tocsr()
    Px = coo_matrix((dataX,(row,column)), shape=(N, M)).tocsr()
    Py = coo_matrix((dataY,(row,column)), shape=(N, M)).tocsr()
    Pz = coo_matrix((dataZ,(row,column)), shape=(N, M)).tocsr()
    return C, [Px,Py,Pz]




def BoundaryEdge(mesh):
    path = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), ".."))
    gradient_code = open(os.path.join(path, 'DiscreteGradient.cpp'), 'r').read()
    compiled_gradient_module = compile_extension_module(code=gradient_code)
    B = BoundaryMesh(mesh,"exterior",False)
    FaceBoundary = numpy.sort(B.entity_map(2).array().astype("float_","C"))
    EdgeBoundary =  numpy.zeros(3*FaceBoundary.size, order="C")
    # Sasasdtime = time.time()
    # c = compiled_gradient_module.FaceToEdge(mesh, FaceBoundary, EdgeBoundary)
    # print time.time()-Sasasdtime

    c = compiled_gradient_module.FaceToEdgeBoundary(mesh, FaceBoundary, FaceBoundary.size, EdgeBoundary)
    return EdgeBoundary #numpy.sort(EdgeBoundary)[::2].astype("float_","C")



def HiptmairMatrixSetupBoundary(mesh, N, M,dim):
    def boundary(x, on_boundary):
        return on_boundary

    # mesh.geometry().dim()
    path = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), ".."))
    gradient_code = open(os.path.join(path, 'DiscreteGradient.cpp'), 'r').read()
    compiled_gradient_module = compile_extension_module(code=gradient_code)
    tic()
    if dim == 3:
        EdgeBoundary = BoundaryEdge(mesh)
        EdgeBoundary = numpy.sort(EdgeBoundary)[::2].astype("float_","C")
    else:
        B = BoundaryMesh(mesh,"exterior",False)
        EdgeBoundary = numpy.sort(B.entity_map(1).array().astype("float_","C"))
    end = toc()
    print ("{:40}").format("Compute edge boundary indices, time: "), " ==>  ",("{:4f}").format(end)


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
    print ("{:40}").format("Data for C and P created, time: "), " ==>  ",("{:4f}").format(end)
    C = coo_matrix((data,(row,column)), shape=(N, M)).tocsr()
    Px = coo_matrix((dataX,(row,column)), shape=(N, M)).tocsr()
    Py = coo_matrix((dataY,(row,column)), shape=(N, M)).tocsr()
    Pz = coo_matrix((dataZ,(row,column)), shape=(N, M)).tocsr()

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
    print ("{:40}").format("Work out boundary matrices, time: "), " ==>  ",("{:4f}").format(end)

    tic()
    C = C*Diaglagrange
    G = PETSc.Mat().createAIJ(size=C.shape,csr=(C.indptr, C.indices, C.data))
    end = toc()
    print ("{:40}").format("BC applied to gradient, time: "), " ==>  ",("{:4f}").format(end)

    if dim == 2:
        tic()
        Px = P[0]*Diaglagrange
        Py = P[1]*Diaglagrange
        end = toc()
        print ("{:40}").format("BC applied to Prolongation, time: "), " ==>  ",("{:4f}").format(end)
        P = [PETSc.Mat().createAIJ(size=Px.shape,csr=(Px.indptr, Px.indices, Px.data)),PETSc.Mat().createAIJ(size=Py.shape,csr=(Py.indptr, Py.indices, Py.data))]
    else:
        tic()
        Px = P[0]*Diaglagrange
        Py = P[1]*Diaglagrange
        Pz = P[2]*Diaglagrange
        end = toc()
        print ("{:40}").format("BC applied to Prolongation, time: "), " ==>  ",("{:4f}").format(end)
        P = [PETSc.Mat().createAIJ(size=Px.shape,csr=(Px.indptr, Px.indices, Px.data)),PETSc.Mat().createAIJ(size=Py.shape,csr=(Py.indptr, Py.indices, Py.data)),PETSc.Mat().createAIJ(size=Pz.shape,csr=(Pz.indptr, Pz.indices, Pz.data))]

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


    print ("{:40}").format("Work out boundary matrices, time: "), " ==>  ",("{:4f}").format(toc())

    tic()
    C = Diagmagnetic*C*Diaglagrange
    G = PETSc.Mat().createAIJ(size=C.shape,csr=(C.indptr, C.indices, C.data))
    print ("{:40}").format("BC applied to gradient, time: "), " ==>  ",("{:4f}").format(toc())

    if dim == 2:
        tic()
        Px = Diagmagnetic*P[0]*Diaglagrange
        Py = Diagmagnetic*P[1]*Diaglagrange
        print ("{:40}").format("BC applied to Prolongation, time: "), " ==>  ",("{:4f}").format(toc())
        P = [PETSc.Mat().createAIJ(size=Px.shape,csr=(Px.indptr, Px.indices, Px.data)),PETSc.Mat().createAIJ(size=Py.shape,csr=(Py.indptr, Py.indices, Py.data))]
    else:
        tic()
        Px = Diagmagnetic*P[0]*Diaglagrange
        Py = Diagmagnetic*P[1]*Diaglagrange
        Pz = Diagmagnetic*P[2]*Diaglagrange
        print ("{:40}").format("BC applied to Prolongation, time: "), " ==>  ",("{:4f}").format(toc())
        P = [PETSc.Mat().createAIJ(size=Px.shape,csr=(Px.indptr, Px.indices, Px.data)),PETSc.Mat().createAIJ(size=Py.shape,csr=(Py.indptr, Py.indices, Py.data)),PETSc.Mat().createAIJ(size=Pz.shape,csr=(Pz.indptr, Pz.indices, Pz.data))]

    return  G, P



def HiptmairAnyOrder(Magnetic,Lagrange):
    mesh = Magnetic.mesh()
    VecLagrange = VectorFunctionSpace(mesh, "CG", Magnetic.__dict__['_FunctionSpace___degree'])

    def boundary(x, on_boundary):
        return on_boundary

    dim = mesh.geometry().dim()
    u0 = []
    for i in range(dim):
        u0.append('0.0')
    u0 = Expression(u0)
    VecBC = DirichletBC(VecLagrange, u0, boundary)
    BCb = DirichletBC(Magnetic, u0, boundary)
    BCr = DirichletBC(Lagrange, Expression(('0.0')), boundary)

    p = TestFunction(Lagrange)
    q = TrialFunction(Lagrange)
    u = TestFunction(Magnetic)
    v = TrialFunction(Magnetic)
    Vu = TestFunction(VecLagrange)
    Vv = TrialFunction(VecLagrange)

    M = assemble(inner(u,v)*dx)
    BCb.apply(M)
    B = assemble(inner(v,grad(p))*dx)
    L = assemble(inner(grad(Vu),grad(Vv))*dx + inner(Vu,Vv)*dx)
    l = assemble(inner(grad(p),grad(q))*dx)
    VecBC.apply(L)
    BCr.apply(l)
    L = CP.Scipy2PETSc(L.sparray())
    B = CP.Scipy2PETSc(B.sparray())
    M = CP.Scipy2PETSc(M.sparray())
    l = CP.Scipy2PETSc(l.sparray())

    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    pc = ksp.getPC()
    ksp.setType('cg')
    pc.setType('bjacobi')
    ksp.setOperators(M,M)

    return VecLagrange, ksp, L, l, B, [BCb, BCr, VecBC]


def HiptmairKSPsetup(VectorLaplacian, ScalarLaplacian, A):
    OptDB = PETSc.Options()
    OptDB['pc_hypre_type'] = 'boomeramg'
    OptDB['pc_hypre_boomeramg_strong_threshold']  = 0.5
    OptDB['pc_hypre_boomeramg_grid_sweeps_all']  = 1

    kspVector = PETSc.KSP()
    kspVector.create(comm=PETSc.COMM_WORLD)
    pcVector = kspVector.getPC()
    kspVector.setType('preonly')
    pcVector.setType('hypre')
    kspVector.max_it = 1
    kspVector.setFromOptions()

    kspScalar = PETSc.KSP()
    kspScalar.create(comm=PETSc.COMM_WORLD)
    pcScalar = kspScalar.getPC()
    kspScalar.setType('preonly')
    pcScalar.setType('hypre')
    kspScalar.setFromOptions()


    kspVector.setOperators(VectorLaplacian,VectorLaplacian)
    kspScalar.setOperators(ScalarLaplacian,ScalarLaplacian)

    diag = A.getDiagonal()
    diag.reciprocal()

    return kspVector, kspScalar, diag


def GradOp(ksp,B,u):
    Bu = B.createVecRight()
    B.multTranspose(u,Bu)
    v = Bu.duplicate()
    ksp.solve(Bu,v)
    return v

def TransGradOp(ksp,B,u):
    Bu = u.duplicate()
    ksp.solve(u,Bu)
    v = B.createVecLeft()
    B.mult(Bu,v)
    return v

def BCapply(V,BC,x,opt = "PETSc"):
    v = Function(V)
    v.vector()[:] = x.array
    BC.apply(v.vector())
    if opt == "PETSc":
        x = IO.arrayToVec(v.vector().array())
        return x
    else:
        return v
    
def PETScToFunc(V,x):
    v = Function(V)
    v.vector()[:] = x.array
    return x   

def FuncToPETSc(x):
    return IO.arrayToVec(x.vector().array())





