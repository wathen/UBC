from dolfin import *
import HiptmairSetup
import memory_profiler
@profile
def foo():
    NN = 2*4
    parameters["form_compiler"]["quadrature_degree"] = -1
    nn = int(2**(NN))
    omega = 1
    mesh = UnitSquareMesh(nn,nn)
    order = 1
    parameters['reorder_dofs_serial'] = False
    Magnetic = FunctionSpace(mesh, "N1curl", order)
    Lagrange = FunctionSpace(mesh, "CG", order)
    W = Magnetic*Lagrange
    Magnetic = None
    #del Magnetic, Lagrange 
    Vdim = W.sub(0).dim()
    Qdim = W.sub(1).dim()
    Wdim = W.dim()
    print "\n\nV:  ",Vdim,"Q:  ",Qdim,"W:  ",Wdim,"\n\n"
    parameters['reorder_dofs_serial'] = False

    parameters['linear_algebra_backend'] = 'uBLAS'
    dim = 2
    #@print_memory
    G, P = HiptmairSetup.HiptmairMatrixSetupBoundary(mesh, W.sub(0).dim(), W.sub(1).dim(),dim)
    G, P = HiptmairSetup.HiptmairBCsetupBoundary(G,P,mesh)
    a = 1

foo()
