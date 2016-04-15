
# import numpy

# from scipy.sparse import csr_matrix, dia_matrix
# from scipy.sparse.linalg import spsolve
# from scipy.io import mmread

# from pyamg.aggregation import smoothed_aggregation_solver
# from pyamg.multilevel import multilevel_solver
# from pyamg.relaxation.smoothing import change_smoothers
# from pyamg.relaxation.relaxation import make_system
# from pyamg.relaxation.relaxation import gauss_seidel
# from pyamg.krylov._cg import cg


# __all__ = ['edgeAMG']

# def hiptmair_smoother(A,x,b,D,iterations=1,sweep='symmetric'):
#     A,x,b = make_system(A,x,b,formats=['csr','bsr'])
#     gauss_seidel(A,x,b,iterations=1,sweep='forward')
#     r = b-A*x
#     x_G = numpy.zeros(D.shape[1])
#     A_G,x_G,b_G = make_system(D.T*A*D,x_G,D.T*r,formats=['csr','bsr'])
#     gauss_seidel(A_G,x_G,b_G,iterations=1,sweep='symmetric')
#     x[:] += D*x_G
#     gauss_seidel(A,x,b,iterations=1,sweep='backward')

# def setup_hiptmair(lvl,iterations=1,sweep='symmetric'):
#     D = lvl.D
#     def smoother(A,x,b):
#         hiptmair_smoother(A,x,b,D,iterations=iterations,sweep=sweep)
#     return smoother

# def edgeAMG(Anode,Acurl,D):
#     nodalAMG = smoothed_aggregation_solver(Anode,max_coarse=10,keep=True)


#     ##
#     # construct multilevel structure
#     levels = []
#     levels.append( multilevel_solver.level() )
#     levels[-1].A = Acurl
#     levels[-1].D = D
#     for i in range(1,len(nodalAMG.levels)):
#         A = levels[-1].A
#         Pnode = nodalAMG.levels[i-1].AggOp
#         P = findPEdge(D, Pnode)
#         R = P.T
#         levels[-1].P = P
#         levels[-1].R = R
#         levels.append( multilevel_solver.level() )
#         A = R*A*P
#         D = csr_matrix(dia_matrix((1.0/((P.T*P).diagonal()),0),shape=(P.shape[1],P.shape[1]))*(P.T*D*Pnode))
#         levels[-1].A = A
#         levels[-1].D = D

#     edgeML = multilevel_solver(levels)
#     for i in range(0,len(edgeML.levels)):
#         edgeML.levels[i].presmoother = setup_hiptmair(levels[i])
#         edgeML.levels[i].postsmoother = setup_hiptmair(levels[i])
#     return edgeML


# def findPEdge ( D, PNode):
#     ###
#     # use D to find edges
#     # each row has exactly two non zeros, a -1 marking the start node, and 1 marking the end node
#     numEdges = D.shape[0]
#     edges = numpy.zeros((numEdges,2))
#     DRowInd = D.nonzero()[0]
#     DColInd = D.nonzero()[1]
#     for i in range(0,numEdges):
#         if ( D[DRowInd[2*i],DColInd[2*i]] == -1.0 ):  # first index is start, second is end
#             edges[DRowInd[2*i],0] = DColInd[2*i]
#             edges[DRowInd[2*i],1] = DColInd[2*i+1]
#         else :  # first index is end, second is start
#             edges[DRowInd[2*i],0] = DColInd[2*i+1]
#             edges[DRowInd[2*i],1] = DColInd[2*i]

#     ###
#     # now that we have the edges, we need to find the nodal aggregates
#     # the nodal aggregates are the columns


#     aggs = PNode.nonzero()[1] # each row has 1 nonzero and that column is its aggregate
#     numCoarseEdges = 0
#     row = []
#     col = []
#     data = []
#     coarseEdges = {}
#     for i in range(0,edges.shape[0]):
#         coarseV1 = aggs[edges[i,0]]
#         coarseV2 = aggs[edges[i,1]]
#         if ( coarseV1 != coarseV2 ): # this is a coarse edges
#             #check if in dictionary
#             if ( coarseEdges.has_key((coarseV1,coarseV2)) ):
#                 row.append(i)
#                 col.append(coarseEdges[(coarseV1,coarseV2)])
#                 data.append(1)
#             elif ( coarseEdges.has_key((coarseV2,coarseV1))):
#                 row.append(i)
#                 col.append(coarseEdges[(coarseV2,coarseV1)])
#                 data.append(-1)
#             else :
#                 coarseEdges[(coarseV1,coarseV2)] = numCoarseEdges
#                 numCoarseEdges = numCoarseEdges + 1
#                 row.append(i)
#                 col.append(coarseEdges[(coarseV1,coarseV2)])
#                 data.append(1)

#     PEdge = csr_matrix( (data, (row,col) ),shape=(numEdges,numCoarseEdges) )
#     return PEdge


from dolfin import *
import numpy as np
import scipy.sparse as sp
import numpy
import matplotlib.pylab as plt
import scipy.io
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
from PyTrilinos import Epetra, ML, AztecOO, Teuchos
from dolfin import *
import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import matplotlib.pylab as plt
import PETScIO as IO
import numpy as np
import scipy.sparse as sparse
import CheckPetsc4py as CP
import scipy.sparse.linalg as sparselin
import scipy as sp
import time
from FIAT import *


m =3
ItsSave = np.zeros((m-1,1))
DimSave = np.zeros((m-1,1))


for xx in xrange(1,m):
    # pass
    nn = 2**(xx+3)
    mesh = UnitSquareMesh(nn,nn)
    # domain_vertices = [Point(0.0, 0.0),
    #                  Point(10.0, 0.0),
    #                  Point(10.0, 2.0),
    #                  Point(8.0, 2.0),
    #                  Point(7.5, 1.0),
    #                  Point(2.5, 1.0),
    #                  Point(2.0, 4.0),
    #                  Point(0.0, 4.0),
    #                  Point(0.0, 0.0)]

    # # Create empty Mesh
    # mesh = Mesh()
    # PolygonalMeshGenerator.generate(mesh, domain_vertices, 0.85/xx);
    # plot(mesh, interactive=True)
    # plot(mesh)
    order = 1
    parameters['reorder_dofs_serial'] = False

    Magnetic = FunctionSpace(mesh, "N1curl", order)
    Lagrange = FunctionSpace(mesh, "CG", order)
    # L= FunctionSpace(mesh, "DG", order-1)
    # build_edge2dof_map(Magnetic)
    DimSave[xx-1] = Magnetic.dim()
    print Magnetic.dim()
    parameters['linear_algebra_backend'] = 'uBLAS'
    b0 = Expression(("x[1]*x[1]*(x[1]-1)","x[0]*x[0]*(x[0]-1)"))

    # <codecell>

    V = FunctionSpace(mesh, "N1curl", order)
    Q = FunctionSpace(mesh, "CG", order)
    W = MixedFunctionSpace([V,Q])

    # <codecell>

    parameters['linear_algebra_backend'] = 'uBLAS'

    # <codecell>

    b0 = Expression(("x[1]*x[1]*(x[1]-1)","x[0]*x[0]*(x[0]-1)"))
    print V.dim(), Q.dim()

    # <codecell>

    def boundary(x, on_boundary):
        return on_boundary
    # bcb = DirichletBC(V,b0, boundary)
    # bc = [bcb]

    # <codecell>

    (v) = TrialFunction(V)
    (u) = TestFunction(V)
    (uMix,pMix) = TrialFunctions(W)
    (vMix,qMix) = TestFunctions(W)
    CurlCurl = Expression(("-6*x[1]+2","-6*x[0]+2"))+b0
    f = CurlCurl

    # <codecell>

    a = inner(curl(v),curl(u))*dx
    m = inner(u,v)*dx
    b = inner(vMix,grad(pMix))*dx

    # <codecell>

    A = assemble(a)
    M = assemble(m)
    Ms = M.sparray()
    A = A.sparray()

    # # <codecell>

    B = assemble(b)
    B = B.sparray()[:V.dim(),W.dim()-Q.dim():]
    plt.spy (B.todense())
    # plt.show()
    # # <codecell>

    ksp = PETSc.KSP().create()
    # parameters['linear_algebra_backend'] = 'PETSc'
    M = assemble(m)
    M = CP.Assemble(M)
    ksp.setOperators(M)
    x = M.getVecLeft()
    ksp.setFromOptions()
    ksp.setType(ksp.Type.CG)
    ksp.setTolerances(1e-6)
    ksp.pc.setType(ksp.pc.Type.BJACOBI)

    # <codecell>

    OptDB = PETSc.Options()
    # OptDB["pc_factor_mat_ordering_type"] = "rcm"
    # OptDB["pc_factor_mat_solver_package"] = "cholmod"
    ksp.setFromOptions()
    C = sparse.csr_matrix((V.dim(),Q.dim()))
    IO.matToSparse

    # <codecell>

    C = sparse.csr_matrix((V.dim(),Q.dim()))
    (v) = TrialFunction(V)
    (u) = TestFunction(V)
    tic()
    for i in range(0,Q.dim()):
        uOut = Function(V)
        uu = Function(Q)
        x = M.getVecRight()
        zero = np.zeros((Q.dim(),1))[:,0]
        zero[i] = 1
        uu.vector()[:] = zero
        L = assemble(inner(u, grad(uu))*dx)
        rhs = IO.arrayToVec(B[:,i].toarray())
        ksp.solve(rhs,x)
    #     x = project(grad(uu),V)
        P = x.array
        uOut.vector()[:] = P
        low_values_indices = np.abs(P) < 1e-3
        P[low_values_indices] = 0
        P=np.around(P)
        pn = P.nonzero()[0]
        for j in range(0,len(pn)):
            C[pn[j],i] = P[pn[j]]
        del uu
    print toc()
    print C.todense()
    # C = sparse.csr_matrix((Magnetic.dim(),Lagrange.dim()))
    # Mmap = Magnetic.dofmap()
    # Lmap = Lagrange.dofmap()
    # # Mapping = Lmap.vertex_to_dof_map(mesh)
    # Mapping = dof_to_vertex_map(Lagrange)
    # c = 0
    # edge2dof = numpy.zeros(mesh.num_edges(), dtype="int")
    # edge2dof2 = numpy.zeros(mesh.num_edges(), dtype="int")
    # tic()
    # for cell in cells(mesh):
    #     cellDOF = Mmap.cell_dofs(c)
    #     edgeVALUES = cell.entities(1)
    #     # print edgeVALUES, cellDOF
    #     edge2dof[edgeVALUES]=cellDOF
    #     c = c+1
    # C = cells(mesh)
    # # for i in range(mesh.num_cells()):
    # #     cell = Cell(mesh,i)
    # #     cellDOF = Mmap.cell_dofs(i)
    # #     edgeVALUES = cell.entities(1)
    # #     # print edgeVALUES, cellDOF
    # #     edge2dof2[cellDOF]=edgeVALUES

    # C = sparse.csr_matrix((Magnetic.dim(),Lagrange.dim()))
    # for vert in edges(mesh):
    #     j = 0
    #     for edge in vertices(vert):
    #         if (j == 0):
    #             # print edge.index()
    #             C[edge2dof[vert.index()],Mapping[edge.index()]] = -1
    #             # print edge.index()
    #         else:
    #             # print edge.index()
    #             # print edge.index()
    #             C[edge2dof[vert.index()],Mapping[edge.index()]] = 1
    #         j = j + 1
    # print "Python time: ", toc()


    from scipy.sparse import coo_matrix
    import os, inspect
    path = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), ".."))
    gradient_code = open(os.path.join(path, 'DiscreteGradient.cpp'), 'r').read()
    compiled_gradient_module = compile_extension_module(code=gradient_code)

    Lmap = Lagrange.dofmap()
    Mapping = dof_to_vertex_map(Lagrange)
    column =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
    row =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
    data =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")

    tic()
    c = compiled_gradient_module.gradient(Magnetic, Mapping.astype("intc"),column,row,data)
    print "C++ time:", toc()
    print row.shape
    print "\n\n\n"
    print c
    CC = coo_matrix((data,(column,row)), shape=(Magnetic.dim(),Lagrange.dim()))
    CC = CC.tocsr()
    print "Max:", (CC.todense() - C.todense()).max()
    # print C.todense()
    print (((A*CC).todense()).max(1)).max(0)
    # def boundary(x, on_boundary


    #     return on_boundary
    # bcb = DirichletBC(Magnetic,b0, boundary)

    # u = Function(Lagrange)
    # for i in range(0, Lagrange.dim() ):
    #     zero = np.zeros((Lagrange.dim(),1))[:,0]
    #     zero[i] = 1
    #     u.vector()[:] = zero

    #     uu = grad(u)
    #     Pv = project(uu,Magnetic,solver_type = "cg")
    #     P = Pv.vector().array()

    #     index = P.nonzero()
    #     index = index[0]
    #     for x in range(0,len(index)):
    #         if np.abs(P[index[x]]) < 1e-3:
    #             P[index[x]] = 0
    #     # print P.shape, C.shape
    #     pn = P.nonzero()[0]
    #     for j in range(0,len(pn)):
    #         C[pn[j],i] = P[pn[j]]
    #     del P


    # def boundary(x, on_boundary):
    #     return on_boundary


    c = 1
    (u) = TrialFunction(Magnetic)
    (v) = TestFunction(Magnetic)
    (p) = TrialFunction(Lagrange)
    (q) = TestFunction(Lagrange)

    a = inner(curl(u),curl(v))*dx + inner(u,v)*dx

    # A = assemble(a)
    # bcb.apply(A)
    # B = (A*C)
    # A = A.sparray()

    # plt.spy(C.todense())
    # plt.show()

    # <codecell>

    # print np.min(np.abs(B.toarray()))
    # print np.max(np.abs(B.toarray()))

    # <codecell>


    l = inner(grad(p),grad(q))*dx+inner(p,q)*dx
    # u0 = Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])","-sin(2*pi*x[0])*cos(2*pi*x[1])"))
    # f = 8*pow(pi,2)*u0+c*u0
    CurlCurl = Expression(("-6*x[1]+2","-6*x[0]+2"))+b0
    f = CurlCurl
    L1  = inner(v, f)*dx


    bc = DirichletBC(Magnetic,b0, boundary)

    # MLList = {

    #     "default values" : "maxwell",
    #     "max levels" : 10,
    #     "output" : 10,
    #     "PDE equations" : 1,
    #     "increasing or decreasing" : "decreasing",
    #     "aggregation: type" : "Uncoupled-MIS",
    #     "aggregation: damping factor" : 1.3333,
    #     "coarse: max size" : 75,
    #     "aggregation: threshold" : 0.0,
    #     "smoother: sweeps" : 2,
    #     "smoother: damping factor" : 0.67,
    #     "smoother: type" : "MLS",
    #     "smoother: MLS polynomial order" : 4,
    #     "smoother: pre or post" : "both",
    #     "coarse: type" : "Amesos-KLU",
    #     "prec type" : "MGV",
    #     "print unused" : -2
    # }
    # ML.SetDefaults("maxwell",List)
    # MLList = {
    #     "default values" : "maxwell",
    #     "max levels"                                     : 10,
    #     "prec type"                                        : "MGV",
    #     "increasing or decreasing"               : "decreasing",
    #     "aggregation: type"                          : "Uncoupled-MIS",
    #     "aggregation: damping factor"         : 4.0/3.0,
    #     "eigen-analysis: type"                      : "cg",
    #     "eigen-analysis: iterations"              : 10,
    #     "smoother: sweeps"                          : 3,
    #     "smoother: damping factor"              : 1.0,
    #     "smoother: pre or post"                     : "both",
    #     "smoother: type"                               : "Hiptmair",
    #     "subsmoother: type"                         : "Chebyshev",
    #     "subsmoother: Chebyshev alpha"    : 27.0,
    #     "subsmoother: node sweeps"           : 4,
    #     "subsmoother: edge sweeps"           : 4,
    #     "PDE equations" : 1,
    #     "coarse: type"                                   : "Amesos-MUMPS",
    #     "coarse: max size"                           : 25,
    #     "print unused" : 0

    # }

    MLList = Teuchos.ParameterList()
    ML.SetDefaults("maxwell",MLList)

    # MList.setParameters()
    MLList.set("default values","maxwell")
    MLList.set("max levels", 10)
    MLList.set("prec type", "MGV")
    MLList.set("increasing or decreasing", "decreasing")
    MLList.set("aggregation: type", "Uncoupled-MIS")
    MLList.set("aggregation: damping factor", 4.0/3.0)
    # MLList.set("eigen-analysis: type", "cg")
    # MLList.set("eigen-analysis: iterations", 10)
    MLList.set("smoother: sweeps", 5)
    MLList.set("smoother: damping factor", 1.0)
    MLList.set("smoother: pre or post", "both")
    MLList.set("smoother: type", "Hiptmair")
    MLList.set("subsmoother: type", "Chebyshev")
    MLList.set("aggregation: threshold", 0.0)
    MLList.set("subsmoother: Chebyshev alpha", 27.0)
    MLList.set("subsmoother: node sweeps", 10)
    MLList.set("subsmoother: edge sweeps", 10)
    MLList.set("PDE equations",1)
    MLList.set("coarse: type", "Amesos-MUMPS")
    MLList.set("coarse: max size", 25)
    MLList.set("print unused",2)
    MLList.set("viz: output format","vtk");
    MLList.set("viz: print starting solution", True);
    comm = Epetra.PyComm()


    parameters['linear_algebra_backend'] = 'Epetra'
    Mass = assemble(inner(u,v)*dx)
    bc.apply(Mass)
    Acurl,b = assemble_system(a,L1,bc)
    Anode = assemble(l)

    # scipy.io.savemat( "CurlCurl.mat", {"CurlCurl":Acurl.sparray()},oned_as='row')
    # scipy.io.savemat( "node.mat", {"node":Anode.sparray()},oned_as='row')
    # scipy.io.savemat( "rhs.mat", {"rhs":b.array()},oned_as='row')

    C = scipy_csr_matrix2CrsMatrix(C, comm)
    Acurl = as_backend_type(Acurl).mat()
    Mass = as_backend_type(Mass).mat()
    Anode = as_backend_type(Anode).mat()

    ML_Hiptmair = ML.MultiLevelPreconditioner(Acurl,C,Anode,MLList,False)
    # ML_Hiptmair = ML.MultiLevelPreconditioner(Acurl,False)
    ML_Hiptmair.ComputePreconditioner()
    x = Function(Magnetic)
    # print 'time to create preconditioner ', toc()
    # A_epetra = as_backend_type(AAA).mat()
    b_epetra = as_backend_type(b).vec()
    x_epetra = as_backend_type(x.vector()).vec()

    tic()
    #u = M.SolveSystem(A,b,V,"cg","amg",1e-6,1e-6,1)
    print toc()

    import PyTrilinos.AztecOO as AztecOO
    solver = AztecOO.AztecOO(Acurl, x_epetra, b_epetra)
    solver.SetPrecOperator(ML_Hiptmair)
    solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_cg);
    solver.SetAztecOption(AztecOO.AZ_output, 1);
    err = solver.Iterate(155000, 1e-5)
    ItsSave[xx-1] =solver.NumIters()






print ItsSave
print DimSave
# # Acurl = Acurl.sparray()
# # Anode = Anode.sparray()
# # D = C

# # x = numpy.random.rand(Acurl.shape[1],1)
# # b = Acurl*x
# # x0 = numpy.ones((Acurl.shape[1],1))



# # ml = edgeAMG(Anode,Acurl,D)
# # MLOp = ml.aspreconditioner()
# # x = numpy.random.rand(Acurl.shape[1],1)
# # b = Acurl*x
# # x0 = numpy.ones((Acurl.shape[1],1))

# # r_edgeAMG = []
# # r_None = []
# # r_SA = []

# # ml_SA = smoothed_aggregation_solver(Acurl)
# # ML_SAOP = ml_SA.aspreconditioner()
# # x_prec,info = cg(Acurl,b,x0,M=MLOp,tol=1e-8,residuals=r_edgeAMG)
# # x_prec,info = cg(Acurl,b,x0,M=None,tol=1e-8,residuals=r_None)
# # x_prec,info = cg(Acurl,b,x0,M=ML_SAOP,tol=1e-8,residuals=r_SA)

# # import pylab
# # pylab.semilogy(range(0,len(r_edgeAMG)), r_edgeAMG, range(0,len(r_None)), r_None, range(0,len(r_SA)), r_SA)
# # pylab.show()
