import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
from dolfin import *

# import matplotlib.pylab as plt
import PETScIO as IO
import numpy as np
import scipy.sparse as sparse
import CheckPetsc4py as CP
import scipy.sparse.linalg as sparselin
import scipy as sp
import scipy.io

mm = 2
for x in xrange(1,mm):
    nn = 2**(x)/2
    print nn
    mesh = UnitSquareMesh(nn,nn)
    print "Num edges   ", mesh.num_edges()
    print "Num vertices", mesh.num_vertices()
    print "Num faces", mesh.num_faces()
    order = 2
    parameters['reorder_dofs_serial'] = False
    V = FunctionSpace(mesh, "N1curl", order)
    Q = FunctionSpace(mesh, "CG", order)
    (v) = TrialFunction(V)
    (u) = TestFunction(V)


    m = inner(u,v)*dx
    # parameters['linear_algebra_backend'] = 'PETSc'
    M = assemble(m)
    M = CP.Assemble(M)

    ksp = PETSc.KSP().create()
    ksp.setOperators(M)
    x = M.getVecLeft()
    ksp.setFromOptions()
    ksp.setType(ksp.Type.PREONLY)
    # ksp.setTolerances(1e-1)
    ksp.pc.setType(ksp.pc.Type.LU)


    OptDB = PETSc.Options()
    ksp.setFromOptions()
    C = sparse.csr_matrix((V.dim(),Q.dim()))

    tic()
    for i in range(0,Q.dim()):
        uOut = Function(V)
        uu = Function(Q)
        x = M.getVecRight()
        zero = np.zeros((Q.dim(),1))[:,0]
        zero[i] = 1
        uu.vector()[:] = zero
        L = assemble(inner(u, grad(uu))*dx)
        # bcb.apply(L)
        rhs = IO.arrayToVec(L.array())
        ksp.solve(rhs,x)
    #     x = project(grad(uu),V)
        P = x.array
        uOut.vector()[:] = P
        low_values_indices = np.abs(P) < 1e-3
        P[low_values_indices] = 0
        #P=np.around(P)
        pn = P.nonzero()[0]
        for j in range(0,len(pn)):
            C[pn[j],i] = P[pn[j]]
        del uu
    print toc()

    pathToGrad = "/home/mwathen/Dropbox/MastersResearch/MHD/FEniCS/GradMatrices/"
    name = pathToGrad+"UnitSquareCrossed_m="+str(nn)
    scipy.io.savemat( 'name', {"C":C},oned_as='row')
