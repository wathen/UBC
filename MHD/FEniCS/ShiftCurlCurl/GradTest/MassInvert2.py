from dolfin import *
# import petsc4py, sys
# petsc4py.init(sys.argv)
# from petsc4py import PETSc
import matplotlib.pylab as plt

import PETScIO as IO
import numpy as np
import scipy.sparse as sparse
import CheckPetsc4py as CP
import scipy.sparse.linalg as sparselin
import scipy as sp
import time
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
from PyTrilinos import Epetra, ML, AztecOO
import TrilinosIO

nn = 16
mesh = RectangleMesh(0, 0, 1, 1, nn, nn,'left')

order  = 1
parameters['reorder_dofs_serial'] = False
V = FunctionSpace(mesh, "N2curl", order)
Q = FunctionSpace(mesh, "CG", order)
W = MixedFunctionSpace([V,Q])

b0 = Expression(("x[1]*x[1]*(x[1]-1)","x[0]*x[0]*(x[0]-1)"))
print V.dim(), Q.dim()

def boundary(x, on_boundary):
    return on_boundary
bcb = DirichletBC(V,b0, boundary)
bc = [bcb]

(v) = TrialFunction(V)
(u) = TestFunction(V)
(uMix,pMix) = TrialFunctions(W)
(vMix,qMix) = TestFunctions(W)
CurlCurl = Expression(("-6*x[1]+2","-6*x[0]+2"))+b0
f = CurlCurl



m = inner(u,v)*dx
b = inner(vMix,grad(pMix))*dx
parameters['linear_algebra_backend'] = 'uBLAS'
M = assemble(m)
# bcb.apply(A)
# bcb.apply(M)
M = CP.Assemble(M)

# B = assemble(b)
# B = B.sparray()[W.dim()-V.dim():,W.dim()-Q.dim():]

# ksp = PETSc.KSP().create()

# ksp.setOperators(M)
# x = M.getVecLeft()
# ksp.setFromOptions()
# ksp.setType(ksp.Type.CG)
# ksp.setTolerances(1e-1)
# ksp.pc.setType(ksp.pc.Type.BJACOBI)


# OptDB = PETSc.Options()
# # OptDB["pc_factor_mat_ordering_type"] = "rcm"
# # OptDB["pc_factor_mat_solver_package"] = "cholmod"
# ksp.setFromOptions()
# C = sparse.csr_matrix((V.dim(),Q.dim()))

# C = sparse.csr_matrix((V.dim(),Q.dim()))
(v) = TrialFunction(V)
(u) = TestFunction(V)
# tic()
# for i in range(0,Q.dim()):
#     uOut = Function(V)
#     uu = Function(Q)
#     x = M.getVecRight()
#     zero = np.zeros((Q.dim(),1))[:,0]
#     zero[i] = 1
#     uu.vector()[:] = zero
#     L = assemble(inner(u, grad(uu))*dx)
#     # bcb.apply(L)
#     rhs = IO.arrayToVec(L.array())
#     ksp.solve(rhs,x)
# #     x = project(grad(uu),V)
#     P = x.array
#     uOut.vector()[:] = P
#     low_values_indices = np.abs(P) < 1e-3
#     P[low_values_indices] = 0
#     P=np.around(P)
#     pn = P.nonzero()[0]
#     for j in range(0,len(pn)):
#         C[pn[j],i] = P[pn[j]]
#     del uu
# print toc()
import scipy.io
name = '../../GradMatrices/UnitSquareLeft_m='+str(nn)+'.mat'
mat = scipy.io.loadmat(name)
C = mat['C']
C = C.tocsr()
# print C.shape
parameters['linear_algebra_backend'] = 'uBLAS'
A = assemble(inner(curl(u), curl(v))*dx)
print np.min(np.abs((A.sparray()*C).toarray()))
print np.max(np.abs((A.sparray()*C).toarray()))

# parameters['linear_algebra_backend'] = 'Epetra'


# <codecell>
(p) = TrialFunction(Q)
(q) = TestFunction(Q)
l = (inner(grad(p),grad(q))*dx) +inner(q, p)*dx
L1  = inner(v, f)*dx

a = inner(curl(v),curl(u))*dx+inner(u, v)*dx
bc = DirichletBC(V,b0, boundary)



MLList = {
  "default values":"maxwell",
  "max levels":10,
  "prec type":"MGV",
  "increasing or decreasing":"decreasing",

  "aggregation: type":"Uncoupled-MIS",
  "aggregation: damping factor":1.333,
  "eigen-analysis: type":"cg",
  "eigen-analysis: iterations":10,
  "aggregation: edge prolongator drop threshold":0.0,

  "smoother: sweeps":1,
  "smoother: damping factor":1.0,
  "smoother: pre or post":"both",
  # "smoother: type":"Hiptmair",
  "smoother: Hiptmair efficient symmetric":True,
  "subsmoother: type": "Chebyshev",
  "subsmoother: Chebyshev alpha": 27.0,
  "subsmoother: node sweeps":4,
  "subsmoother: edge sweeps":4,

  "coarse: type":"Amesos-KLU",
  "coarse: max size":128,
  "coarse: pre or post":"post",
  "coarse: sweeps":1

}

comm = Epetra.PyComm()



Acurl,b = assemble_system(a,L1,bc)
Anode = assemble(l)


scipy.io.savemat( "System.mat", {"CurlCurl":Acurl.sparray(),"node":Anode.sparray(),"C":C,"rhs":b.array()},oned_as='row')
scipy.io.savemat( "node.mat", {"node":Anode.sparray()},oned_as='row')
scipy.io.savemat( "rhs.mat", {"rhs":b.array()},oned_as='row')

C = scipy_csr_matrix2CrsMatrix(C, comm)
Acurl = scipy_csr_matrix2CrsMatrix(Acurl.sparray(), comm)
Anode = scipy_csr_matrix2CrsMatrix(Anode.sparray(), comm)
# Acurl = as_backend_type(Acurl).mat()
# Anode = as_backend_type(Anode).mat()

ML_Hiptmair = ML.MultiLevelPreconditioner(Acurl,C,Anode,MLList,True)
ML_Hiptmair.ComputePreconditioner()
x = Function(V)

b_epetra = x_epetra = TrilinosIO._numpyToTrilinosVector(b.array())
x_epetra = TrilinosIO._numpyToTrilinosVector(x.vector().array())

tic()
#u = M.SolveSystem(A,b,V,"cg","amg",1e-6,1e-6,1)
print toc()

import PyTrilinos.AztecOO as AztecOO
solver = AztecOO.AztecOO(Acurl, x_epetra, b_epetra)
solver.SetPrecOperator(ML_Hiptmair)
solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_cg);
solver.SetAztecOption(AztecOO.AZ_output, 50);
err = solver.Iterate(155000, 1e-10)

x = Function(V)
x.vector()[:] = TrilinosIO._trilinosToNumpyVector(x_epetra)
