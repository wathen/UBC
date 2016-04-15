

from dolfin import *
import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import matplotlib.pylab as plt
import PETScIO as IO
import numpy as np
import scipy.sparse as sp


nn = 4
mesh = RectangleMesh(0, 0, 1, 1, nn, nn,'left')

order  = 2
parameters['reorder_dofs_serial'] = False
V = FunctionSpace(mesh, "N2curl", order)
Q = FunctionSpace(mesh, "CG", order)
W = MixedFunctionSpace([V,Q])

print V.dim(), Q.dim()

parameters['linear_algebra_backend'] = 'PETSc'


b0 = Expression(("x[1]*x[1]*(x[1]-1)","x[0]*x[0]*(x[0]-1)"))


def boundary(x, on_boundary):
    return on_boundary
bcb = DirichletBC(V,b0, boundary)
bc = [bcb]


(u) = TrialFunction(V)
(v) = TestFunction(V)
(uMix,pMix) = TrialFunctions(W)
(vMix,qMix) = TestFunctions(W)
CurlCurl = Expression(("-6*x[1]+2","-6*x[0]+2"))+b0
f = CurlCurl


a = inner(curl(v),curl(u))*dx
m = inner(u,v)*dx
b = inner(vMix,grad(pMix))*dx



A = assemble(a)
M = assemble(m)
bcb.apply(A)
bcb.apply(M)
M = as_backend_type(M).mat()


parameters['linear_algebra_backend'] = 'uBLAS'
B = assemble(b)
B = B.sparray()[W.dim()-V.dim():,W.dim()-Q.dim():]


ksp = PETSc.KSP().create()
ksp.setOperators(M)
x = M.getVecLeft()
ksp.setFromOptions()
ksp.setType(ksp.Type.PREONLY)


ksp.pc.setType(ksp.pc.Type.CHOLESKY)




OptDB = PETSc.Options()
OptDB["pc_factor_mat_ordering_type"] = "rcm"
# OptDB["pc_factor_mat_solver_package"] = "mumps"
ksp.setFromOptions()
C = sp.csr_matrix((V.dim(),Q.dim()))


for i in range(0,Q.dim()):
# for i in range(0,4):
    print i
    x = M.getVecLeft()
    rhs = PETSc.Vec().create()
    rhs.createWithArray(B[:,i].toarray())
    tic()
    ksp.solve(rhs,x)
    print toc()
    P = x.array
    low_values_indices = P < 1e-4
    P[low_values_indices] = 0
    pn = P.nonzero()[0]
    for j in range(0,len(pn)):
        C[pn[j],i] = P[pn[j]]


B = (A.array()*C)

print np.min(np.abs(B))
print np.max(np.abs(B))