import os, inspect
from dolfin import *
import numpy
from scipy.sparse import coo_matrix
path = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), ".."))
gradient_code = open(os.path.join(path, 'DiscreteGradient.cpp'), 'r').read()
compiled_gradient_module = compile_extension_module(code=gradient_code)

nn = 2**1
mesh = RectangleMesh(0, 0, 1, 1, nn, nn,'left')
order = 1
Magnetic = FunctionSpace(mesh, "N1curl", order)
Lagrange = FunctionSpace(mesh, "CG", order)

column =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
row =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
data =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
Mapping = dof_to_vertex_map(Lagrange)
Pcolumn =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
Prow =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
PdataX =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")
PdataY =  numpy.zeros(2*mesh.num_edges(), order="C") #, dtype="intc")


tic()
c = compiled_gradient_module.gradient(Magnetic, Mapping.astype('intc'),column,row,data)
print toc()

C = coo_matrix((data,(row,column)), shape=(Lagrange.dim(),Magnetic.dim()))

Cx = C*mesh.coordinates()[:,1]
Cy = C*mesh.coordinates()[:,2]
# try {
# // check function works
# }

B = assemble(b)
B = B.sparray()[W.dim()-V.dim():,W.dim()-Q.dim():]

# # <codecell>

ksp = PETSc.KSP().create()
parameters['linear_algebra_backend'] = 'PETSc'
M = assemble(m)
M = CP.Assemble(M)
ksp.setOperators(M)
x = M.getVecLeft()
ksp.setFromOptions()
ksp.setType(ksp.Type.CG)
ksp.setTolerances(1e-2)
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
    rhs = IO.arrayToVec(L.array())
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