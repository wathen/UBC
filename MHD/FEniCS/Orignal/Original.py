
from dolfin import *
from numpy import *
import scipy as Sci
import scipy.linalg
from math import pi,sin,cos,sqrt
import scipy.sparse as sps
import scipy.io as save
import scipy
import pdb


parameters['linear_algebra_backend'] = 'uBLAS'
j = 1
n = 2
n =2
# print n
mesh = UnitSquareMesh(n,n)
# mesh = Mesh('untitled.xml')
c = 1
# print "starting assemble"
tic()
parameters['reorder_dofs_serial'] = False
V = FunctionSpace(mesh, "N1curl", 1)
Q = FunctionSpace(mesh, "DG", 0)
parameters['reorder_dofs_serial'] = False
W = V*Q
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# f=Expression(("2+x[1]*(1-x[1])","2+x[0]*(1-x[0])"))
f= Expression(("(8*pow(pi,2)+1)*sin(2*pi*x[1])*cos(2*pi*x[0])","-(8*pow(pi,2)+1)*sin(2*pi*x[0])*cos(2*pi*x[1])"))
ue= Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])","-sin(2*pi*x[0])*cos(2*pi*x[1])"))
# ue = Expression(("x[1]*(1-x[1])","x[0]*(1-x[0])"))

u0 = Expression(('0','0'))

def u0_boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(W.sub(0), u0, u0_boundary)


a11 = inner(curl(v),curl(u))*dx-c*inner(u,v)*dx
a12 = inner(v,grad(p))
a21 = inner(u,grad(q))
a = a11+a12+a21
b = dolfin.inner(f,v)*dx
A, bb = assemble_system(a, b, bc)
time = toc()

u = Function(V)
print "solve"
set_log_level(PROGRESS)
solver = KrylovSolver("cg","icc")
solver.parameters["relative_tolerance"] = 1e-10
solver.parameters["absolute_tolerance"] = 1e-7
solver.solve(A,u.vector(),bb)
set_log_level(PROGRESS)


# parameters.linear_algebra_backend = "uBLAS"
# AA, bB = assemble_system(a, b)
# print "store matrix"

# rows, cols, values = AA.data()
# # rows1, values1 = bB.data()
# # print AA.data()
# Aa = sps.csr_matrix((values, cols, rows))
# # b = sps.csr_matrix((values1, cols1, rows1))
# # print Aa
# print "save matrix"
# scipy.io.savemat("Ab.mat", {"A": Aa,"b": bB.data()},oned_as='row')
ue= Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])","-sin(2*pi*x[0])*cos(2*pi*x[1])"))
err = ue - u
L2normerr = sqrt(assemble(dolfin.inner(err,err)*dx))
print n,L2normerr
# error[j-1,0] = L2normerr
parameters.linear_algebra_backend = "PETSc"
# plot( mesh,interactive=True)

plot(u, interactive=True)

