import scipy
import os
import cProfile
from dolfin import *
from block import *
from block.algebraic.trilinos import *
import block.iterative as bat
import numpy
import scipy.sparse as sps
import scipy.io as save

def StoreMatrix(A,name):
    Aarray = A.array()
    sA = sps.csr_matrix(Aarray)
    test ="".join([name,".mat"])
    scipy.io.savemat( test, {name: sA},oned_as='row')

mesh = UnitSquareMesh(32,32)

V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)
W = V*Q

def boundary(x, on_boundary):
    return on_boundary

u0 = Expression(('1','1'))
p0 = Expression('0')

bc = DirichletBC(W.sub(0),u0, boundary)
bc1 = DirichletBC(W.sub(1), p0, boundary)
bcs = [bc, bc1]
# v, u = TestFunction(V), TrialFunction(V)
# q, p = TestFunction(Q), TrialFunction(Q)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

f = Expression(('(2*x[0]-1)*(x[1]*x[1]-x[1])','(2*x[1]-1)*(x[0]*x[0]-x[0])'))
# u_k = interpolate(Constant(('0','0')), V)
u_k = Function(V)
mu = Constant(1e-2)
N = FacetNormal(mesh)
h = CellSize(mesh)
h_avg =avg(h)
alpha = 10.0
gamma =10.0
a11 = inner(grad(v), grad(u))*dx \
        - inner(avg(grad(v)), outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
        - inner(outer(v('+'),N('+'))+outer(v('-'),N('-')), avg(grad(u)))*dS \
        + alpha/h_avg*inner(outer(v('+'),N('+'))+outer(v('-'),N('-')),outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
        - inner(outer(v,N), grad(u))*ds \
        - inner(grad(v), outer(u,N))*ds \
        + gamma/h*inner(v,u)*ds

O = inner((grad(u)*u_k),v)*dx- (1/2)*inner(u_k,N)*inner(u,v)*ds \
     -(1/2)*(inner(u_k('+'),N('+'))+inner(u_k('-'),N('-')))*avg(inner(u,v))*ds \
    -dot(avg(v),dot(outer(u('+'),N('+'))+outer(u('-'),N('-')),avg(u_k)))*dS

a12 = div(v)*p*dx
a21 = div(u)*q*dx
L1  = inner(v, f)*dx + gamma/h*inner(u0,v)*ds - inner(grad(v),outer(u0,N))*ds
a = a11+O-a12-a21



uu = Function(W)     # new unknown function
eps = 1.0           # error measure ||u-u_k||
tol = 1.0E-8       # tolerance
iter = 0            # iteration counter
maxiter = 100        # max no of iterations allowed

while eps > tol and iter < maxiter:
    iter += 1
    AA, bb = assemble_system(a, L1, bcs)

    # solver = KrylovSolver('m')
    solve(a == L1, uu, bcs)
    uu1, pa = uu.split()
    u1 = Function(V,uu1)

    diff = u1.vector().array() - u_k.vector().array()
    eps = numpy.linalg.norm(diff, ord=numpy.Inf)

    print '\n\n\niter=%d: norm=%g' % (iter, eps)
    # u_k.assign(uu)   # update for next iteration
    u_k.assign(u1)







 # u = uu.vector().array()
ue = Expression(('1','1'))
pe = Expression('x[0]*x[1]*(x[0]-1)*(x[1]-1)')

erru = ue - u1
errp = pe - Function(Q,pa)

print sqrt(assemble(dolfin.inner(erru,erru)*dx))
print sqrt(assemble(dolfin.inner(errp,errp)*dx))
# Plot solution
# plot(Function(V, uu))
# plot(Function(Q, pp))
# interactive()
#
#
#
#


