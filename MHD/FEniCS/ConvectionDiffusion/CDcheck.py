import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
from dolfin import *
import HiptmairSetup
import time
import MatrixOperations as MO
import NSprecondSetup
import numpy as np

n = 16
mesh = UnitSquareMesh(n,n)
V = VectorFunctionSpace( mesh, "CG", 1)
Q = FunctionSpace( mesh, "CG", 1)

p = TrialFunction(Q)
q = TestFunction(Q)

u = Expression(('1','1'))
u_k = interpolate(u,V)
mu = 1
N = FacetNormal(mesh)
h = CellSize(mesh)
h_avg = avg(h)

alpha = 10.
gamma = 10.
p = Expression('x[0]*x[0]')
p = interpolate(p,Q)
L = assemble(mu*(inner(grad(q), grad(p))*dx(mesh))).array()
print L
print (jump(q)*jump(p)).shape()
LL = assemble(jump(q)*jump(p)*dS).sparray()

Fp = assemble(mu*(inner(grad(q), grad(p))*dx(mesh) \
       - inner(avg(grad(q)), outer(p('+'),N('+'))+outer(p('-'),N('-')))*dS (mesh)\
       - inner(outer(q('+'),N('+'))+outer(q('-'),N('-')), avg(grad(p)))*dS (mesh)\
       + alpha/h_avg*inner(outer(q('+'),N('+'))+outer(q('-'),N('-')),outer(p('+'),N('+'))+outer(p('-'),N('-')))*dS(mesh) \
       - inner(outer(q,N), grad(p))*ds(mesh) \
       - inner(grad(q), outer(p,N))*ds(mesh) \
       + gamma/h*inner(q,p)*ds(mesh)) \
       + inner(inner(grad(p),u_k),q)*dx(mesh)- (1/2)*inner(u_k,N)*inner(q,p)*ds(mesh) \
       -(1/2)*(inner(u_k('+'),N('+'))+inner(u_k('-'),N('-')))*avg(inner(q,p))*ds(mesh) \
       -dot(avg(q),dot(outer(p('+'),N('+'))+outer(p('-'),N('-')),avg(u_k)))*dS(mesh)).sparray()




approx = L*p.vector().array()
exact = -p.vector().array()

print approx
print exact

print np.linalg.norm(approx-exact)

