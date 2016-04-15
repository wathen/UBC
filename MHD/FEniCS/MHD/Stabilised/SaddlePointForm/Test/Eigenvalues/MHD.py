from dolfin import *
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
n = 2**2
mesh = UnitSquareMesh(n,n)

parameters['reorder_dofs_serial'] = False
parameters['linear_algebra_backend'] = 'uBLAS'
V = VectorFunctionSpace(mesh, 'CG', 2)
Q = FunctionSpace(mesh, 'CG', 1)
C = FunctionSpace(mesh, 'N1curl', 1)

v = TestFunction(V)
u = TrialFunction(V)
q = TestFunction(Q)
p = TrialFunction(Q)
c = TestFunction(C)
b = TrialFunction(C)

b_k = interpolate(Expression(('0.0','0.0')), C)
u_k = interpolate(Expression(('0.0','0.0')), V)
n = FacetNormal(mesh)

F = assemble(inner(grad(v), grad(u))*dx(mesh)+ inner((grad(u)*u_k),v)*dx(mesh) +(1./2)*div(u_k)*inner(u,v)*dx(mesh) - (1./2)*inner(u_k,n)*inner(u,v)*ds(mesh)).sparray()
M = assemble(inner(curl(c),curl(b))*dx).sparray()
B = assemble(-div(u)*q*dx).sparray()
D = assemble(inner(b, grad(q))*dx).sparray()
C = assemble((v[0]*b_k[1]-v[1]*b_k[0])*curl(b)*dx).sparray()
L = assemble(inner(grad(p),grad(q))*dx).sparray()
X = assemble(inner(b,c)*dx).sparray()

Aurg = D*linalg.inv(X)*D.T

A = sp.bmat([[M, D.T]])
