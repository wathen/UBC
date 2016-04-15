ls#!/usr/bin/python

import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc


from dolfin import *
import numpy as np
import matplotlib.pylab as plt
import PETScIO as IO
import CavityDomain3d as CD


def remove_ij(x, i, j):

    # Remove the ith row
    idx = range(x.shape[0])
    idx.remove(i)
    x = x[idx,:]

    # Remove the jth column
    idx = range(x.shape[1])
    idx.remove(j)
    x = x[:,idx]

    return x


# Create mesh and define function space
nn = 2**4

mesh, boundaries = CD.CavityMesh3d(nn)

parameters['reorder_dofs_serial'] = False
V = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
M = FunctionSpace(mesh, "N1curl", 1)
L = FunctionSpace(mesh, "CG", 1)
parameters['reorder_dofs_serial'] = False
W = MixedFunctionSpace([V,P,M,L])

def boundary(x, on_boundary):
   return on_boundary

u01 =Expression(("0","0","0"),cell=triangle)
u02 =Expression(("1","0","0"),cell=triangle)
b0 = Expression(("1","0","0"),cell=triangle)
r0 = Expression(("0"),cell=triangle)


bcu1 = DirichletBC(W.sub(0),u01, boundaries,1)
bcu2 = DirichletBC(W.sub(0),u02, boundaries,2)
bcb = DirichletBC(W.sub(2),b0, boundary)
bcr = DirichletBC(W.sub(3),r0, boundary)
bc = [bcu1,bcu2,bcb,bcr]

(u, p, b, r) = TrialFunctions(W)
(v, q, c,s ) = TestFunctions(W)

K = 1e5
Mu_m = 1e5
MU = 1e-2

fns = Expression(("0","0","0"),mu = MU, k = K)
fm = Expression(("0","0","0"),k = K,mu_m = Mu_m)

"'Maxwell Setup'"
a11 = K*Mu_m*inner(curl(c),curl(b))*dx
a12 = inner(c,grad(r))*dx
a21 = inner(b,grad(s))*dx
Lmaxwell  = inner(c, fm)*dx
maxwell = a11+a12+a21


"'NS Setup'"
u_k = Function(Velocity)
u_k.vector()[:] = u_k.vector()[:]*0
n = FacetNormal(mesh)
a11 = MU*inner(grad(v), grad(u))*dx+inner((grad(u)*u_k),v)*dx+(1/2)*div(u_k)*inner(u,v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds
a12 = -div(v)*p*dx
a21 = -div(u)*q*dx
Lns  = inner(v, fns)*dx
ns = a11+a12+a21


"'Coupling term Setup'"
b_k = Function(Electric)
b_k.vector()[:] = b_k.vector()[:]*0

eps = 1.0
tol = 1.0E-5
iter = 0
maxiter = 20
SolutionTime = 0

while eps > tol and iter < maxiter:
    iter += 1

    uu = Function(W)
    tic()
    AA, bb = assemble_system(maxwell+ns+CoupleTerm, Lmaxwell + Lns, bc)
    As = AA.sparray()
    StoreMatrix(As,"A")
    VelPres = Velocitydim[xx-1][0] +Pressuredim[xx-1][0]
    Adelete = remove_ij(As,VelPres-1,VelPres-1)
    A = PETSc.Mat().createAIJ(size=Adelete.shape,csr=(Adelete.indptr, Adelete.indices, Adelete.data))
    print toc()
    b = np.delete(bb,VelPres-1,0)
    zeros = 0*b
    bb = IO.arrayToVec(b)
    x = IO.arrayToVec(zeros)
    ksp = PETSc.KSP().create()
    pc = PETSc.PC().create()
    ksp.setOperators(A)

    ksp.setFromOptions()
    print 'Solving with:', ksp.getType()

    # Solve!
    tic()
    # start = time.time()
    ksp.solve(bb, x)
    # %timit ksp.solve(bb, x)
    # print time.time() - start
    time = toc()
    print time
    SolutionTime = SolutionTime +time
    # print ksp.its

    X = IO.vecToArray(x)
    uu = X[0:Velocitydim[xx-1][0]]
    bb1 = X[VelPres-1:VelPres+Electricdim[xx-1][0]-1]

    u1 = Function(Velocity)
    u1.vector()[:] = u1.vector()[:] + uu
    diff = u1.vector().array() - u_k.vector().array()
    epsu = np.linalg.norm(diff, ord=np.Inf)

    b1 = Function(Electric)
    b1.vector()[:] = b1.vector()[:] + bb1
    diff = b1.vector().array() - b_k.vector().array()
    epsb = np.linalg.norm(diff, ord=np.Inf)
    eps = epsu+epsb
    print '\n\n\niter=%d: norm=%g' % (iter, eps)
    u_k.assign(u1)
    b_k.assign(b1)


