#!/usr/bin/python

from dolfin import *
import CavityDomain3d as CD


parameters['linear_algebra_backend'] = 'uBLAS'

# Create mesh and define function space
nn = 2**2

mesh, boundaries = CD.CavityMesh3d(nn)

parameters['reorder_dofs_serial'] = False
V = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
M = FunctionSpace(mesh, "N1curl", 1)
L = FunctionSpace(mesh, "CG", 1)
parameters['reorder_dofs_serial'] = False
W = MixedFunctionSpace([V,P,M,L])
Vdim = V.dim()
Pdim = P.dim()
Mdim = M.dim()

print W.dim()
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
u_k = Function(V)
u_k.vector()[:] = u_k.vector()[:]*0
n = FacetNormal(mesh)
a11 = MU*inner(grad(v), grad(u))*dx+inner((grad(u)*u_k),v)*dx+(1/2)*div(u_k)*inner(u,v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds
a12 = -div(v)*p*dx
a21 = -div(u)*q*dx
Lns  = inner(v, fns)*dx
ns = a11+a12+a21


"'Coupling term Setup'"
b_k = Function(M)
b_k.vector()[:] = b_k.vector()[:]*0
CoupleTerm = K*inner(cross(v,b_k),curl(b))*dx - K*inner(cross(u,b_k),curl(c))*dx

eps = 1.0
tol = 1.0E-5
iter = 0
maxiter = 20
w = Function(W)

# Compute solution
while eps > tol and iter < maxiter:
    iter += 1
    uu = Function(W)
    tic()
    AA, bb = assemble_system(maxwell+ns+CoupleTerm, Lmaxwell + Lns, bc)
    As = AA.sparray()
    StoreMatrix(As,"A")
    VelPres = Vdim +Pdim
    Adelete = remove_ij(As,VelPres-1,VelPres-1)
    A = PETSc.Mat().createAIJ(size=Adelete.shape,csr=(Adelete.indptr, Adelete.indices, Adelete.data))
    print toc()

    PP,Pb = assemble_system(M11+M22+NS11+NS22,Lmaxwell + Lns, bc)
    Ps = PP.sparray()

    StoreMatrix(Ps,"A")
    VelPres = Vdim +Pdim
    Pdelete = remove_ij(Ps,VelPres-1,VelPres-1)
    P = PETSc.Mat().createAIJ(size=Pdelete.shape,csr=(Pdelete.indptr, Pdelete.indices, Pdelete.data))

    b = np.delete(bb,VelPres-1,0)
    zeros = 0*b
    bb = IO.arrayToVec(b)
    x = IO.arrayToVec(zeros)

    ksp = PETSc.KSP().create()
    pc = PETSc.PC().create()
    ksp.setFromOptions()
    ksp.setTolerances(1e-16)
    ksp.setType('preonly')
    ksp.setOperators(A)
    ksp.solve(bb, x)

    X = IO.vecToArray(x)
    uu = X[0:Vdim]
    bb1 = X[VelPres-1:VelPres+Mdim-1]

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


X = IO.vecToArray(x)
xu = X[0:Velocitydim[xx-1][0]]
ua = Function(Velocity)
ua.vector()[:] = xu

pp = X[Velocitydim[xx-1][0]:VelPres-1]
# xp[-1] = 0
# pa = Function(Pressure)
# pa.vector()[:] = xp

n = pp.shape
pp = np.insert(pp,n,0)
pa = Function(Pressure)
pa.vector()[:] = pp

pend = assemble(pa*dx)

ones = Function(Pressure)
ones.vector()[:]=(0*pp+1)
pp = Function(Pressure)
pp.vector()[:] = pa.vector().array()- assemble(pa*dx)/assemble(ones*dx)






xb = X[VelPres-1:VelPres+Electricdim[xx-1][0]-1]
ba = Function(Electric)
ba.vector()[:] = xb

xr = X[VelPres+Electricdim[xx-1][0]-1:]
ra = Function(Lagrange)
ra.vector()[:] = xr