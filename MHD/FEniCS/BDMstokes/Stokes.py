import scipy
from MatrixOperations import *
import os
import cProfile
from dolfin import *
from block import *
from block.algebraic.trilinos import *
import block.iterative as bat
import numpy
import scipy.sparse as sps
import scipy.io as save

MO = MatrixOperations()

n = 4
mesh = RectangleMesh(-1, -1, 1, 1, n, n)

parameters['reorder_dofs_serial'] = False

tic()
V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)
print 'time to create function spaces', toc(),'\n\n'
W = V*Q

def boundary(x, on_boundary):
    return on_boundary

u0 = Expression(("pow(x[1],2)-1","pow(x[0],2)-1"))
p0 = Expression("0")

bc = DirichletBC(V,u0, boundary)
bc1 = DirichletBC(Q, p0, boundary)
bcs = [bc, None]

v, u = TestFunction(V), TrialFunction(V)
q, p = TestFunction(Q), TrialFunction(Q)

f = Expression(('-2','-2'))

u_k = Function(V)
mu = Constant(1e-0)

n = FacetNormal(mesh)
h = CellSize(mesh)
h_avg =avg(h)
alpha = 10.0
gamma =10.0
a11 = inner(grad(v), grad(u))*dx \
    - inner(avg(grad(v)), outer(u('+'),n('+'))+outer(u('-'),n('-')))*dS \
    - inner(outer(v('+'),n('+'))+outer(v('-'),n('-')), avg(grad(u)))*dS \
    + alpha/h_avg*inner(outer(v('+'),n('+'))+outer(v('-'),n('-')),outer(u('+'),n('+'))+outer(u('-'),n('-')))*dS \
    + gamma/h*inner(v,u)*ds

a12 = div(v)*p*dx
a21 = div(u)*q*dx

L1  = -inner(v, f)*dx
i  = p*q*dx

uu = Function(V)

tic()
AA= block_assemble([[-a11, a12],[a21,  0 ]], bcs=bcs)
bb = block_assemble([L1, 0], bcs=bcs)
PP = block_assemble([[a11, 0],[0,  i]], bcs=bcs)
print 'time to create linear system', toc(),'\n\n'
[B1,B2 ] = bb

[[A, B],
[C, _]] = AA

[[L, _],
[_, I]] = PP


# MO.SaveEpertaMatrix(A.down_cast().mat(),"L")
# MO.SaveEpertaMatrix(D.down_cast().mat(),"B1")
# MO.SaveEpertaMatrix(C.down_cast().mat(),"C1")

# MO.StoreMatrix(B1,"rhs")

# DoF = B1.array().size+B2.array().size

Ap = ML(L)
Ip = LumpedInvDiag(I)
#BB= collapse(C*LumpedInvDiag(QQ)*B)
#MLBB = ML(BB)
#QAQ = collapse(C*LumpedInvDiag(QQ)*A*LumpedInvDiag(QQ)*B)
#Dp = MLBB*(QAQ)*MLBB


# MO.SaveEpertaMatrix(A.down_cast().mat(),"L")
# MO.SaveEpertaMatrix(B.down_cast().mat(),"B")
# MO.SaveEpertaMatrix(C.down_cast().mat(),"C")
# # MO.SaveEpertaMatrix(D.down_cast().mat(),"D")
# MO.SaveEpertaMatrix(I.down_cast().mat(),"I")

# MO.StoreMatrix(B1,"B1")
# MO.StoreMatrix(B2,"B2")


# ue = Expression(("pow(x[1],2)-1","pow(x[0],2)-1"))




prec = block_mat([[Ap,0],[0,Ip]])

# print DoF,'\n\n\n'

AAinv = bat.MinRes(AA, precond = prec, tolerance=1e-6, maxiter=5000000)
uu, pp = AAinv * bb

ue = Expression(("pow(x[1],2)-1","pow(x[0],2)-1"))
pe = Expression("0")

erru = ue - Function(V,uu)
errp = pe - Function(Q,pp)

print sqrt(assemble(dolfin.inner(erru,erru)*dx))
print sqrt(assemble(dolfin.inner(errp,errp)*dx))
# Plot solution
plot(Function(V, uu))
plot(interpolate(ue,V))

plot(Function(Q, pp))
plot(interpolate(pe,Q))
interactive()
