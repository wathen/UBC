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
mesh = UnitSquareMesh(16,16)

tic()
V = FunctionSpace(mesh, "RT", 2)
Q = FunctionSpace(mesh, "DG", 1)
print 'time to create function spaces', toc(),'\n\n'
W = V*Q

def boundary(x, on_boundary):
    return on_boundary

u0 = Expression(('0','0'))
p0 = Expression('0')

bc = DirichletBC(V,u0, boundary)
bc1 = DirichletBC(Q, Expression('0'), boundary)
bcs = [bc, bc1]
v, u = TestFunction(V), TrialFunction(V)
q, p = TestFunction(Q), TrialFunction(Q)
# (u, p) = TrialFunctions(W)
# (v, q) = TestFunctions(W)
f = Expression(('-pi*pi*cos(pi*x[0])','-pi*pi*cos(pi*x[1])'))

u_k = Function(V)
mu = Constant(1e-0)

a11 = mu*inner(grad(v), grad(u))*dx
a12 = div(v)*p*dx
a21 = div(u)*q*dx
L1  = inner(v, f)*dx
a = a11+a12+a21
# AA = assemble(a11)
# MO.SaveEpertaMatrix(AA.down_cast().mat(),"L")

uu = Function(V)
tic()
AA= block_assemble([[a11, a12],[a21,  0 ]], bcs=bcs)
bb = block_assemble([L1, 0], bcs=bcs)
# AA, bb = assemble_system(a, L1, bcs)
print 'time to create linear system', toc(),'\n\n'
[B1,B2 ] = bb

[[A, B],
[C, _]] = AA


# MO.SaveEpertaMatrix(A.down_cast().mat(),"L")
# MO.SaveEpertaMatrix(B.down_cast().mat(),"B")
# MO.SaveEpertaMatrix(C.down_cast().mat(),"C")

# MO.StoreMatrix(B1,"rhs")

DoF = B1.array().size+B2.array().size
I  = assemble(p*q*dx)

Ap = ML(A)
Ip = LumpedInvDiag(I)
#BB= collapse(C*LumpedInvDiag(QQ)*B)
#MLBB = ML(BB)
#QAQ = collapse(C*LumpedInvDiag(QQ)*A*LumpedInvDiag(QQ)*B)
#Dp = MLBB*(QAQ)*MLBB

prec = block_mat([[Ap,0],[0,Ip]])

print DoF,'\n\n\n'

AAinv = bat.MinRes(AA, precond = prec, tolerance=1e-6, maxiter=500)
uu, pp = AAinv * bb

ue = Expression(('cos(pi*x[0])','cos(pi*x[1])'))
pe = Expression('0')

erru = ue - Function(V,uu)
errp = pe - Function(Q,pp)

print sqrt(assemble(dolfin.inner(erru,erru)*dx))
print sqrt(assemble(dolfin.inner(errp,errp)*dx))
# Plot solution
# plot(Function(V, uu))
# plot(Function(Q, pp))
# interactive()
