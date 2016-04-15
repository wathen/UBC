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

mesh = UnitSquareMesh(16,16)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

def boundary(x, on_boundary):
    return on_boundary

u0 = Expression(('0','0'))
p0 = Expression('0')

bc = DirichletBC(V, Expression(('1','1')), boundary)
bc1 = DirichletBC(Q, Expression('0'), boundary)
bcs = [bc, bc1]
v, u = TestFunction(V), TrialFunction(V)
q, p = TestFunction(Q), TrialFunction(Q)

f = Expression(('(2*x[0]-1)*(x[1]*x[1]-x[1])','(2*x[1]-1)*(x[0]*x[0]-x[0])'))
# u_k = interpolate(Constant(('0','0')), V)
u_k = Function(V)
mu = Constant(1e-2)

a11 = mu*inner(grad(v), grad(u))*dx + dolfin.inner(dolfin.dot(u_k,grad(u)),v)*dx
a12 = div(v)*p*dx
a21 = div(u)*q*dx
L1  = inner(v, f)*dx
QQ = inner(u,v)*dx

QQ = assemble(QQ)
bc.apply(QQ)

uu = Function(V)     # new unknown function
eps = 1.0           # error measure ||u-u_k||
tol = 1.0E-4        # tolerance
iter = 0            # iteration counter
maxiter = 100        # max no of iterations allowed

while eps > tol and iter < maxiter:
    iter += 1
    AA = block_assemble([[a11, a12],
                         [a21,  0 ]], bcs=bcs)
    bb = block_assemble([L1, 0], bcs=bcs)


    [B1,B2 ] = bb
    [[A, B],
     [C, _]] = AA

    DoF =B1.array().size + B2.array().size

    Ap = ML(A)

    # Dp=(C*ML(A)*B)
    BB= collapse(C*LumpedInvDiag(QQ)*B)
    # BB = collapse(C*CC)
    MLBB = ML(BB)
    QAQ = collapse(C*LumpedInvDiag(QQ)*A*LumpedInvDiag(QQ)*B)
    Dp = MLBB*(QAQ)*MLBB
    # Dp = C*(ML(A)*B)
    # Dp = ML(P22)
    # StoreMatrix(QAQ.down_cast(),QAQ)
    prec = block_mat([[bat.LGMRES(A, precond = ML(A), tolerance = 1e-10), B],[0, -Dp]])

    #prec = block_mat([[Ap,B],[0,-(Dp)]])
    # AA*bb
    print DoF,'\n\n\n'
    AAinv = bat.LGMRES(AA, precond = prec, tolerance=1e-6, maxiter=500,inner_m=30, outer_k=500)
    uu, pp = AAinv * bb
    u1 = Function(V,uu)
    diff = u1.vector().array() - u_k.vector().array()
    eps = numpy.linalg.norm(diff, ord=numpy.Inf)

    print '\n\n\niter=%d: norm=%g' % (iter, eps)
    # u_k.assign(uu)   # update for next iteration
    u_k.assign(u1)







 # u = uu.vector().array()
ue = Expression(('1','1'))
pe = Expression('x[0]*x[1]*(x[0]-1)*(x[1]-1)')

erru = ue - Function(V,uu)
errp = pe - Function(Q,pp)

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


