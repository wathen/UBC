import scipy
#from MatrixOperations import *
import os
import cProfile
from dolfin import *
from block import *
from block.algebraic.trilinos import *
import block.iterative as bat
import numpy
import scipy.sparse as sps
import scipy.io as save

#MO = MatrixOperations()
# mesh = UnitSquareMesh(16,16)
n = 16
mesh = RectangleMesh(-1, -1, 1, 1, n, n)
tic()
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
print 'time to create function spaces', toc(),'\n\n'
W = V*Q

def boundary(x, on_boundary):
    return on_boundary
MU = 1
# u0 = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
#kkjk u0 = interpolate(u0,V)
# p0 = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")
u0 = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
p0 = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)+5")


bc = DirichletBC(V,u0, boundary)
bc1 = DirichletBC(Q,p0 , boundary)
bcs = [bc, None]
v, u = TestFunction(V), TrialFunction(V)
q, p = TestFunction(Q), TrialFunction(Q)
# (u, p) = TrialFunctions(W)
# (v, q) = TestFunctions(W)
#f = Expression(("4*x[0]+120*x[1]*x[0]","120*(pow(x[0],2)-pow(x[1],2))"))

# f = Expression(("4*x[0]+120*x[1]*x[0]","0"))
# #

f = Expression(("0","0"), mu = MU)

u_k = Function(V)
mu = Constant(1e-0)

a11 = mu*inner(grad(v), grad(u))*dx
a12 = -div(v)*p*dx
a21 = -div(u)*q*dx
L1  = inner(v, f)*dx
a = a11+a12+a21
# AA = assemble(a11)
# MO.SaveEpertaMatrix(AA.down_cast().mat(),"L")
i  = p*q*dx




uu = Function(V)
tic()
AA= block_assemble([[a11, a12],[a21,  0 ]], bcs=bcs)
bb = block_assemble([L1, 0], bcs=bcs)
PP= block_assemble([[a11, 0],[0,  i]], bcs=bcs)
# AA, bb = assemble_system(a, L1, bcs)
print 'time to create linear system', toc(),'\n\n'


[B1,B2 ] = bb

[[A, B],
[C, D]] = AA

[[L,_],
[_,I]] = PP

# MO.SaveEpertaMatrix(A.down_cast().mat(),"L")
# MO.SaveEpertaMatrix(B.down_cast().mat(),"B")
# MO.SaveEpertaMatrix(C.down_cast().mat(),"C")
# # # MO.SaveEpertaMatrix(D.down_cast().mat(),"D")
# MO.SaveEpertaMatrix(I.down_cast().mat(),"I")
# MO.SaveEpertaMatrix(L.down_cast().mat(),"L2")
# MO.StoreMatrix(B1,"B1")
# MO.StoreMatrix(B2,"B2")




# tic()
# set_log_level(PROGRESS)
# solver = KrylovSolver("cg","ml_amg")
# solver.parameters["relative_tolerance"] = 1e-10
# solver.parameters["absolute_tolerance"] = 1e-10
# solver.solve(AA,u.vector(),bb)
# set_log_level(PROGRESS)
# print 'time to solve linear system', toc(),'\n\n'




# DoF = B1.array().size+B2.array().size


Ap = ML(L)
Ip = LumpedInvDiag(I)
#BB= collapse(C*LumpedInvDiag(QQ)*B)
#MLBB = ML(BB)
#QAQ = collapse(C*LumpedInvDiag(QQ)*A*LumpedInvDiag(QQ)*B)
#Dp = MLBB*(QAQ)*MLBB

prec = block_mat([[Ap,0],[0,Ip]])

# print DoF,'\n\n\n'

AAinv = bat.MinRes(AA, precond = prec, tolerance=1e-6, maxiter=5000)
uu, pp = AAinv * bb

ue = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
pe = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")
ue = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
pe = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)+5")

erru = ue - Function(V,uu)
errp = pe - Function(Q,pp)

# print errornorm(ue,Function(V,uu),norm_type="L2", degree_rise=3,mesh=mesh)
# print errornorm(pe,Function(Q,pp),norm_type="L2", degree_rise=3,mesh=mesh)
# Plot solution
plot(Function(V, uu))
# plot(interpolate(ue,V))
# plot(Function(Q, pp))
# plot(interpolate(pe,Q))
# interactive()
