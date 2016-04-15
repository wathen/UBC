from PyTrilinos import Amesos,  Epetra
from dolfin import *
from numpy import *
import scipy.sparse as sps
import scipy.io as save
import scipy
import ipdb
import threading

def StoreMatrix(A,name):
    Aarray = A.array()
    sA = sps.csr_matrix(Aarray)
    test ="".join([name,".mat"])
    scipy.io.savemat( test, {name: sA},oned_as='row')

parameters.linear_algebra_backend = 'Epetra'
# parameters.linear_algebra_backend = 'uBLAS'
# Load mesh
mesh = UnitSquareMesh(16,16)

# Define function spaces
VV =  VectorFunctionSpace(mesh, "N1curl", 1)
QQ =  FunctionSpace(mesh, "CG", 1)
V  =  VectorFunctionSpace(mesh, "CG", 2)
Q  =  FunctionSpace(mesh, "CG", 1)

#W = V * Q * VV * QQ
# Boundaries
def u0_boundary(x, on_boundary):
    return on_boundary

u0 = Expression(('0','0'))
p0 = Expression('0')

Ubc = DirichletBC(V,Expression(('0','0')), u0_boundary)
Pbc = DirichletBC(Q, Expression('0'), u0_boundary)
Bbc = DirichletBC(VV, Expression(('0','0')), u0_boundary)
Rbc = DirichletBC(QQ, Expression('0'), u0_boundary)

# Collect boundary conditions
bcs = [Ubc, Pbc,Bbc,Rbc]

# Define variational problem
#(u, b, p, s) = TrialFunctions(W)
#(v, c, q, r) = TestFunctions(W)
u_ = Function(V)
b_ = Function(VV)
#a = (dolfin.inner(grad(u),grad(v))+ dolfin.inner(dolfin.inner(u_,grad(u_)),v)+dolfin.inner(cross(v,b_),curl(b_))-div(v)*p+dolfin.inner(curl(b),curl(c))-doflin.inner(cross(u,b),u)+dolfin.inner(c,grad(r))-div(u)*q+doflin.inner(b,grad(s)))*dx


b, c = TestFunction(V), TrialFunction(V)
r, s = TestFunction(Q), TrialFunction(Q)

u, v = TestFunction(VV), TrialFunction(VV)
p, q = TestFunction(QQ), TrialFunction(QQ)

k = 1
f=Expression(("2-x[1]*(1-x[1])","2-x[0]*(1-x[0])"))
# a = dolfin.inner(curl(u),curl(v))*dx - k**2*dolfin.inner(u,v)*dx + dolfin.inner(grad(p),v)*dx +dolfin.inner(u,grad(q))*dx
# L = dolfin.inner(f,v)*dx




A = dolfin.inner(grad(u_),grad(v))*dx
O = dolfin.inner(dolfin.inner(u_,grad(u_)),v)*dx
M = dolfin.inner(curl(b),curl(c))*dx
Ct = dolfin.inner(cross(v,b_),curl(b_))*dx
C = dolfin.inner(cross(u_,b_),curl(c))*dx
Bt = div(v)*p*dx
B = div(u)*q*dx
Dt = dolfin.inner(c,grad(r))*dx
D = dolfin.inner(b,grad(s))*dx






# Form for use in constructing preconditioner matrix
gamma = 1-k^2
# b = dolfin.inner(curl(u),curl(v))*dx +gamma*dolfin.inner(u,v)*dx+dolfin.inner(grad(p),grad(q))*dx

# Assemble system
# A, bb = assemble_system(a, L, bcs)

# Assemble preconditioner system
# P, btmp = assemble_system(b, L, bcs)
# lock = threading.Lock()
# lock.acquire()
# a = dolfin.inner(curl(u),curl(v))*dx
# m = dolfin.inner(u,v)*dx
# b = dolfin.inner(v,grad(p))*dx
# bt = dolfin.inner(u,grad(q))*dx
# l = dolfin.inner(grad(p),grad(q))*dx
# rhs = dolfin.inner(f,v)*dx
# A = assemble(a)
# M = assemble(m)
# B = assemble(b)
# Bt = assemble(bt)
# # L = assemble(l)
# RHS = assemble(rhs)

# A11 = A-k*k*M
# A12 = B
# A21 = Bt
# lock = threading.Lock()
# lock.acquire()
# # bc.apply(A11,RHS)
# # bc1.apply(A21)
# # bc.apply(A12)


# p11 = (a+gamma*m)
# p22 = (l)
# from block import *
# from block.iterative import *
# from block.algebraic.trilinos import *

# # AA = block_mat([[A11,A12],[A21,0]])



# AA, AArhs = block_symmetric_assemble([[a-k*k*m, b],[bt,  0 ]], bcs=bcs)
# bb  = block_assemble([rhs, 0], bcs=bcs, symmetric_mod=AArhs)
# P,  Prhs= block_symmetric_assemble([[p11,0],[0,p22]],bcs=bcs)
# lock.release()

# [[A,B],[C,_]] = AA
# [B1,B2] =bb
# [[P11,_],[_,P22]] =P
# # A12" = "test"
# # StoreMatrix(A,"A11")
# # StoreMatrix(B,"A12")
# # StoreMatrix(C,"A21")
# # StoreMatrix(BB,"rhs")
# # # AA = block_mat([])
# x = Function(V)
# xx = Function(Q)

# DoF =B1.array().size + B2.array().size
# # mlList = {"max levels"        : 4,
# #           "output"            : 100,
# #           "smoother: type"    : "symmetric Gauss-Seidel",
# #           "aggregation: type" : "Smooth"
# #          }
# # prec = ML.MultiLevelPreconditioner(p11, False)
# # prec.SetParameterList(mlList)
# # prec.ComputePreconditioner()
# # PP = block_mat([[ML(p11),0],[0,ML(p22)]])
# # bb = block_vec([RHS,0])
# # PP = block_mat([[ML(P11),0],[0,ML(P22)]])
# PP = block_mat([[MinRes(P11,precond=ML(P11),tolerance=1e-6,maxiter=50000,show=1),0],[0,MinRes(P22,precond=ML(P22),tolerance=1e-6,maxiter=50000,show=1)]])


# print "DoF:",DoF
# AAinv = MinRes(AA, precond = PP, tolerance=1e-8,maxiter=50000)

# #AAinv = LinearSolver('direct')
# #U = Function(Q)
# #solve(AA,U.vector(),bb)
# uu, pp = AAinv * bb
# # plot (Function(V,uu))
# # plot (Function(Q,pp))

# ue = Expression(("x[1]*(1-x[1])","x[0]*(1-x[0])"))
# pe = Expression('0')

# erru = ue - Function(V,uu)
# errp = pe - Function(Q,pp)

# print sqrt(assemble(dolfin.inner(erru,erru)*dx))
# print sqrt(assemble(dolfin.inner(errp,errp)*dx))

