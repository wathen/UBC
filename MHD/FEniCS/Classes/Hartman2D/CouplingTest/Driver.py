import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
from dolfin import *
import sympy as sy
import MatrixOperations as MO
import CheckPetsc4py as CP
import numpy as np
import scipy.sparse as sp

def PETSc2Scipy(A):
    row, col, value = A.getValuesCSR()
    return sp.csr_matrix((value, col, row), shape=A.size)

def arrayToVec(vecArray):

    vec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    vec.setSizes(len(vecArray))
    vec.setUp()
    (Istart,Iend) = vec.getOwnershipRange()
    return vec.createWithArray(vecArray[Istart:Iend],
            comm=PETSc.COMM_WORLD)
    vec.destroy()

def myCCode(A):
    return sy.ccode(A).replace('M_PI','pi')

def Solution():
    x = sy.symbols('x[0]')
    y = sy.symbols('x[1]')
    uu = y*x*sy.exp(x+y)
    u = sy.diff(uu,y)
    v = -sy.diff(uu,x)
    uu = x*sy.exp(x+y)
    b = sy.diff(uu,y)
    d = -sy.diff(uu,x)
    # u = y
    # v = x
    # b = y
    # d = x
    NS1 = -d*(sy.diff(d,x)-sy.diff(b,y))
    NS2 = b*(sy.diff(d,x)-sy.diff(b,y))

    print NS1
    print NS2

    M1 = sy.diff(u*d-v*b,y)
    M2 = -sy.diff(u*d-v*b,x)

    print M1
    print M2

    J11 = sy.diff(u, x)
    J12 = sy.diff(u, y)
    J21 = sy.diff(v, x)
    J22 = sy.diff(v, y)

    Ct = Expression((myCCode(NS1),myCCode(NS2)))
    C = Expression((myCCode(M1),myCCode(M2)))
    u0 = Expression((myCCode(u),myCCode(v)))
    b0 = Expression((myCCode(b),myCCode(d)))

    Neumann = as_matrix(((Expression(myCCode(J11)), Expression(myCCode(J12))), (Expression(myCCode(J21)), Expression(myCCode(J22)))))

    return u0, b0, C, Ct, Neumann

n = int(2**1)

mesh = UnitSquareMesh(n, n)
parameters['reorder_dofs_serial'] = False

V = VectorFunctionSpace(mesh, "CG", 2)
S = FunctionSpace(mesh, "N1curl", 1)
W = MixedFunctionSpace([V, S])

(u, b) = TrialFunctions(W)
(v, c) = TestFunctions(W)
N = FacetNormal(mesh)

u0, b0, C, Ct, Neumann = Solution()
b0 = interpolate(b0, S)
u0 = interpolate(u0, V)

CoupleT = (v[0]*b0[1]-v[1]*b0[0])*curl(b)*dx
Couple = -(u[0]*b0[1]-u[1]*b0[0])*curl(c)*dx

L_D = inner(Ct, v)*dx - inner(C, c)*dx
L_N = inner(Ct, v)*dx - inner(C, c)*dx - inner(Neumann*N, v)*ds

def boundary(x, on_boundary):
    return on_boundary

u_is = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs())
b_is = PETSc.IS().createGeneral(W.sub(1).dofmap().dofs())

u_solution = arrayToVec(u0.vector().array())
b_solution = arrayToVec(b0.vector().array())

MO.PrintStr("Fluid coupling test: Dirichlet only",2,"=","\n\n","\n")

bcu = DirichletBC(W.sub(0), u0, boundary)
bcb = DirichletBC(W.sub(1), b0, boundary)
bc = [bcu, bcb]
A = assemble(CoupleT+Couple)
b_d = assemble(L_D)
b_n = assemble(L_N)
for bcs in bc:
    bcs.apply(A)
    bcs.apply(b_d)
# print A.array()
# print b_n.array()
# print A.array() - b_d.array()
A, b = CP.Assemble(A, b_n)
u = arrayToVec(np.concatenate((u0.vector().array(), b0.vector().array()), axis=1))
print "norm(A*u-f)  ", np.linalg.norm((A*u-b).array)
print (A*u).array - b.array

C = A.getSubMatrix(b_is, u_is)
Ct = A.getSubMatrix(u_is, b_is)

f_u = b.getSubVector(u_is)
f_b = b.getSubVector(b_is)

print "norm(Ct*b-f)  ", np.linalg.norm((Ct*b_solution-f_u).array)
print "norm(C*u-f):   ", np.linalg.norm((C*u_solution-f_b).array)
print (Ct*b_solution).array- f_u.array
print (C*u_solution-f_b).array

# print f_b.array


