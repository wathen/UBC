import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
from dolfin import *
import sympy as sy
import MatrixOperations as MO

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

    NS1 = -d*(sy.diff(d,x)-sy.diff(b,y))
    NS2 = b*(sy.diff(d,x)-sy.diff(b,y))

    M1 = sy.diff(u*d-v*b,y)
    M2 = -sy.diff(u*d-v*b,x)

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

n = int(2**4)
mesh = UnitSquareMesh(n, n)
V = VectorFunctionSpace(mesh, "CG", 2)
S = FunctionSpace(mesh, "N1curl", 1)

u = TrialFunction(V)
v = TestFunction(V)
b = TrialFunction(S)
c = TestFunction(S)
N = FacetNormal(mesh)

u0, b0, C, Ct, Neumann = Solution()
b0 = interpolate(b0, S)
u0 = interpolate(u0, V)

CoupleT = (v[0]*b0[1]-v[1]*b0[0])*curl(b)*dx
Couple = -(u[0]*b0[1]-u[1]*b0[0])*curl(c)*dx

Lct_D = inner(Ct, v)*dx
Lct_n = inner(Ct, v)*dx - inner(Neumann*N,u)*ds

Lc_D = inner(C, c)*dx

def boundary(x, on_boundary):
    return on_boundary

MO.PrintStr("Fluid coupling test: Dirichlet only",2,"=","\n\n","\n")

bcu = DirichletBC(V, u0, boundary)
A, b = assemble_system(CoupleT, Lct_D, bcu)




