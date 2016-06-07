import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
from dolfin import *
import sympy as sy
import MatrixOperations as MO


def Solution():
    x = symbols('x[0]')
    y = symbols('x[1]')
    uu = y*x*exp(x+y)
    u = diff(uu,y)
    v = -diff(uu,x)
    uu = x*exp(x+y)
    b = diff(uu,y)
    d = -diff(uu,x)

    NS1 = -d*(diff(d,x)-diff(b,y))
    NS2 = b*(diff(d,x)-diff(b,y))

    M1 = diff(u*d-v*b,y)
    M2 = -diff(u*d-v*b,x)

    J11 = sy.diff(u, x)
    J12 = sy.diff(u, y)
    J21 = sy.diff(v, x)
    J22 = sy.diff(v, y)

    Ct = Expression((ccode(NS1),ccode(NS2)))
    C = Expression((ccode(M1),ccode(M2)))
    u0 = Expression((ccode(u),ccode(v)))
    b0 = Expression((ccode(b),ccode(d)))
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

u0, b0, C, Ct, Neumann = Solution()
b0 = interpolate(b0, S)
u0 = interpolate(u0, V)

MO.PrintStr("Fluid coupling test",2,"=","\n\n","\n")

CoupleT = (v[0]*b0[1]-v[1]*b0[0])*curl(b)*dx
Couple = -(u[0]*b0[1]-u[1]*b0[0])*curl(c)*dx


