from dolfin import *

import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import CheckPetsc4py as CP
import splitCG as cg
n = 2**4

mesh = UnitSquareMesh(n,n)

V = FunctionSpace(mesh,"CG",1)

u = TestFunction(V)
v = TrialFunction(V)

A = assemble(inner(grad(u),grad(v))*dx)
M = assemble(inner(u,v)*dx)
AM = assemble(inner(grad(u),grad(v))*dx+inner(u,v)*dx)

AM = CP.Assemble(AM)
A = CP.Assemble(A)
M = CP.Assemble(M)

u, b = AM.getVecs()
b.set(1)
b = AM*b
u.set(0)

ksp = PETSc.KSP()
ksp.create(comm=PETSc.COMM_WORLD)
pc = ksp.getPC()
ksp.setType('cg')
pc.setType('none')
ksp.setOperators(AM,AM)
ksp.solve(b,u)
print u.array, ksp.its
u, b = AM.getVecs()
b.set(1)
b = (A+M)*b
u.set(0)


kspS = PETSc.KSP()
kspS.create(comm=PETSc.COMM_WORLD)
pcS = kspS.getPC()
kspS.setType('cg')
pcS.setType('none')
P = PETSc.Mat().createPython([A.size[0], A.size[0]])
P.setType('python')
p = cg.SplitMulti(A,M)

P.setPythonContext(p)
kspS.setOperators(P)
kspS.solve(b,u)

print u.array, ksp.its

