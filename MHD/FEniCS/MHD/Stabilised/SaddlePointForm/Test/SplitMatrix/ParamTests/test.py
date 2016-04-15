
from dolfin import *

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, 'N1curl', 1)
u = TrialFunction(V)
v = TestFunction(V)

M = assemble(inner(curl(u), curl(v))*dx)

