from dolfin import *
from numpy import *

mesh = UnitSquareMesh(2, 2)
V = FunctionSpace(mesh, "N1curl", 1)

u = TrialFunction(V)
v = TestFunction(V)

a = inner(u,v)*dx
