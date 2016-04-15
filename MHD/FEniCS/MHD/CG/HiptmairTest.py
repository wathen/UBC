
from dolfin import *
nn = 4
mesh = RectangleMesh(0, 0, 1, 1, nn, nn,'crossed')

order = 2
Magnetic = FunctionSpace(mesh, "N1curl", order)
Lagrange = FunctionSpace(mesh, "CG", order)

v = Function(Lagrange)

interpolate(grad(v),Magnetic)