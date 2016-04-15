import sympy as sy
from dolfin import *
import numpy as np

mesh = UnitSquareMesh(100,100)
V = FunctionSpace(mesh, 'N1curl', 1)
Q = FunctionSpace(mesh, 'CG', 1)

x = sy.symbols('x[0]')
y = sy.symbols('x[1]')
rho = sy.sqrt(x**2 + y**2)
phi = sy.atan2(y,x)

f = rho**(2./3)*sy.sin((2./3)*phi)
b = sy.diff(f,x)
d = sy.diff(f,y)
print sy.ccode(b)


b0 = Expression((str(sy.ccode(b)),str(sy.ccode(d))))
f = Expression(str(sy.ccode(f)))


B = interpolate(b0, V)
F = interpolate(f, Q)
BB = project(grad(F), V)

# print BB.vector().array()
# print B.vector().array()
print np.linalg.norm(BB.vector().array() - B.vector().array())