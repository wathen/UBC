from dolfin import *

l = 3
n = 2**l

mesh2D = UnitSquareMesh(n,n)
mesh3D = UnitCubeMesh(n,n,n)

plot(mesh2D)
plot(mesh3D)
interactive()
