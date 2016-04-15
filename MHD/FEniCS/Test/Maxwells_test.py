from dolfin import *
from pdb import *
from numpy import array
from math import pi,sin,cos

mesh = UnitSquareMesh(10,10)
V = FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)
File("mesh.xml") << mesh
#def boundary(x):
#    normal = V.cell().n
#    return cross(normal,x)

# def boundary(x):
#     normal = FacetNormal(mesh)
#     return dot(x[1]*array((0,1)))-dot(array((1,0))*x[0])


u0 = Constant(("0"))
#bc=DirichletBC(V,u0,boundary)

# Define basis and bilinear form
u = TrialFunction(V)
v = TestFunction(V)

c = 1
A = inner(curl(u),curl(v))*dx+c*inner(u,v)*dx
f=Expression(("2+x[1]*(1-x[1])","2+x[0]*(1-x[0])"))
#f= Expression(("(8*pow(pi,2)+1)*sin(2*pi*x[1])*cos(2*pi*x[0])","-(8*pow(pi,2)+1)*sin(2*pi*x[0])*cos(2*pi*x[1])"))
rhs = inner(f,v)*dx

#F = Assemble(rhs)

u = Function(V)
solve(A==rhs,u)

print A(1,1)
file = File("maxwells.xml")
file << u

#plot(u, interactive=True)
