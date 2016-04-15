from dolfin import *
from numpy import *
import scipy as Sci
import scipy.linalg
from math import pi,sin,cos
import scipy.sparse as sps
import scipy.io as save
import scipy
import pdb

j = 1
n = 2
for x in xrange(1,8):
    j = j+1
    n = 2*n
    mesh = UnitSquareMesh(n,n)
    V = FunctionSpace(mesh, "N1curl", 1)
    u0 = Expression(('0','0'))

    def u0_boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u0, u0_boundary)

    u = TrialFunction(V)
    v = TestFunction(V)

    c = 1
    a = dolfin.inner(curl(u),curl(v))*dx+c*dolfin.inner(u,v)*dx
