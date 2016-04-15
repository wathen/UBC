import sympy as sy
import numpy as np
from dolfin import *
set_log_active(False)



x = sy.symbols('x[0]')
y = sy.symbols('x[1]')
rho = sy.sqrt(x**2 + y**2)
phi = sy.atan2(y,x)

f = rho**(2./3)*sy.sin((2./3)*phi)
b = sy.diff(f,x)
d = sy.diff(f,y)

b0Upper = Expression((sy.ccode(b),sy.ccode(d)))
b0Lower = Expression((str(sy.ccode(b)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)'),str(sy.ccode(d)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)')))
fUpper = Expression(str(sy.ccode(f)))
fLower = Expression(str(sy.ccode(f)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)'))
class b0(Expression):
    def __init__(self, mesh, bu0, bb0):
        self.mesh = mesh
        self.b0 = bu0
        self.bb0 = bb0
    def eval_cell(self, values, x, ufc_cell):
        if abs(x[0]) < 1e-8 and abs(x[1]) < 1e-8:
            values[0] = 0.0
            values[1] = 0.0
        else:
            if x[1] < 0:
                values[0] = self.bb0(x[0], x[1])[0]
                values[1] = self.bb0(x[0], x[1])[1]
            else:
                values[0] = self.b0(x[0], x[1])[0]
                values[1] = self.b0(x[0], x[1])[1]
            # print values
    def value_shape(self):
        return (2,)
class f0(Expression):
    def __init__(self, mesh, pu0, pb0):
        self.mesh = mesh
        self.p0 = pu0
        self.b0 = pb0
    def eval_cell(self, values, x, ufc_cell):
        if abs(x[0]) < 1e-8 and abs(x[1]) < 1e-8:
            values[0] = 0.0
        else:
            if x[1] < 0:
                values[0] = self.b0(x[0], x[1])
            else:
                values[0] = self.p0(x[0], x[1])

n = 2**2
for i in range(11):
    n = 2**i
    mesh = RectangleMesh(-1,-1,1,1,n,n, 'left')
    cell_f = CellFunction('size_t', mesh, 0)
    for cell in cells(mesh):
        v = cell.get_vertex_coordinates()
        y = v[np.arange(0,6,2)]
        x = v[np.arange(1,6,2)]
        xone = np.ones(3)
        xone[x > 0] = 0
        yone = np.ones(3)
        yone[y < 0] = 0
        if np.sum(xone)+ np.sum(yone)>5.5:
            cell_f[cell] = 1
    mesh = SubMesh(mesh, cell_f, 0)

    b = b0(mesh, b0Upper, b0Lower)
    f = f0(mesh, fUpper, fLower)


    # print
    V = FunctionSpace(mesh, 'N1curl', 1)
    Q = FunctionSpace(mesh, 'CG', 1)

    B = interpolate(b, V)
    F = interpolate(f, Q)
    BB = project(grad(F), V)
    print '\n\n'
    print 'mesh level ', i
    print 'curl(b_h)*curl(b_h)   ', assemble(curl(BB)*curl(BB)*dx)
    print 'curl(b)*curl(b)       ', assemble(curl(B)*curl(B)*dx)
    print np.linalg.norm(BB.vector().array() - B.vector().array())