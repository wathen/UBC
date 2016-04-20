import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import mshr
from dolfin import *
import sympy as sy
import numpy as np
import ExactSol
import MatrixOperations as MO
import CheckPetsc4py as CP
from  dolfin import __version__
import MaxwellPrecond as MP
import StokesPrecond as SP
import time

def Domain(n):


    mesh = RectangleMesh(0., -1., 10., 1.,n,n)
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 10.0)

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], -1.0)

    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 1.0)

    left = Left()
    top = Top()
    right = Right()
    bottom = Bottom()
    # Initialize mesh function for the domain
    domains = CellFunction("size_t", mesh)
    domains.set_all(0)

    # Initialize mesh function for boundary domains
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)

    left.mark(boundaries, 2)
    top.mark(boundaries, 1)
    bottom.mark(boundaries, 1)
    right.mark(boundaries, 2)

    return mesh, boundaries, domains




def ExactSol(mesh, params):

    Re = 1./params[2]
    Ha = sqrt(params[0]/(params[1]*params[2]))
    G = 10.

    x = sy.Symbol('x')
    y = sy.Symbol('y')
    b = (G/params[0])*(sy.sinh(y*Ha)/sy.sinh(Ha)-y)
    print "b", b
    d = 1.0
    print "d", d
    p = -G*x - (G**2)/(2*params[0])*(sy.sinh(y*Ha)/sy.sinh(Ha)-y)**2
    print "\np", p
    u = (G/(params[2]*Ha*sy.tanh(Ha)))*(1-sy.cosh(y*Ha)/sy.cosh(Ha))
    print "\nu", u
    v = sy.diff(x,y)
    print "v", v


    L1 = sy.diff(u,x,x)+sy.diff(u,y,y)
    print "\nL1", L1
    L2 = sy.diff(v,x,x)+sy.diff(v,y,y)
    print "L2", L2

    A1 = u*sy.diff(u,x)+v*sy.diff(u,y)
    print "A1", A1
    A2 = u*sy.diff(v,x)+v*sy.diff(v,y)
    print "A2", A2

    P1 = sy.diff(p,x)
    print "P1", P1
    P2 = sy.diff(p,y)
    print "P2", P2
    C1 = sy.diff(d,x,y) - sy.diff(b,y,y)
    print "C1", C1
    C2 = sy.diff(b,x,y) - sy.diff(d,x,x)
    print "C2", C2
    NS1 = -d*(sy.diff(d,x) - sy.diff(b,y))
    print "NS1", NS1
    NS2 = b*(sy.diff(d,x) - sy.diff(b,y))
    print "NS2", NS2

    M1 = sy.diff(u*d-v*b,y)
    print "M1", M1
    M2 = -sy.diff(u*d-v*b,x)
    print "M2", M2
    # params = [params[0],Mu_m,MU]

    F1 = -params[2]*L1 + P1 - params[0]*NS1 + A1
    print "F1", F1
    F2 = -params[2]*L2 + P2 - params[0]*NS2 + A2
    print "F2", F2

    G1 = params[1]*params[2]*C1 - params[0]*M1
    print "G1", G1
    G2 = params[1]*params[2]*C2 - params[0]*M2
    print "G2", G2
    ssssss
    class u0(Expression):
        def __init__(self, u ,v, X, Y):
            self.u = u
            self.v = v
            self.X = X
            self.Y = Y
        def eval_cell(self, values, x, ufc_cell):
            # print u, v
            values[0] = self.u.subs({self.X:x[0], self.Y:x[1]}).evalf()
            values[1] = self.v.subs({self.X:x[0], self.Y:x[1]}).evalf()
        def value_shape(self):
            return (2,)

    class b0(Expression):
        def __init__(self, b, d, X, Y):
            self.b = b
            self.d = d
            self.X = X
            self.Y = Y
        def eval_cell(self, values, x, ufc_cell):

            values[0] = self.b.subs({self.X:x[0], self.Y:x[1]}).evalf()
            values[1] = self.d
        def value_shape(self):
            return (2,)

    class p0(Expression):
        def __init__(self, p, X, Y):
            self.p = p
            self.X = X
            self.Y = Y
        def eval_cell(self, values, x, ufc_cell):
            values[0] = self.p.subs({self.X:x[0], self.Y:x[1]}).evalf()

    class r0(Expression):
        def __init__(self):
            self.M = 1
        def eval_cell(self, values, x, ufc_cell):
            values[0] = 0

    class F(Expression):
        def __init__(self, F1, F2, X, Y):
            self.F1 = F1
            self.F2 = F2
            self.X = X
            self.Y = Y
        def eval_cell(self, values, x, ufc_cell):

            values[0] = self.F1.subs({self.X:x[0], self.Y:x[1]}).evalf()
            values[1] = self.F2.subs({self.X:x[0], self.Y:x[1]}).evalf()
        def value_shape(self):
            return (2,)

    class G(Expression):
        def __init__(self, G1, G2, X, Y):
            self.G1 = G1
            self.G2 = G2
            self.X = X
            self.Y = Y
        def eval_cell(self, values, x, ufc_cell):
            values[0] = self.G1.subs({self.X:x[0], self.Y:x[1]}).evalf()
            values[1] = self.G2.subs({self.X:x[0], self.Y:x[1]}).evalf()
        def value_shape(self):
            return (2,)

    u0 = u0(u, v, x, y)
    b0 = b0(b, d, x, y)
    p0 = p0(p, x, y)
    r0 = r0()
    F = F(F1,F2, x, y)
    G = G(G1,G2, x, y)
    return u0, b0, p0, r0, F, G




def ExactSol22(mesh, params):

    Re = 1./params[2]
    Ha = sqrt(params[0]/(params[1]*params[2]))
    G = 10.

    class u0(Expression):
        def __init__(self):
            self.u = 1.0
        def eval_cell(self, values, x, ufc_cell):
            # print u, v
            values[0] = -99998.3333527776*cosh(0.01*x[1]) + 100003.333311111
            values[1] = 0.0
        def value_shape(self):
            return (2,)

    class b0(Expression):
        def __init__(self):
            self.b = 1
        def eval_cell(self, values, x, ufc_cell):
            values[0] =  -10.0*x[1] + 999.983333527776*np.sinh(0.01*x[1])
            values[1] = 1.0
        def value_shape(self):
            return (2,)

    class p0(Expression):
        def __init__(self):
            self.p = 1.0
        def eval_cell(self, values, x, ufc_cell):
            values[0] = -10.0*x[0] - 50.0*(-x[1] + 99.9983333527776*sinh(0.01*x[1]))**2
            # values[0] = -10.0*x[0] - 0.5*(-10.0*x[1] + 999.983333527776*np.sinh(0.01*x[1]))**2

    class r0(Expression):
        def __init__(self):
            self.M = 1
        def eval_cell(self, values, x, ufc_cell):
            values[0] = 0.0

    class F(Expression):
        def __init__(self):
            self.F1 = 1
        def eval_cell(self, values, x, ufc_cell):
            values[0] = 0.0
            values[1] = 0.0
        def value_shape(self):
            return (2,)

    class G(Expression):
        def __init__(self):
            self.G1 = 1
        def eval_cell(self, values, x, ufc_cell):
            values[0] = 0.0
            values[1] = 0.0
        def value_shape(self):
            return (2,)

    u0 = u0()
    b0 = b0()
    p0 = p0()
    r0 = r0()
    F = F()
    G = G()
    return u0, b0, p0, r0, F, G






# Sets up the initial guess for the MHD problem
def Stokes(V, Q, F, u0, pN, params, boundaries, domains):
    parameters['reorder_dofs_serial'] = False

    W = V*Q
    IS = MO.IndexSet(W)
    mesh = W.mesh()

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    n = FacetNormal(mesh)

    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    a11 = params[2]*inner(grad(v), grad(u))*dx(0)
    a12 = -div(v)*p*dx(0)
    a21 = -div(u)*q*dx(0)
    a = a11+a12+a21

    L = inner(v, F)*dx(0) - inner(pN*n,v)*ds(2)

    pp = params[2]*inner(grad(v), grad(u))*dx(0)+ (1./params[2])*p*q*dx(0)
    def boundary(x, on_boundary):
        return on_boundary

    bcu = DirichletBC(W.sub(0), Expression(("0.0","0.0")), boundaries, 1)
    # bcu2 = DirichletBC(W.sub(0), u0, boundaries, 2)
    # bcu = [bcu1, bcu2]
    A, b = assemble_system(a, L, bcu)
    A, b = CP.Assemble(A, b)
    C = A.getSubMatrix(IS[1],IS[1])
    u = b.duplicate()

    P, Pb = assemble_system(pp,L,bcu)
    # MO.StoreMatrix(P.sparray(),"P"+str(W.dim()))
    P =CP.Assemble(P)
    M =  P.getSubMatrix(IS[1],IS[1])
    # print M
    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    pc = ksp.getPC()
    ksp.setType('preonly')
    pc.setType('lu')
    OptDB = PETSc.Options()
    if __version__ != '1.6.0':
        OptDB['pc_factor_mat_solver_package']  = "mumps"
    OptDB['pc_factor_mat_ordering_type']  = "rcm"
    ksp.setFromOptions()
    ksp.setOperators(A,A)

    # ksp = PETSc.KSP().create()
    # pc = ksp.getPC()

    # ksp.setType(ksp.Type.MINRES)
    # ksp.setTolerances(1e-8)
    # ksp.max_it = 500
    # #ksp.max_it = 2
    # pc.setType(PETSc.PC.Type.PYTHON)
    # pc.setPythonContext(SP.Approx(W,M))
    # ksp.setOperators(A,P)

    scale = b.norm()
    b = b/scale
    del A
    start_time = time.time()
    ksp.solve(b,u)
    print 333
    # Mits +=dodim
    u = u*scale
    print ("{:40}").format("Stokes solve, time: "), " ==>  ",("{:4f}").format(time.time() - start_time),("{:9}").format("   Its: "), ("{:4}").format(ksp.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    u_k = Function(V)
    p_k = Function(Q)
    u_k.vector()[:] = u.getSubVector(IS[0]).array
    p_k.vector()[:] = u.getSubVector(IS[1]).array
    # ones = Function(Q)
    # ones.vector()[:]=(0*ones.vector().array()+1)
    # p_k.vector()[:] += -assemble(p_k*dx('everywhere'))/assemble(ones*dx('everywhere'))
    return u_k, p_k


def Maxwell(V, Q, F, b0, r0, params, boundaries,HiptmairMatrices, Hiptmairtol):
    parameters['reorder_dofs_serial'] = False

    W = V*Q
    IS = MO.IndexSet(W)

    (b, r) = TrialFunctions(W)
    (c, s) = TestFunctions(W)

    a11 = params[1]*params[2]*inner(curl(b), curl(c))*dx('everywhere')
    a21 = inner(b,grad(s))*dx('everywhere')
    a12 = inner(c,grad(r))*dx('everywhere')
    L = inner(c, F)*dx('everywhere')
    a = a11+a12+a21

    def boundary(x, on_boundary):
        return on_boundary
    class b0(Expression):
        def __init__(self):
            self.p = 1
        def eval_cell(self, values, x, ufc_cell):
            values[0] = 0.0
            values[1] = 1.0
        def value_shape(self):
            return (2,)

    bcb = DirichletBC(W.sub(0), b0(), boundary)
    bcr = DirichletBC(W.sub(1), Expression("0.0"), boundary)
    bc = [bcb, bcr]

    A, b = assemble_system(a, L, bc)
    A, b = CP.Assemble(A, b)
    u = b.duplicate()

    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    pc = ksp.getPC()
    ksp.setType('preonly')
    pc.setType('lu')
    OptDB = PETSc.Options()
    if __version__ != '1.6.0':
        OptDB['pc_factor_mat_solver_package']  = "mumps"
    OptDB['pc_factor_mat_ordering_type']  = "rcm"
    ksp.setFromOptions()

    # ksp = PETSc.KSP().create()
    # ksp.setTolerances(1e-8)
    # ksp.max_it = 200
    # pc = ksp.getPC()
    # pc.setType(PETSc.PC.Type.PYTHON)
    # ksp.setType('minres')
    # pc.setPythonContext(MP.Hiptmair(W, HiptmairMatrices[3], HiptmairMatrices[4], HiptmairMatrices[2], HiptmairMatrices[0], HiptmairMatrices[1], HiptmairMatrices[6],Hiptmairtol))
    scale = b.norm()
    b = b/scale
    ksp.setOperators(A,A)
    del A
    start_time = time.time()
    ksp.solve(b,u)
    print ("{:40}").format("Maxwell solve, time: "), " ==>  ",("{:4f}").format(time.time() - start_time),("{:9}").format("   Its: "), ("{:4}").format(ksp.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    u = u*scale

    b_k = Function(V)
    r_k = Function(Q)
    b_k.vector()[:] = u.getSubVector(IS[0]).array
    r_k.vector()[:] = u.getSubVector(IS[1]).array

    return b_k, r_k








