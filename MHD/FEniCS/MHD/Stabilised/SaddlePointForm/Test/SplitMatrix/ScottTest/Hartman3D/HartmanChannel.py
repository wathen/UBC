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
def Domain(n, L, y0, z0):


    if __version__ == '1.6.0':
        mesh = BoxMesh(Point(0., -y0, -z0), Point(L, y0, z0), n, n, n)
    else:
        # mesh = BoxMesh(0., -y0, -z0, L, y0, z0, 5*n, 2*n, n)
        mesh = BoxMesh(0., -y0, -z0, L, y0, z0, n, n, n)
    cell_f = CellFunction('size_t', mesh, 0)

    class NeumannIn(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0)

    class NeumannOut(SubDomain):
        # def __init__(self, L):
        #     self.L = L
        def inside(self, x, on_boundary):
            return near(x[0], 10.0)

    class DirichletY1(SubDomain):
        # def __init__(self, y0):
        #     self.y0 = y0
        def inside(self, x, on_boundary):
            return near(x[1], -2.0)

    class DirichletY2(SubDomain):
        # def __init__(self, y0):
        #     self.y0 = y0
        def inside(self, x, on_boundary):
            return near(x[1], 2.0)

    class DirichletZ1(SubDomain):
        # def __init__(self, z0):
        #     self.z0 = z0
        def inside(self, x, on_boundary):
            return near(x[2], -1.0)

    class DirichletZ2(SubDomain):
        # def __init__(self, z0):
        #     self.z0 = z0
        def inside(self, x, on_boundary):
            return near(x[2], 1.0)


    NeumannIn = NeumannIn()
    NeumannOut = NeumannOut()
    DirichletY1 = DirichletY1()
    DirichletY2 = DirichletY2()
    DirichletZ1 = DirichletZ1()
    DirichletZ2 = DirichletZ2()


    # Initialize mesh function for the domain
    domains = CellFunction("size_t", mesh)
    domains.set_all(0)

    # Initialize mesh function for boundary domains
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)

    NeumannIn.mark(boundaries, 1)
    NeumannOut.mark(boundaries, 1)
    DirichletY1.mark(boundaries, 2)
    DirichletY2.mark(boundaries, 2)
    DirichletZ1.mark(boundaries, 2)
    DirichletZ2.mark(boundaries, 2)

    return mesh, boundaries, domains




def ExactSol(mesh, params, y0, z0, trunc):

    Re = 1./params[2]
    Ha = sqrt(params[0]/(params[1]*params[2]))
    G = 0.5
    print Ha
    x = sy.symbols('x')
    y = sy.symbols('y')
    z = sy.symbols('z')
    n = sy.symbols('n')
    # n = 0
    Lambda = ((2*n+1)*sy.pi)/(2*z0)

    p1 = sy.sqrt(Lambda**2 + (Ha**2)/2 + Ha*sy.sqrt(Lambda**2 + (Ha**2)/4))
    p2 = sy.sqrt(Lambda**2 + (Ha**2)/2 - Ha*sy.sqrt(Lambda**2 + (Ha**2)/4))

    Delta = p2*(Lambda**2-p1**2)*sy.sinh(p1*y0)*sy.cosh(p2*y0) - p1*(Lambda**2-p2**2)*sy.sinh(p2*y0)*sy.cosh(p1*y0)

    un0 = (-2*G*sy.sin(Lambda*z0))/(params[2]*z0*Lambda**3)

    A = (-p1*(Lambda**2-p2**2)/Delta)*un0*sy.sinh(p2*y0)
    B = (p2*(Lambda**2-p1**2)/Delta)*un0*sy.sinh(p1*y0)

    ATrunc = np.zeros((trunc,1))
    BTrunc = np.zeros((trunc,1))
    p1Trunc = np.zeros((trunc,1))
    p2Trunc = np.zeros((trunc,1))
    un0Trunc = np.zeros((trunc,1))
    LambdaTrunc = np.zeros((trunc,1))
    for i in range(trunc):
        print ' n = ', i
        ATrunc[i] = A.subs(n,float(i)).evalf()
        BTrunc[i] = B.subs(n,float(i)).evalf()
        p1Trunc[i] = p1.subs(n,float(i)).evalf()
        p2Trunc[i] = p2.subs(n,float(i)).evalf()
        un0Trunc[i] = un0.subs(n,float(i)).evalf()
        LambdaTrunc[i] = Lambda.subs(n,float(i)).evalf()
        print '  A_n (lambda^2-p_1^2)/p_1 = ',ATrunc[i]*(LambdaTrunc[i]**2 - p1Trunc[i]**2)/p1Trunc[i]
        print '  B_n (lambda^2-p_2^2)/p_2 = ',BTrunc[i]*(LambdaTrunc[i]**2 - p2Trunc[i]**2)/p2Trunc[i]


    print ATrunc

    P = [G, Re, z0]
    class b0(Expression):
        def __init__(self, trunc, ATrunc, BTrunc, p1Trunc, p2Trunc, LambdaTrunc, Re, params):
            self.trunc = trunc
            self.A = ATrunc
            self.B = BTrunc
            self.p1 = p1Trunc
            self.p2 = p2Trunc
            self.Lambda = LambdaTrunc
            self.Re = Re
            self.params = params

        def eval_cell(self, values, x, ufc_cell):
            # bFourier =
            bn = (1./(self.Re*params[0])*(self.A*np.sinh(self.p1*x[1])*(self.Lambda**2-self.p1**2)/self.p1 + self.B*np.sinh(self.p2*x[1])*(self.Lambda**2-self.p2**2)/self.p2))*np.cos(self.Lambda*x[2])
            values[0] = np.sum(bn)#.evalf()
            values[1] = 1.0
            values[2] = 0
            values = values*1e5
            # print values

        def value_shape(self):
            return (3,)

    class u0(Expression):
        def __init__(self, trunc, ATrunc, BTrunc, p1Trunc, p2Trunc, LambdaTrunc,P):
            self.trunc = trunc
            self.A = ATrunc
            self.B = BTrunc
            self.p1 = p1Trunc
            self.p2 = p2Trunc
            self.Lambda = LambdaTrunc
            self.P = P
        def eval_cell(self, values, x, ufc_cell):
            un = (self.A*np.cosh(self.p1*x[1]) + self.B*np.cosh(self.p2*x[1]))*np.cos(self.Lambda*x[2])
            values[0] = (-1./2)*self.P[0]*self.P[1]*(x[2]**2-self.P[2]**2) + np.sum(un)
            values[1] = 0
            values[2] = 0
            # print values

        def value_shape(self):
            return (3,)
    PP = [Re,G]
    class pN(Expression):
        def __init__(self, trunc, ATrunc, BTrunc, p1Trunc, p2Trunc, LambdaTrunc, P, params):
            self.trunc = trunc
            self.A = ATrunc
            self.B = BTrunc
            self.p1 = p1Trunc
            self.p2 = p2Trunc
            self.Lambda = LambdaTrunc
            self.P = P
            self.params = params
        def eval_cell(self, values, x, ufc_cell):
            # bFourier =
            bn = (1./(self.P[0]*params[0])*(self.A*np.sinh(self.p1*x[1])*(self.Lambda**2-self.p1**2)/self.p1 + self.B*np.sinh(self.p2*x[1])*(self.Lambda**2-self.p2**2)/self.p2))*np.cos(self.Lambda*x[2])
            values[0] = (-self.P[1]*x[0] - (self.params[0]*np.sum(bn)**2)/2 + 10)

    class pN2(Expression):
        def __init__(self, trunc, ATrunc, BTrunc, p1Trunc, p2Trunc, LambdaTrunc, P, params):
            self.trunc = trunc
            self.A = ATrunc
            self.B = BTrunc
            self.p1 = p1Trunc
            self.p2 = p2Trunc
            self.Lambda = LambdaTrunc
            self.P = P
            self.params = params
        def eval_cell(self, values, x, ufc_cell):
            # bFourier =
            bn = (1./(self.P[0]*params[0])*(self.A*np.sinh(self.p1*x[1])*(self.Lambda**2-self.p1**2)/self.p1 + self.B*np.sinh(self.p2*x[1])*(self.Lambda**2-self.p2**2)/self.p2))*np.cos(self.Lambda*x[2])
            values[0] = -(-self.P[1]*x[0] - (self.params[0]*np.sum(bn)**2)/2 + 10)
            # print values


    u0 = u0(trunc, ATrunc, BTrunc, p1Trunc, p2Trunc, LambdaTrunc, P)
    b0 = b0(trunc, ATrunc, BTrunc, p1Trunc, p2Trunc, LambdaTrunc, Re, params)
    pN = pN(trunc, ATrunc, BTrunc, p1Trunc, p2Trunc, LambdaTrunc, PP, params)
    pN2 = pN2(trunc, ATrunc, BTrunc, p1Trunc, p2Trunc, LambdaTrunc, PP, params)

    return u0, b0, pN, pN2









# Sets up the initial guess for the MHD problem
def Stokes(V, Q, F, u0, pN, params, boundaries, domains):
    parameters['reorder_dofs_serial'] = False

    W = V*Q
    IS = MO.IndexSet(W)
    mesh = W.mesh()
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    dx = Measure('dx', domain=mesh)
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    n = FacetNormal(W.mesh())

    a11 = params[2]*inner(grad(v), grad(u))*dx('everywhere')
    a12 = -div(v)*p*dx('everywhere')
    a21 = -div(u)*q*dx('everywhere')
    a = a11+a12+a21

    L = inner(v, F)*dx('everywhere') #+ inner(pN*n,v)*ds(1)

    pp = params[2]*inner(grad(v), grad(u))*dx('everywhere')+ (1./params[2])*p*q*dx('everywhere')
    def boundary(x, on_boundary):
        return on_boundary

    bcu = DirichletBC(W.sub(0), u0, boundary)
    #bcr = DirichletBC(W.sub(1), Expression(("0.0")), boundary)

    A, b = assemble_system(a, L, bcu)
    A, b = CP.Assemble(A, b)
    C = A.getSubMatrix(IS[1],IS[1])
    u = b.duplicate()

    P, Pb = assemble_system(pp,L,bcu)
    # MO.StoreMatrix(P.sparray(),"P"+str(W.dim()))
    P =CP.Assemble(P)
    M =  P.getSubMatrix(IS[1],IS[1])
    # print M
#    ksp = PETSc.KSP()
#    ksp.create(comm=PETSc.COMM_WORLD)
#    pc = ksp.getPC()
#    ksp.setType('preonly')
#    pc.setType('lu')
#    OptDB = PETSc.Options()
#    if __version__ != '1.6.0':
#        OptDB['pc_factor_mat_solver_package']  = "mumps"
#    OptDB['pc_factor_mat_ordering_type']  = "rcm"
#    ksp.setFromOptions()
#    ksp.setOperators(A,A)

    ksp = PETSc.KSP().create()
    pc = ksp.getPC()

    ksp.setType(ksp.Type.MINRES)
    ksp.setTolerances(1e-8)
    ksp.max_it = 500
    #ksp.max_it = 2
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setPythonContext(SP.Approx(W,M))
    ksp.setOperators(A,P)

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

    bcb = DirichletBC(W.sub(0), b0, boundary)
    bcr = DirichletBC(W.sub(1), r0, boundary)
    bc = [bcb, bcr]

    A, b = assemble_system(a, L, bc)
    A, b = CP.Assemble(A, b)
    u = b.duplicate()

#    ksp = PETSc.KSP()
#    ksp.create(comm=PETSc.COMM_WORLD)
#    pc = ksp.getPC()
#    ksp.setType('preonly')
#    pc.setType('lu')
#    OptDB = PETSc.Options()
#    if __version__ != '1.6.0':
#        OptDB['pc_factor_mat_solver_package']  = "mumps"
#    OptDB['pc_factor_mat_ordering_type']  = "rcm"
#    ksp.setFromOptions()

    ksp = PETSc.KSP().create()
    ksp.setTolerances(1e-8)
    ksp.max_it = 200
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    ksp.setType('minres')
    pc.setPythonContext(MP.Hiptmair(W, HiptmairMatrices[3], HiptmairMatrices[4], HiptmairMatrices[2], HiptmairMatrices[0], HiptmairMatrices[1], HiptmairMatrices[6],Hiptmairtol))
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








