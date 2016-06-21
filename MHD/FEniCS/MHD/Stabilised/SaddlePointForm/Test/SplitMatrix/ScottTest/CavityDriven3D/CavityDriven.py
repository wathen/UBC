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
import StokesPrecond
import time

def Domain(n):
    mesh = BoxMesh(Point(-1., -1., -1.), Point(1., 1., 1.), n, n, n)
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], -1.0)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1.0)

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], -1.0)

    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 1.0)

    class Side1(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[2], -1.0)

    class Side2(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[2], 1.0)
    left = Left()
    top = Top()
    right = Right()
    bottom = Bottom()
    side1 = Side1()
    side2 = Side2()
    # Initialize mesh function for the domain
    domains = CellFunction("size_t", mesh)
    domains.set_all(0)

    # Initialize mesh function for boundary domains
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)

    left.mark(boundaries, 1)
    top.mark(boundaries, 1)
    bottom.mark(boundaries, 1)
    right.mark(boundaries, 1)
    side1.mark(boundaries, 1)
    side2.mark(boundaries, 2)

    return mesh, boundaries, domains

    # class u0(Expression):
    #     def __init__(self):
    #     def eval_cell(self, values, x, ufc_cell):
    #         values[0] = 1.0
    #         values[1] = 0


# Sets up the initial guess for the MHD problem
def Stokes(V, Q, F, u0, params, boundaries, domains):
    parameters['reorder_dofs_serial'] = False

    W = V*Q
    IS = MO.IndexSet(W)
    mesh = W.mesh()
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    n = FacetNormal(W.mesh())

    a11 = params[2]*inner(grad(v), grad(u))*dx('everywhere')
    a12 = -div(v)*p*dx('everywhere')
    a21 = -div(u)*q*dx('everywhere')
    a = a11+a12+a21

    L = inner(v, F)*dx('everywhere') #+ inner(gradu0,v)*ds(2)

    def boundary(x, on_boundary):
        return on_boundary
    bcu1 = DirichletBC(W.sub(0), Expression(("0.0","0.0", "0.0")), boundaries, 1)
    bcu2 = DirichletBC(W.sub(0), u0, boundaries, 2)
    bcu = [bcu1, bcu2]

    A, b = assemble_system(a, L, bcu)
    A, b = CP.Assemble(A, b)
    pp = params[2]*inner(grad(v), grad(u))*dx + (1./params[2])*p*q*dx
    P, Pb = assemble_system(pp, L, bcu)
    P, Pb = CP.Assemble(P, Pb)

    # print b.array
    # sss
    u = b.duplicate()

    # ksp = PETSc.KSP()
    # ksp.create(comm=PETSc.COMM_WORLD)
    # pc = ksp.getPC()
    # ksp.setType('preonly')
    # pc.setType('lu')
    # OptDB = PETSc.Options()
    # # if __version__ != '1.6.0':
    # OptDB['pc_factor_mat_solver_package']  = "umfpack"
    # OptDB['pc_factor_mat_ordering_type']  = "rcm"
    # ksp.setFromOptions()

    ksp = PETSc.KSP().create()
    ksp.setTolerances(1e-8)
    ksp.max_it = 200
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    ksp.setType('minres')
    pc.setPythonContext(StokesPrecond.Approx(W, 1))
    ksp.setOperators(A,P)
    # print b.array
    # bbb
    scale = b.norm()
    b = b/scale
    # ksp.setOperators(A,A)
    del A
    start_time = time.time()
    ksp.solve(b,u)
    print ("{:40}").format("Stokes solve, time: "), " ==>  ",("{:4f}").format(time.time() - start_time),("{:9}").format("   Its: "), ("{:4}").format(ksp.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    # Mits +=dodim
    u = u*scale
    # print u.array
    # ss
    u_k = Function(V)
    p_k = Function(Q)
    u_k.vector()[:] = u.getSubVector(IS[0]).array
    p_k.vector()[:] = u.getSubVector(IS[1]).array
    ones = Function(Q)
    ones.vector()[:]=(0*ones.vector().array()+1)
    p_k.vector()[:] += -assemble(p_k*dx('everywhere'))/assemble(ones*dx('everywhere'))
    return u_k, p_k


def Maxwell(V, Q, F, b0, params, HiptmairMatrices, Hiptmairtol):
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
    bcr = DirichletBC(W.sub(1), Expression(("0.0")), boundary)
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
    # if __version__ != '1.6.0':
    OptDB['pc_factor_mat_solver_package']  = "umfpack"
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
