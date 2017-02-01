import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
from dolfin import *
import sympy as sy
import numpy as np
import ExactSol
import MatrixOperations as MO
import CheckPetsc4py as CP
import MaxwellPrecond as MP
import StokesPrecond
from  dolfin import __version__
import time

def Domain(n):
    mesh = UnitSquareMesh(n, n)
    # RectangleMesh(Point(-1., -1.), Point(1., 1.), n, n)
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1.0)

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.0)

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

    left.mark(boundaries, 1)
    top.mark(boundaries, 2)
    bottom.mark(boundaries, 1)
    right.mark(boundaries, 1)
    return mesh, boundaries, domains

    # class u0(Expression):
    #     def __init__(self):
    #     def eval_cell(self, values, x, ufc_cell):
    #         values[0] = 1.0
    #         values[1] = 0


# Sets up the initial guess for the MHD problem
def Stokes(V, Q, F, params, boundaries, domains, mesh):
    parameters['reorder_dofs_serial'] = False

    W = FunctionSpace(mesh, MixedElement([V, Q]))

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

    bcu1 = DirichletBC(W.sub(0), Expression(("0.0","0.0"), degree=4), boundaries, 1)
    bcu2 = DirichletBC(W.sub(0), Expression(("1.0","0.0"), degree=4), boundaries, 2)
    bcu = [bcu1, bcu2]

    A, b = assemble_system(a, L, bcu)
    A, b = CP.Assemble(A, b)
    pp = params[2]*inner(grad(v), grad(u))*dx + (1./params[2])*p*q*dx
    P, Pb = assemble_system(pp, L, bcu)
    P, Pb = CP.Assemble(P, Pb)

    u = b.duplicate()

    ksp = PETSc.KSP().create()
    ksp.setTolerances(1e-8)
    ksp.max_it = 200
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    ksp.setType('minres')
    pc.setPythonContext(StokesPrecond.Approx(W, 1))
    ksp.setOperators(A,P)

    scale = b.norm()
    b = b/scale
    ksp.setOperators(A,A)
    del A
    start_time = time.time()
    ksp.solve(b,u)
    print ("{:40}").format("Stokes solve, time: "), " ==>  ",("{:4f}").format(time.time() - start_time),("{:9}").format("   Its: "), ("{:4}").format(ksp.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    u = u*scale
    u_k = Function(FunctionSpace(mesh, V))
    p_k = Function(FunctionSpace(mesh, Q))
    u_k.vector()[:] = u.getSubVector(IS[0]).array
    p_k.vector()[:] = u.getSubVector(IS[1]).array
    ones = Function(FunctionSpace(mesh, Q))
    ones.vector()[:]=(0*ones.vector().array()+1)
    p_k.vector()[:] += -assemble(p_k*dx("everywhere"))/assemble(ones*dx("everywhere"))
    return u_k, p_k

def Maxwell(V, Q, F, params, HiptmairMatrices, Hiptmairtol, mesh):
    parameters['reorder_dofs_serial'] = False

    W = FunctionSpace(mesh, MixedElement([V, Q]))

    IS = MO.IndexSet(W)

    print params
    (b, r) = TrialFunctions(W)
    (c, s) = TestFunctions(W)

    a11 = params[1]*params[2]*inner(curl(b), curl(c))*dx('everywhere')
    a21 = inner(b,grad(s))*dx('everywhere')
    a12 = inner(c,grad(r))*dx('everywhere')
    L = inner(c, F)*dx('everywhere')
    a = a11+a12+a21

    def boundary(x, on_boundary):
        return on_boundary

    bcb = DirichletBC(W.sub(0), Expression(("1.0","0.0"), degree=4), boundary)
    bcr = DirichletBC(W.sub(1), Expression(("0.0"), degree=4), boundary)
    bc = [bcb, bcr]

    A, b = assemble_system(a, L, bc)
    A, b = CP.Assemble(A, b)
    u = b.duplicate()

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

    b_k = Function(FunctionSpace(mesh, V))
    r_k = Function(FunctionSpace(mesh, Q))
    b_k.vector()[:] = u.getSubVector(IS[0]).array
    r_k.vector()[:] = u.getSubVector(IS[1]).array
    return b_k, r_k
