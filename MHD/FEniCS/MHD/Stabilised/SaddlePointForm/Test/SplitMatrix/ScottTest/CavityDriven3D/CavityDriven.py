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
from  dolfin import __version__
import MaxwellPrecond as MP
import StokesPrecond
import time

def Domain(n):
    mesh = UnitCubeMesh(n, n, n)
    # BoxMesh(Point(-1., -1., -1.), Point(1., 1., 1.), n, n, n)
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

    class Side1(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[2], 0.0)

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
def Stokes(V, Q, F, u0, pN, params, mesh):
    parameters['reorder_dofs_serial'] = False

    W = FunctionSpace(mesh, MixedElement([V, Q]))

    IS = MO.IndexSet(W)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    n = FacetNormal(mesh)

    # dx = Measure('dx', domain=mesh, subdomain_data=domains)
    # ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    a11 = params[2]*inner(grad(v), grad(u))*dx
    a12 = -div(v)*p*dx
    a21 = -div(u)*q*dx
    a = a11+a12+a21

    L = inner(v, F)*dx #- inner(pN*n,v)*ds(2)

    pp = params[2]*inner(grad(v), grad(u))*dx + (1./params[2])*p*q*dx
    def boundary(x, on_boundary):
        return on_boundary
    # bcu = DirichletBC(W.sub(0), u0, boundaries, 1)
    bcu = DirichletBC(W.sub(0), u0, boundary)
    # bcu = [bcu1, bcu2]
    A, b = assemble_system(a, L, bcu)
    A, b = CP.Assemble(A, b)
    C = A.getSubMatrix(IS[1],IS[1])
    u = b.duplicate()
    P, Pb = assemble_system(pp, L, bcu)
    P, Pb = CP.Assemble(P, Pb)

    # ksp = PETSc.KSP()
    # ksp.create(comm=PETSc.COMM_WORLD)
    # pc = ksp.getPC()
    # ksp.setType('preonly')
    # pc.setType('lu')
    # OptDB = PETSc.Options()
    # # if __version__ != '1.6.0':
    # OptDB['pc_factor_mat_solver_package']  = "pastix"
    # OptDB['pc_factor_mat_ordering_type']  = "rcm"
    # ksp.setFromOptions()
    # ksp.setOperators(A,A)

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
    del A
    start_time = time.time()
    ksp.solve(b,u)
    # Mits +=dodim
    u = u*scale
    print ("{:40}").format("Stokes solve, time: "), " ==>  ",("{:4f}").format(time.time() - start_time),("{:9}").format("   Its: "), ("{:4}").format(ksp.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    u_k = Function(FunctionSpace(mesh, V))
    p_k = Function(FunctionSpace(mesh, Q))
    u_k.vector()[:] = u.getSubVector(IS[0]).array
    p_k.vector()[:] = u.getSubVector(IS[1]).array
    ones = Function(FunctionSpace(mesh, Q))
    ones.vector()[:]=(0*ones.vector().array()+1)
    p_k.vector()[:] += -assemble(p_k*dx)/assemble(ones*dx)
    return u_k, p_k


def Maxwell(V, Q, F, b0, r0, params, mesh,HiptmairMatrices, Hiptmairtol):
    parameters['reorder_dofs_serial'] = False

    W = V*Q
    W = FunctionSpace(mesh, MixedElement([V, Q]))

    IS = MO.IndexSet(W)

    (b, r) = TrialFunctions(W)
    (c, s) = TestFunctions(W)
    if params[0] == 0.0:
        a11 = params[1]*inner(curl(b), curl(c))*dx
    else:
        a11 = params[1]*params[0]*inner(curl(b), curl(c))*dx
    a21 = inner(b,grad(s))*dx
    a12 = inner(c,grad(r))*dx
    # print F
    L = inner(c, F)*dx
    a = a11+a12+a21

    def boundary(x, on_boundary):
        return on_boundary
    # class b0(Expression):
    #     def __init__(self):
    #         self.p = 1
    #     def eval_cell(self, values, x, ufc_cell):
    #         values[0] = 0.0
    #         values[1] = 1.0
    #     def value_shape(self):
    #         return (2,)

    bcb = DirichletBC(W.sub(0), b0, boundary)
    bcr = DirichletBC(W.sub(1), r0, boundary)
    bc = [bcb, bcr]

    A, b = assemble_system(a, L, bc)
    A, b = CP.Assemble(A, b)
    u = b.duplicate()

    # ksp = PETSc.KSP()
    # ksp.create(comm=PETSc.COMM_WORLD)
    # pc = ksp.getPC()
    # ksp.setType('preonly')
    # pc.setType('lu')
    # OptDB = PETSc.Options()
    # OptDB['pc_factor_mat_solver_package']  = "pastix"
    # OptDB['pc_factor_mat_ordering_type']  = "rcm"
    # ksp.setFromOptions()

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








