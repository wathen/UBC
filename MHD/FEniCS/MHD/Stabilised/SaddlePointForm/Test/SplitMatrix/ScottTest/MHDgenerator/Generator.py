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

def myCCode(A):
    return sy.ccode(A).replace('M_PI','pi')

def Print3D(u,v,w,p,opt):
    if opt == "NS":
        print "  u = (",str(u).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),",",str(v).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),",",str(w).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),")\n"
        print "  p = (",str(p).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),")\n"
    if opt == "M":
        print "  b = (",str(u).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),",",str(v).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),",",str(w).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),")\n"
        print "  r = (",str(p).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),")\n"

def Domain(n):
    mesh = BoxMesh(Point(0., 0., 0.), Point(10., 1., 1.), n, n, n)
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 10.0)

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

    left.mark(boundaries, 2)
    top.mark(boundaries, 1)
    bottom.mark(boundaries, 1)
    #right.mark(boundaries, 1)
    side1.mark(boundaries, 1)
    side2.mark(boundaries, 1)

    return mesh, boundaries, domains

    # class u0(Expression):
    #     def __init__(self):
    #     def eval_cell(self, values, x, ufc_cell):
    #         values[0] = 1.0
    #         values[1] = 0


def ExactSolution(params, B0, delta, x_on, x_off):
    Re = 1./params[2]
    Ha = sqrt(params[0]/(params[1]*params[2]))
    G = 10.

    x = sy.Symbol('x[0]')
    y = sy.Symbol('x[1]')
    z = sy.Symbol('x[2]')

    b = sy.diff(y,x)
    d = (B0/2)*(sy.tanh((x-x_on)/delta)-sy.tanh((x-x_off)/delta))
    e = sy.diff(y,x)


    p = sy.diff(y,x)

    u = sy.diff(x, x)
    v = sy.diff(x, y)
    w = sy.diff(x, y)

    r = sy.diff(x, y)

    u0 = Expression((myCCode(u), myCCode(v), myCCode(w)))
    p0 = Expression(myCCode(p))
    b0 = Expression((myCCode(b), myCCode(d), myCCode(e)))
    r0 = Expression(myCCode(r))


    Print3D(u,v,w,p,"NS")
    Print3D(b,d,e,r,"M")
    # uu = y*x*sy.exp(x+y)
    # u = sy.diff(uu, y)
    #
    # v = -sy.diff(uu, x)

    # p = sy.sin(x)*sy.exp(y)
    # bb = x*y*sy.cos(x)
    # b = sy.diff(bb, y)
    # d = -sy.diff(bb, x
    # )
    # r = x*sy.sin(2*sy.pi*y)*sy.sin(2*sy.pi*x)
    # r = sy.diff(x, y)

    # b = y
    # d = sy.diff(x, y)
    # r = sy.diff(y, y)
    J11 = p - params[2]*sy.diff(u, x)
    J12 = - params[2]*sy.diff(u, y)
    J21 = - params[2]*sy.diff(v, x)
    J22 = p - params[2]*sy.diff(v, y)

    L1 = sy.diff(u,x,x)+sy.diff(u,y,y) + sy.diff(u,z,z)
    L2 = sy.diff(v,x,x)+sy.diff(v,y,y) + sy.diff(v,z,z)
    L3 = sy.diff(w,x,x)+sy.diff(w,y,y) + sy.diff(w,z,z)

    A1 = u*sy.diff(u,x)+v*sy.diff(u,y)+w*sy.diff(u,z)
    A2 = u*sy.diff(v,x)+v*sy.diff(v,y)+w*sy.diff(v,z)
    A3 = u*sy.diff(w,x)+v*sy.diff(w,y)+w*sy.diff(w,z)

    P1 = sy.diff(p, x)
    P2 = sy.diff(p, y)
    P3 = sy.diff(p, z)

    C1 = sy.diff(d,x,y) - sy.diff(b,y,y) - sy.diff(b,z,z) +sy.diff(e,x,z)
    C2 = sy.diff(e,y,z) - sy.diff(d,z,z) - sy.diff(d,x,x) +sy.diff(b,x,y)
    C3 = sy.diff(b,x,z) - sy.diff(e,x,x) - sy.diff(e,y,y) +sy.diff(d,y,z)

    R1 = sy.diff(r,x)
    R2 = sy.diff(r,y)
    R3 = sy.diff(r,z)

    f = u*e-d*w
    g = b*w-u*e
    h = u*d-v*d

    NS1 = sy.diff(h,y)-sy.diff(g,z)
    NS2 = sy.diff(f,z)-sy.diff(h,x)
    NS3 = sy.diff(g,x)-sy.diff(f,y)

    m = sy.diff(e,y)-sy.diff(d,z)
    n = sy.diff(b,z)-sy.diff(e,x)
    p = sy.diff(d,x)-sy.diff(b,y)

    M1 = n*e - d*p
    M2 = b*p - m*e
    M3 = m*d - n*b



    Print3D(-params[2]*L1+A1+P1-params[0]*NS1,-params[2]*L2+A2+P2-params[0]*NS2,-params[2]*L3+A3+P3-params[0]*NS3, p,"NS")
    Print3D(params[0]*params[1]*C1+R1-params[0]*M1,params[0]*params[1]*C2+R2-params[0]*M2,params[0]*params[1]*C3+R3-params[0]*M3,r,"M")

    Laplacian = Expression((myCCode(L1), myCCode(L2), myCCode(L3)))
    Advection = Expression((myCCode(A1), myCCode(A2), myCCode(A3)))
    gradPres = Expression((myCCode(P1), myCCode(P2), myCCode(P3)))
    NScouple = Expression((myCCode(NS1), myCCode(NS2), myCCode(NS3)))

    CurlCurl = Expression((myCCode(C1), myCCode(C2), myCCode(C3)))
    gradLagr = Expression((myCCode(R1), myCCode(R2), myCCode(R3)))
    Mcouple = Expression((myCCode(M1), myCCode(M2), myCCode(M3)))

    # pN = as_matrix(((Expression(myCCode(J11)), Expression(myCCode(J12))), (Expression(myCCode(J21)), Expression(myCCode(J22)))))

    return u0, p0, b0, r0, Laplacian, Advection, gradPres, NScouple, CurlCurl, gradLagr, Mcouple



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

    print b.array
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
