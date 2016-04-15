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

def Domain(n):

    # defining the L-shaped domain
    if __version__ == '1.6.0':
        mesh = RectangleMesh(Point(-1., -1.), Point(1., 1.),n,n)
    else:
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

    # cell_markers = CellFunction("bool", mesh)
    # cell_markers.set_all(False)
    # origin = Point(0., 0.)
    # for cell in cells(mesh):
    #     p = cell.midpoint()
    #     if abs(p.distance(origin)) < 0.6:
    #         cell_markers[cell] = True

    # mesh = refine(mesh, cell_markers)


    # cell_markers = CellFunction("bool", mesh)
    # cell_markers.set_all(False)
    # origin = Point(0., 0.)
    # for cell in cells(mesh):
    #     p = cell.midpoint()
    #     if abs(p.distance(origin)) < 0.4:
    #         cell_markers[cell] = True

    # mesh = refine(mesh, cell_markers)


    # cell_markers = CellFunction("bool", mesh)
    # cell_markers.set_all(False)
    # origin = Point(0., 0.)
    # for cell in cells(mesh):
    #     p = cell.midpoint()
    #     if abs(p.distance(origin)) < 0.2:
    #         cell_markers[cell] = True

    # mesh = refine(mesh, cell_markers)



    # Creating classes that define the boundary of the domain
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

    class CornerTop(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.0) and between(x[0], (0.0,1.0))

    class CornerLeft(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0) and between(x[1], (-1.0,0.0))

    class PointAssemble(SubDomain):
        def inside(self, x, on_boundary):
            return (between(x[0], (0.5-0.5, 0.5+0.5)) and between(x[1], (0.5-0.5, 0.5+0.5)))

    left = Left()
    top = Top()
    right = Right()
    bottom = Bottom()
    cleft = CornerLeft()
    ctop = CornerTop()
    PointAssemble = PointAssemble()

    # Initialize mesh function for the domain
    domains = CellFunction("size_t", mesh)
    domains.set_all(0)
    # PointAssemble.mark(domains,1)

    # Initialize mesh function for boundary domains
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)

    top.mark(boundaries, 1)
    bottom.mark(boundaries, 1)
    left.mark(boundaries, 1)
    cleft.mark(boundaries, 1)
    ctop.mark(boundaries, 1)
    right.mark(boundaries, 2)

    return mesh, boundaries, domains




def polarx(u, rho, phi):
    return sy.cos(phi)*sy.diff(u, rho) - (1./rho)*sy.sin(phi)*sy.diff(u, phi)

def polary(u, rho, phi):
    return sy.sin(phi)*sy.diff(u, rho) + (1./rho)*sy.cos(phi)*sy.diff(u, phi)

def polarr(u, x, y):
    return (1./sqrt(x**2 + y**2))*(x*sy.diff(u,x)+y*sy.diff(u,y))

def polart(u, x, y):
    return -y*sy.diff(u,x)+x*sy.diff(u,y)

def SolutionSetUp():
    tic()
    l = 0.54448373678246
    omega = (3./2)*np.pi

    z = sy.symbols('z')

    x = sy.symbols('x[0]')
    y = sy.symbols('x[1]')
    rho = sy.sqrt(x**2 + y**2)
    phi = sy.atan2(y,x)

    # looked at all the exact solutions and they seems to be the same as the paper.....
    psi = (sy.sin((1+l)*phi)*sy.cos(l*omega))/(1+l) - sy.cos((1+l)*phi) - (sy.sin((1-l)*phi)*sy.cos(l*omega))/(1-l) + sy.cos((1-l)*phi)

    psi_prime = polart(psi, x, y)
    psi_3prime = polart(polart(psi_prime, x, y), x, y)

    u = rho**l*((1+l)*sy.sin(phi)*psi + sy.cos(phi)*psi_prime)
    v = rho**l*(-(1+l)*sy.cos(phi)*psi + sy.sin(phi)*psi_prime)

    uu0 = Expression((sy.ccode(u),sy.ccode(v)))
    ub0 = Expression((str(sy.ccode(u)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)'),str(sy.ccode(v)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)')))

    p = -rho**(l-1)*((1+l)**2*psi_prime + psi_3prime)/(1-l)
    pu0 = Expression(sy.ccode(p))
    pb0 = Expression(str(sy.ccode(p)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)'))

    f = rho**(2./3)*sy.sin((2./3)*phi)
    b = sy.diff(f,x)
    d = sy.diff(f,y)

    bu0 = Expression((sy.ccode(b),sy.ccode(d)))
    bb0 = Expression((str(sy.ccode(b)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)'),str(sy.ccode(d)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)')))

    r = sy.diff(x,y)
    ru0 = Expression(sy.ccode(r))

    A1 = u*sy.diff(u, x) + v*sy.diff(u, y)
    A2 = u*sy.diff(v, x) + v*sy.diff(v, y)

    AdvectionU = Expression((sy.ccode(A1),sy.ccode(A2)))
    AdvectionB = Expression((str(sy.ccode(A1)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)'),str(sy.ccode(A2)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)')))

    M = -(u*d-v*b)
    Mu = Expression((sy.ccode(M)))
    Mb = Expression((str(sy.ccode(M)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)')))
    return uu0, ub0, pu0, pb0, bu0, bb0, ru0, AdvectionU, AdvectionB, Mu, Mb


def SolutionMeshSetup(mesh, params,uu0, ub0, pu0, pb0, bu0, bb0, AdvectionU, AdvectionB, Mu, Mb):


    class u0(Expression):
        def __init__(self, mesh, uu0, ub0):
            self.mesh = mesh
            self.u0 = uu0
            self.b0 = ub0
        def eval_cell(self, values, x, ufc_cell):
            if abs(x[0]) < 1e-8 and abs(x[1]) < 1e-8:
                values[0] = 0.0
                values[1] = 0.0
            else:
                if x[1] < 0:
                    values[0] = self.b0(x[0], x[1])[0]
                    values[1] = self.b0(x[0], x[1])[1]
                else:
                    values[0] = self.u0(x[0], x[1])[0]
                    values[1] = self.u0(x[0], x[1])[1]
        def value_shape(self):
            return (2,)

    class p0(Expression):
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

    class r0(Expression):
        def __init__(self, mesh, element=None):
            self.mesh = mesh
        def eval(self, values, x):
            values[0] = 0.0

    class Advection(Expression):
        def __init__(self, mesh, AdvectionU, AdvectionB):
            self.mesh = mesh
            self.u0 = AdvectionU
            self.b0 = AdvectionB
        def eval_cell(self, values, x, ufc_cell):
            if abs(x[0]) < 1e-8 and abs(x[1]) < 1e-8:
                values[0] = 0.0
                values[1] = 0.0
            else:
                if x[1] < 0:
                    values[0] = self.b0(x[0], x[1])[0]
                    values[1] = self.b0(x[0], x[1])[1]
                else:
                    values[0] = self.u0(x[0], x[1])[0]
                    values[1] = self.u0(x[0], x[1])[1]
        def value_shape(self):
            return (2,)

    class Mcouple(Expression):
        def __init__(self, mesh, Mu, Mb):
            self.mesh = mesh
            self.p0 = Mu
            self.b0 = Mb
        def eval_cell(self, values, x, ufc_cell):
            if abs(x[0]) < 1e-8 and abs(x[1]) < 1e-8:
                values[0] = 0.0
            else:
                if x[1] < 0:
                    values[0] = self.b0(x[0], x[1])
                else:
                    values[0] = self.p0(x[0], x[1])

    u0 = u0(mesh, uu0, ub0)
    p0 = p0(mesh, pu0, pb0)
    b0 = b0(mesh, bu0, bb0)
    r0 = r0(mesh)

    Advection = Advection(mesh, AdvectionU, AdvectionB)
    Mcouple = Mcouple(mesh, Mu, Mb)

    return u0, p0, b0, r0, Advection, Mcouple







# Setting up the Stokes initial guess
def Stokes(V, Q, F, u0, Neumann, params, boundaries):
    parameters['reorder_dofs_serial'] = False

    W = V*Q
    IS = MO.IndexSet(W)
    mesh = W.mesh()

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    n = FacetNormal(W.mesh())
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    a11 = params[2]*inner(grad(v), grad(u))*dx('everywhere')
    a12 = -div(v)*p*dx('everywhere')
    a21 = -div(u)*q*dx('everywhere')
    a = a11+a12+a21

    L = inner(v, F)*dx('everywhere') #+ inner(Neumann, v)*ds(2)


    def boundary(x, on_boundary):
        return on_boundary

    bcu = DirichletBC(W.sub(0), u0, boundary)

    A, b = assemble_system(a, L, bcu)
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

    scale = b.norm()
    b = b/scale
    ksp.setOperators(A,A)
    del A
    ksp.solve(b,u)
    # Mits +=dodim
    u = u*scale
    u_k = Function(V)
    p_k = Function(Q)
    u_k.vector()[:] = u.getSubVector(IS[0]).array
    p_k.vector()[:] = u.getSubVector(IS[1]).array
    ones = Function(Q)
    ones.vector()[:]=(0*ones.vector().array()+1)
    p_k.vector()[:] += -assemble(p_k*dx('everywhere'))/assemble(ones*dx('everywhere'))
    return u_k, p_k


# Setting up the Maxwell initial guess
def Maxwell(V, Q, F, b0, r0, params, boundaries):
    parameters['reorder_dofs_serial'] = False

    W = V*Q
    IS = MO.IndexSet(W)

    mesh = W.mesh()

    # dx = Measure('dx', domain=mesh)
    print params
    (b, r) = TrialFunctions(W)
    (c, s) = TestFunctions(W)

    a11 = params[1]*params[2]*inner(curl(b), curl(c))*dx('everywhere')
    a21 = inner(b,grad(s))*dx('everywhere')
    a12 = inner(c,grad(r))*dx('everywhere')
    L = inner(curl(c), F)*dx('everywhere')
    a = a11+a12+a21

    def boundary(x, on_boundary):
        return on_boundary

    bcb1 = DirichletBC(W.sub(0), b0, boundary)
    bcr = DirichletBC(W.sub(1), r0, boundary)
    bc = [bcb1, bcr]

    A, b = assemble_system(a, L, bc)
    # MO.StoreMatrix(A.sparray(),"A"+str(W.dim()))
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
    scale = b.norm()
    b = b/scale
    ksp.setOperators(A,A)
    del A
    ksp.solve(b,u)
    u = u*scale

    b_k = Function(V)
    r_k = Function(Q)
    b_k.vector()[:] = u.getSubVector(IS[0]).array
    r_k.vector()[:] = u.getSubVector(IS[1]).array

    return b_k, r_k



def Errors(X,mesh,FSpaces,ExactSolution,k,dim):

    Vdim = dim[0]
    Pdim = dim[1]
    Mdim = dim[2]
    Rdim = dim[3]
    # k +=2
    VelocityE = VectorFunctionSpace(mesh,"CG",6)
    u = interpolate(ExactSolution[0],VelocityE)

    PressureE = FunctionSpace(mesh,'CG',5)
    parameters["form_compiler"]["quadrature_degree"] = 10

    xu = X[0:Vdim]
    ua = Function(FSpaces[0])
    ua.vector()[:] = xu

    pp = X[Vdim:Vdim+Pdim]
    pa = Function(FSpaces[1])
    pa.vector()[:] = pp
    pend = assemble(pa*dx)

    ones = Function(FSpaces[1])
    ones.vector()[:]=(0*pp+1)
    pp = Function(FSpaces[1])
    pp.vector()[:] = pa.vector().array()- assemble(pa*dx)/assemble(ones*dx)

    pInterp = interpolate(ExactSolution[1],PressureE)
    pe = Function(PressureE)
    pe.vector()[:] = pInterp.vector().array()
    const = - assemble(pe*dx)/assemble(ones*dx)
    pe.vector()[:] = pe.vector()[:]+const



    ErrorU = Function(FSpaces[0])
    ErrorP = Function(FSpaces[1])

    ErrorU = u-ua
    ErrorP = pe-pp
    errL2u= sqrt(abs(assemble(inner(ErrorU, ErrorU)*dx('everywhere'))))
    errH1u= sqrt(abs(assemble(inner(grad(ErrorU), grad(ErrorU))*dx('everywhere'))))
    errL2p= sqrt(abs(assemble(inner(ErrorP, ErrorP)*dx('everywhere'))))

    MagneticE = FunctionSpace(mesh,"N1curl",5)
    LagrangeE = FunctionSpace(mesh,"CG",5)
    b = interpolate(ExactSolution[2],MagneticE)
    r = interpolate(ExactSolution[3],LagrangeE)


    xb = X[Vdim+Pdim:Vdim+Pdim+Mdim]
    ba = Function(FSpaces[2])
    ba.vector()[:] = xb

    xr = X[Vdim+Pdim+Mdim:]
    ra = Function(FSpaces[3])
    ra.vector()[:] = xr


    ErrorB = Function(FSpaces[2])
    ErrorR = Function(FSpaces[3])


    ErrorB = b-ba
    ErrorR = r-ra

    V = FunctionSpace(mesh, 'CG', 3)
    v = project(curl(b),V)
    # print v.vector().array()



    print '               Exact solution curl   ', assemble(curl(b)*curl(b)*dx('everywhere')), '    assemble(curl(b)*curl(b)*dx)'
    print '               Exact solution curl   ', assemble(curl(b)*dx('everywhere')), '    assemble(curl(b)*dx)'
    print '               Approx solution curl  ', assemble(curl(ba)*curl(ba)*dx('everywhere')), '    assemble(curl(ba)*curl(ba)*dx)'
    print '               Error curl            ', assemble(curl(ErrorB)*dx('everywhere')), '    assemble(curl(ErrorB)*dx)'
    print '               Error curl-curl       ', assemble(curl(ErrorB)*curl(ErrorB)*dx('everywhere')), '    assemble(curl(ErrorB)*curl(ErrorB)*dx)'
    print '               Error inner curl-curl ', assemble(inner(curl(ErrorB),curl(ErrorB))*dx('everywhere')), '    assemble(inner(curl(ErrorB),curl(ErrorB))*dx)'
    errL2b= sqrt(abs(assemble(inner(ErrorB, ErrorB)*dx('everywhere'))))
    errCurlb = errornorm(b, ba, norm_type='HCurl', degree_rise=8)
    errL2r= sqrt(abs(assemble(inner(ErrorR, ErrorR)*dx('everywhere'))))
    errH1r= sqrt(abs(assemble(inner(grad(ErrorR), grad(ErrorR))*dx('everywhere'))))


    # errL2b= errornorm(b, ba, norm_type='L2', degree_rise=8)
    # errCurlb = sqrt(abs(assemble(curl(ba)*curl(ba)*dx)))
    # errL2r= errornorm(r, ra, norm_type='L2', degree_rise=8)
    # errH1r= errornorm(r, ra, norm_type='H10', degree_rise=8)

    return errL2u, errH1u, errL2p, errL2b, errCurlb, errL2r, errH1r


def PandasFormat(table,field,format):
    table[field] = table[field].map(lambda x: format %x)
    return table













