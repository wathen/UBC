#!/usr/bin/python

# interpolate scalar gradient onto nedelec space

import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
from dolfin import *
import mshr
Print = PETSc.Sys.Print
# from MatrixOperations import *
import numpy as np
import PETScIO as IO
import common
import scipy
import scipy.io
import time

import BiLinear as forms
import IterOperations as Iter
import MatrixOperations as MO
import CheckPetsc4py as CP
import ExactSol
import Solver as S
import MHDmatrixPrecondSetup as PrecondSetup
import NSprecondSetup
import MHDprec as MHDpreconditioner
import memory_profiler
import gc
import MHDmulti
import MHDmatrixSetup as MHDsetup
import scipy.sparse as sp
import sympy as sy

def Domain(n):

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


    left = Left()
    top = Top()
    right = Right()
    bottom = Bottom()
    cleft = CornerLeft()
    ctop = CornerTop()

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
    cleft.mark(boundaries, 2)
    ctop.mark(boundaries, 2)
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

    ru0 = Expression('0.0')

    #Laplacian
    L1 = sy.diff(u,x,x)+sy.diff(u,y,y)
    L2 = sy.diff(v,x,x)+sy.diff(v,y,y)

    A1 = u*sy.diff(u,x)+v*sy.diff(u,y)
    A2 = u*sy.diff(v,x)+v*sy.diff(v,y)

    P1 = sy.diff(p,x)
    P2 = sy.diff(p,y)


    # Curl-curl
    C1 = sy.diff(d,x,y) - sy.diff(b,y,y)
    C2 = sy.diff(b,x,y) - sy.diff(d,x,x)


    NS1 = -d*(sy.diff(d,x)-sy.diff(b,y))
    NS2 = b*(sy.diff(d,x)-sy.diff(b,y))

    M1 = sy.diff(u*d-v*b,y)
    M2 = -sy.diff(u*d-v*b,x)
    print '                                             ', toc()
    # graduu0 = Expression(sy.ccode(sy.diff(u, rho) + (1./rho)*sy.diff(u, phi)))
    # graduu0 = Expression((sy.ccode(sy.diff(u, rho)),sy.ccode(sy.diff(v, rho))))
    tic()
    Laplacian = Expression((sy.ccode(L1),sy.ccode(L2)))
    Advection = Expression((sy.ccode(A1),sy.ccode(A2)))
    gradPres = Expression((sy.ccode(P1),sy.ccode(P2)))
    CurlCurl = Expression((sy.ccode(C1),sy.ccode(C2)))
    gradR = Expression(('0.0','0.0'))
    NS_Couple = Expression((sy.ccode(NS1),sy.ccode(NS2)))
    M_Couple = Expression((sy.ccode(M1),sy.ccode(M2)))
    print '                                             ', toc()

    return uu0, ub0, pu0, pb0, bu0, bb0, ru0, Laplacian, Advection, gradPres, CurlCurl, gradR, NS_Couple, M_Couple



def SolutionMeshSetup(mesh, params,uu0, ub0, pu0, pb0, bu0, bb0, ru0, Laplacian, Advection, gradPres, CurlCurl, gradR, NS_Couple, M_Couple):


    class u0(Expression):
        def __init__(self, mesh, uu0, ub0):
            self.mesh = mesh
            self.u0 = uu0
            self.b0 = ub0
        def eval_cell(self, values, x, ufc_cell):
            if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
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
            if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
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
            if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
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

    class bNone(Expression):
        def __init__(self, mesh, bu0, bb0):
            self.mesh = mesh
            self.b0 = bu0
            self.bb0 = bb0
        def eval_cell(self, values, x, ufc_cell):
            if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
                values[0] = 0.0
                values[1] = 0.0
            else:
                if x[1] < 0:
                    values[0] = 1.
                    values[1] = 0.0
                else:
                    values[0] = 0.0
                    values[1] = 1.
                # print values
        def value_shape(self):
            return (2,)

        def value_shape(self):
            return (2,)


    class r0(Expression):
        def __init__(self, mesh, element=None):
            self.mesh = mesh
        def eval(self, values, x):
            values[0] = 1.0
        # def value_shape(self):
        #     return ( )


    class F_NS(Expression):
        def __init__(self, mesh, Laplacian, Advection, gradPres, NS_Couple, params):
            self.mesh = mesh
            self.Laplacian = Laplacian
            self.Advection = Advection
            self.gradPres = gradPres
            self.NS_Couple = NS_Couple
            self.params = params
        def eval_cell(self, values, x, ufc_cell):
            if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
                values[0] = 0.0
                values[1] = 0.0
            else:
                r = sqrt(x[0]**2 + x[1]**2)
                theta = np.arctan2(x[1],x[0])
                if theta < 0:
                    theta += 2*np.pi

                values[0] =  self.Advection(r,theta)[0] - self.params[0]*self.NS_Couple(r,theta)[0]
                values[1] =  self.Advection(r,theta)[1] - self.params[0]*self.NS_Couple(r,theta)[1]
                # ssss
                # print values

        def value_shape(self):
            return (2,)

    class F_S(Expression):
        def __init__(self, mesh, Laplacian, gradPres, params):
            self.mesh = mesh
            self.Laplacian = Laplacian
            self.gradPres = gradPres
            self.params = params
        def eval_cell(self, values, x, ufc_cell):
                values[0] = 0
                values[1] = 0
                # print r, theta, self.Laplacian(r,theta)

        def value_shape(self):
            return (2,)


        # params[1]*params[0]*CurlCurl+gradR -params[0]*M_Couple
    class F_M(Expression):
        def __init__(self, mesh, CurlCurl, gradR ,M_Couple, params):
            self.mesh = mesh
            self.CurlCurl = CurlCurl
            self.gradR = gradR
            self.M_Couple = M_Couple
            self.params = params
        def eval_cell(self, values, x, ufc_cell):
            if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
                values[0] = 0.0
                values[1] = 0.0
            else:
                r = sqrt(x[0]**2 + x[1]**2)
                theta = np.arctan2(x[1],x[0])
                if theta < 0:
                    theta += 2*np.pi
                values[0] = - self.params[0]*self.M_Couple(r,theta)[0]
                values[1] = - self.params[0]*self.M_Couple(r,theta)[1]

        def value_shape(self):
            return (2,)
    class F_MX(Expression):
        def __init__(self, mesh):
            self.mesh = mesh
        def eval_cell(self, values, x, ufc_cell):
            values[0] = 0.0
            values[1] = 0.0


        def value_shape(self):
            return (2,)


    class Neumann(Expression):
        def __init__(self, mesh, pu0, graduu0, params, n):
            self.mesh = mesh
            self.p0 = pu0
            self.gradu0 = graduu0
            self.params = params
            self.n = n
        def eval_cell(self, values, x, ufc_cell):
            if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
                values[0] = 2.0
                values[1] = 0.0
            else:
                # print x[0], x[1]
                r = sqrt(x[0]**2 + x[1]**2)
                theta = np.arctan2(x[1],x[0])
                if theta < 0:
                    theta += 2*np.pi
                # cell = Cell(self.mesh, ufc_cell.index)
                # print ufc_cell
                # n = cell.normal(ufc_cell.local_facet)
                # n = FacetNormal(self.mesh)
                # print self.n
                # sss
                values[0] = (self.p0(r,theta) - self.params[0]*self.gradu0(r,theta)[021])
                # print -(self.p0(r,theta) - self.params[0]*self.gradu0(r,theta))
                values[1] = -(self.params[0]*self.gradu0(r,theta)[1])

        def value_shape(self):
            return (2,)

    u0 = u0(mesh, uu0, ub0)
    p0 = p0(mesh, pu0, pb0)
    bNone = bNone(mesh, bu0, bb0)
    # p0vec = p0Vec(mesh, pu0)
    b0 = b0(mesh, bu0, bb0)
    r0 = r0(mesh)
    F_NS = F_NS(mesh, Laplacian, Advection, gradPres, NS_Couple, params)
    F_M = F_M(mesh, CurlCurl, gradR, M_Couple, params)
    F_MX = F_MX(mesh)
    F_S = F_S(mesh, Laplacian, gradPres, params)
    # gradu0 = gradu0(mesh, graduu0)
    # Neumann = Neumann(mesh, pu0, graduu0, params, FacetNormal(mesh))
    # NeumannGrad = NeumannGrad(mesh, p0, graduu0, params, FacetNormal(mesh))
    return u0, p0, b0, r0, F_NS, F_M, F_MX, F_S, 1, 1, 1, bNone





#@profile
m = 11

set_log_active(False)
errL2u =np.zeros((m-1,1))
errH1u =np.zeros((m-1,1))
errL2p =np.zeros((m-1,1))
errL2b =np.zeros((m-1,1))
errCurlb =np.zeros((m-1,1))
errL2r =np.zeros((m-1,1))
errH1r =np.zeros((m-1,1))



l2uorder =  np.zeros((m-1,1))
H1uorder =np.zeros((m-1,1))
l2porder =  np.zeros((m-1,1))
l2border =  np.zeros((m-1,1))
Curlborder =np.zeros((m-1,1))
l2rorder =  np.zeros((m-1,1))
H1rorder = np.zeros((m-1,1))

NN = np.zeros((m-1,1))
DoF = np.zeros((m-1,1))
Velocitydim = np.zeros((m-1,1))
Magneticdim = np.zeros((m-1,1))
Pressuredim = np.zeros((m-1,1))
Lagrangedim = np.zeros((m-1,1))
Wdim = np.zeros((m-1,1))
iterations = np.zeros((m-1,1))
SolTime = np.zeros((m-1,1))
udiv = np.zeros((m-1,1))
MU = np.zeros((m-1,1))
level = np.zeros((m-1,1))
NSave = np.zeros((m-1,1))
Mave = np.zeros((m-1,1))
TotalTime = np.zeros((m-1,1))
Dimensions = np.zeros((m-1,4))

def PETSc2Scipy(A):
    row, col, value = A.getValuesCSR()
    return sp.csr_matrix((value, col, row), shape=A.size)

nn = 2
parameters['linear_algebra_backend'] = 'uBLAS'

dim = 2
ShowResultPlots = 'yes'
split = 'Linear'

MU[0]= 1e0
# uu0, ub0, pu0, pb0, bu0, bb0, ru0, Laplacian, Advection, gradPres, CurlCurl, gradR, NS_Couple, M_Couple = SolutionSetUp()

for xx in xrange(1,m):
    print xx
    level[xx-1] = xx + 0
    nn = 2**(level[xx-1])



    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn/2
    parameters["form_compiler"]["quadrature_degree"] = -1
    # parameters = CP.ParameterSetup()
    mesh = UnitSquareMesh(nn,nn)
    # mesh, boundaries, domains = Domain(nn)

    # domain = mshr.Rectangle(Point(0., 0.), Point(1., 2.)) + mshr.Rectangle(Point(1., 0.), Point(2., 1.))
    # mesh = mshr.generate_mesh(domain, nn)
    # mesh, boundaries, domains = Domain(nn)
    # set_log_level(WARNING)

    order = 2
    parameters['reorder_dofs_serial'] = False
    Velocity = VectorFunctionSpace(mesh, "CG", order)
    Pressure = FunctionSpace(mesh, "CG", order-1)
    VecPressure = VectorFunctionSpace(mesh, "CG", order-1)
    Magnetic = FunctionSpace(mesh, "N1curl", order-1)
    Lagrange = FunctionSpace(mesh, "CG", order-1)
    W = MixedFunctionSpace([Velocity, Pressure,Magnetic,Lagrange])
    # W = Velocity*Pressure*Magnetic*Lagrange
    Velocitydim[xx-1] = Velocity.dim()
    Pressuredim[xx-1] = Pressure.dim()
    Magneticdim[xx-1] = Magnetic.dim()
    Lagrangedim[xx-1] = Lagrange.dim()
    Wdim[xx-1] = W.dim()
    print "\n\nW:  ",Wdim[xx-1],"Velocity:  ",Velocitydim[xx-1],"Pressure:  ",Pressuredim[xx-1],"Magnetic:  ",Magneticdim[xx-1],"Lagrange:  ",Lagrangedim[xx-1],"\n\n"
    dim = [Velocity.dim(), Pressure.dim(), Magnetic.dim(), Lagrange.dim()]
    # (  u, p,b, r ) = TrialFunctions(W)
    # ( v, q,c, s) = TestFunctions(W)
    # def boundary(x, on_boundary):
    #     return on_boundary
    # bcu = DirichletBC(W.sub(0),Expression(("0.0","0.0")), boundary)
    # bcb = DirichletBC(W.sub(2),Expression(("0.0","0.0")), boundary)
    # bcr = DirichletBC(W.sub(3),Expression(("0.0")), boundary)
    # bc = [bcu,bcb,bcr]

    # FSpaces = [Velocity,Pressure,Magnetic,Lagrange]

    # kappa = 1.0
    # Mu_m =10.0
    # MU = 1.0

    # N = FacetNormal(mesh)

    # # g = inner(p0*N - MU*grad(u0)*N,v)*dx

    # IterType = 'Full'
    # Split = "No"
    # Saddle = "No"
    # Stokes = "No"
    # SetupType = 'python-class'
    # # F_NS = -MU*Laplacian+Advection+gradPres-kappa*NS_Couple
    # # if kappa == 0:
    # #     F_M = Mu_m*CurlCurl+gradR -kappa*M_Couple
    # # else:
    # #     F_M = Mu_m*kappa*CurlCurl+gradR -kappa*M_Couple
    # # params = [1,1,1]
    # params = [1.,1.,1.]

    # # u0, p0, b0, r0, F_NS, F_M, F_MX, F_S, gradu0, Neumann, p0vec, bNone = SolutionMeshSetup(mesh, params, uu0, ub0, pu0, pb0, bu0, bb0, ru0, Laplacian, Advection, gradPres, CurlCurl, gradR, NS_Couple, M_Couple)


    # # Neumann = interpolate(p0, Pressure)*n - grad(interpolate(u0,Velocity))*n
    # u0 = Expression(('sin(x[1])','sin(x[0])'))
    # n = FacetNormal(mesh)
    # u_k = interpolate(u0,Velocity)
    # p_k = Function(Pressure)
    # b_k = interpolate(u0,Magnetic)
    # r_k = Function(Lagrange)
    # print str(int(level[0][0]))



    # # m11 = params[1]*params[0]*inner(curl(b),curl(c))*dx
    # # m21 = inner(c,grad(r))*dx
    # # m12 = inner(b,grad(s))*dx

    # # a11 = params[2]*inner(grad(v), grad(u))*dx + inner((grad(u)*u_k),v)*dx + (1./2)*div(u_k)*inner(u,v)*dx - (1./2)*inner(u_k,n)*inner(u,v)*ds
    # # a12 = -div(v)*p*dx
    # # a21 = -div(u)*q*dx

    # # CoupleT = params[0]*(v[0]*b_k[1]-v[1]*b_k[0])*curl(b)*dx
    # # Couple = -params[0]*(u[0]*b_k[1]-u[1]*b_k[0])*curl(c)*dx

    # # a = m11 + m12 + m21 + a11 + a21 + a12 + Couple + CoupleT + inner(grad(r),grad(s))*dx
    # # A = assemble(a)

    # # IS = MO.IndexSet(W)
    # # A = CP.Assemble(A)
    # # F = PETSc2Scipy(A.getSubMatrix(IS[0],IS[0]))
    # # F.eliminate_zeros()
    # # F = MO.StoreMatrix(F, 'F'+str(int(level[xx-1][0]))+'_Lshape')
    # # B = PETSc2Scipy(A.getSubMatrix(IS[1],IS[0]))
    # # B.eliminate_zeros()
    # # B = MO.StoreMatrix(B, 'B'+str(int(level[xx-1][0]))+'_Lshape')
    # # C = PETSc2Scipy(A.getSubMatrix(IS[2],IS[0]))
    # # C.eliminate_zeros()
    # # C = MO.StoreMatrix(C, 'C'+str(int(level[xx-1][0]))+'_Lshape')
    # # M = PETSc2Scipy(A.getSubMatrix(IS[2],IS[2]))
    # # M.eliminate_zeros()
    # # M = MO.StoreMatrix(M, 'M'+str(int(level[xx-1][0]))+'_Lshape')
    # # D = PETSc2Scipy(A.getSubMatrix(IS[3],IS[2]))
    # # D.eliminate_zeros()
    # # D = MO.StoreMatrix(D, 'D'+str(int(level[xx-1][0]))+'_Lshape')
    # # L = PETSc2Scipy(A.getSubMatrix(IS[3],IS[3]))
    # # L.eliminate_zeros()
    # # L = MO.StoreMatrix(L, 'L'+str(int(level[xx-1][0]))+'_Lshape')

    # # A = A.sparray()
    # # A.eliminate_zeros()
    # # MO.StoreMatrix(A, "K"+str(int(level[xx-1][0]))+'_Lshape')
    Dimensions[xx-1,:] = np.array([Velocity.dim(), Pressure.dim(), Magnetic.dim(),Lagrange.dim()])


    # # W = MixedFunctionSpace([Magnetic,Lagrange])
    # # bcb = DirichletBC(W.sub(0),Expression(("0.0","0.0")), boundary)
    # # bcr = DirichletBC(W.sub(1),Expression(("0.0")), boundary)

    # # (b, r) = TrialFunctions(W)
    # # (c, s) = TestFunctions(W)
    # # a = params[1]*params[0]*inner(curl(b),curl(c))*dx + inner(c,grad(r))*dx + inner(b,grad(s))*dx
    # # A = assemble(a)

    # # # bcb.apply(A)
    # # # bcr.apply(A)

    # # A = A.sparray()
    # # A.eliminate_zeros()
    # # MO.StoreMatrix(A, "P"+'_Lshape')


    # u  = TrialFunction(Velocity)
    # v = TestFunction(Velocity)
    # p = TrialFunction(Pressure)
    # q = TestFunction(Pressure)
    # b = TrialFunction(Magnetic)
    # c = TestFunction(Magnetic)
    # r = TrialFunction(Lagrange)
    # s = TestFunction(Lagrange)
    # # ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    # M = assemble(inner(curl(b),curl(c))*dx)
    # D = assemble(inner(b,grad(s))*dx)
    # X = assemble(inner(b,c)*dx)

    # L = assemble(inner(grad(r),grad(s))*dx)
    # A = assemble(inner(grad(v), grad(u))*dx)
    # O = assemble(inner((grad(u)*u_k),v)*dx + (1./2)*div(u_k)*inner(u,v)*dx - (1./2)*inner(u_k,n)*inner(u,v)*ds)
    # Q = assemble(inner(u,v)*dx)
    # F = assemble(inner(grad(v), grad(u))*dx + inner((grad(u)*u_k),v)*dx + (1./2)*div(u_k)*inner(u,v)*dx - (1./2)*inner(u_k,n)*inner(u,v)*ds)
    # Q = assemble(inner(u,v)*dx)
    # mat =  as_matrix([[b_k[1]*b_k[1],-b_k[1]*b_k[0]],[-b_k[1]*b_k[0],b_k[0]*b_k[0]]])
    # Qs = assemble(inner(mat*u,v)*dx)
    # Xs = assemble(inner(mat*b,c)*dx)
    # B = assemble(-div(u)*q*dx)

    # Fp = assemble(inner(grad(q), grad(p))*dx(mesh)+inner((u_k[0]*grad(p)[0]+u_k[1]*grad(p)[1]),q)*dx(mesh) + (1./2)*div(u_k)*inner(p,q)*dx(mesh) - (1./2)*(u_k[0]*n[0]+u_k[1]*n[1])*inner(p,q)*ds(mesh))
    # Mp = assemble(inner(q,p)*dx)


    # C = assemble(-params[0]*(u[0]*b_k[1]-u[1]*b_k[0])*curl(c)*dx)

    # Qs = Qs.sparray()
    # Qs.eliminate_zeros()
    # Qs = MO.StoreMatrix(Qs, 'Qs'+str(int(level[xx-1][0])))
    # X = X.sparray()
    # X.eliminate_zeros()
    # X = MO.StoreMatrix(X, 'X'+str(int(level[xx-1][0])))

    # Mp = Mp.sparray()
    # Mp.eliminate_zeros()
    # Mp = MO.StoreMatrix(Mp, 'Mp'+str(int(level[xx-1][0])))
    # Fp = Fp.sparray()
    # Fp.eliminate_zeros()
    # Fp = MO.StoreMatrix(Fp, 'Fp'+str(int(level[xx-1][0])))

    # F = F.sparray()
    # F.eliminate_zeros()
    # F = MO.StoreMatrix(F, 'F'+str(int(level[xx-1][0])))
    # B = B.sparray()
    # B.eliminate_zeros()
    # B = MO.StoreMatrix(B, 'B'+str(int(level[xx-1][0])))
    # C = C.sparray()
    # C.eliminate_zeros()
    # C = MO.StoreMatrix(C, 'C'+str(int(level[xx-1][0])))
    # M = M.sparray()
    # M.eliminate_zeros()
    # M = MO.StoreMatrix(M, 'M'+str(int(level[xx-1][0])))
    # D = D.sparray()
    # D.eliminate_zeros()
    # D = MO.StoreMatrix(D, 'D'+str(int(level[xx-1][0])))
    # L = L.sparray()
    # L.eliminate_zeros()
    # L = MO.StoreMatrix(L, 'L'+str(int(level[xx-1][0])))
    # Q = Q.sparray()
    # Q.eliminate_zeros()
    # Q = MO.StoreMatrix(Q, 'Q'+str(int(level[xx-1][0])))

    # A = A.sparray()
    # A.eliminate_zeros()
    # A = MO.StoreMatrix(A, 'A'+str(int(level[xx-1][0])))
    # O = O.sparray()
    # O.eliminate_zeros()
    # O = MO.StoreMatrix(O, 'O'+str(int(level[xx-1][0])))

    # Xs = Xs.sparray()
    # Xs.eliminate_zeros()
    # Xs = MO.StoreMatrix(Xs, 'Xs'+str(int(level[xx-1][0])))


    # bcu = DirichletBC(Velocity,Expression(("0.0","0.0")), boundary)
    # bcb = DirichletBC(Magnetic,Expression(("0.0","0.0")), boundary)
    # bcr = DirichletBC(Lagrange,Expression(("0.0")), boundary)
    # VelocityBoundary = bcu.get_boundary_values()
    # MagneticBoundary = bcb.get_boundary_values()
    # MultiplierBoundary = bcr.get_boundary_values()

    # np.savetxt('vBoundary'+str(int(level[xx-1][0]))+".t",np.array(VelocityBoundary.keys()))
    # np.savetxt('bBoundary'+str(int(level[xx-1][0]))+".t",np.array(MagneticBoundary.keys()))
    # np.savetxt('rBoundary'+str(int(level[xx-1][0]))+".t",np.array(MultiplierBoundary.keys()))


np.savetxt('dimensions.t',Dimensions)













