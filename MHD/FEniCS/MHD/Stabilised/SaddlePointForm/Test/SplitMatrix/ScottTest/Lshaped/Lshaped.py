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
import mshr

def Domain(n):

    # defining the L-shaped domain
    domain = mshr.Rectangle(Point(-1., -1.), Point(1., 1.)) - mshr.Rectangle(Point(0., -1.), Point(1., 0.) )
    mesh = mshr.generate_mesh(domain, n)
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


# functions that perform partial derivatives of x and y with respect to polar coordinates
def polarx(u, rho, phi):
    return sy.cos(phi)*sy.diff(u, rho) - (1./rho)*sy.sin(phi)*sy.diff(u, phi)

def polary(u, rho, phi):
    return sy.sin(phi)*sy.diff(u, rho) + (1./rho)*sy.cos(phi)*sy.diff(u, phi)

def polarr(u, x, y):
    return (1./sqrt(x**2 + y**2))*(x*sy.diff(u,x)+y*sy.diff(u,y))

def polart(u, x, y):
    return -y*sy.diff(u,x)+x*sy.diff(u,y)

# def SolutionPolar(mesh, params):

#     l = 0.54448373678246
#     omega = (3./2)*np.pi

#     phi = sy.symbols('x[1]')
#     rho = sy.symbols('x[0]')
#     z = sy.symbols('z')

#     # looked at all the exact solutions and they seems to be the same as the paper.....
#     psi = (sy.sin((1+l)*phi)*sy.cos(l*omega))/(1+l) - sy.cos((1+l)*phi) - (sy.sin((1-l)*phi)*sy.cos(l*omega))/(1-l) + sy.cos((1-l)*phi)
#     psi_prime = sy.diff(psi, phi)
#     psi_3prime = sy.diff(psi, phi, phi, phi)

#     u = rho**l*((1+l)*sy.sin(phi)*psi + sy.cos(phi)*psi_prime)
#     v = rho**l*(-(1+l)*sy.cos(phi)*psi + sy.sin(phi)*psi_prime)

#     uu0 = Expression((sy.ccode(u),sy.ccode(v)))

#     # sssss
#     p = -rho**(l-1)*((1+l)**2*psi_prime + psi_3prime)/(1-l)
#     pu0 = Expression(sy.ccode(p))

#     f = rho**(2./3)*sy.sin((2./3)*phi)
#     # b = sy.diff(f,rho)
#     b = polarx(f, rho, phi)
#     # d = (1./rho)*sy.diff(f,phi)
#     d = polary(f, rho, phi)

#     bu0 = Expression((sy.ccode(b),sy.ccode(d)), degree=4)

#     r = sy.diff(phi,rho)
#     ru0 = Expression(sy.ccode(r), degree=4)


#     # Defining polarx and polary as the x and y derivatives with respect to polar coordinates (rho, phi). Writing the right handside with respect to cartesian coords

#     #Laplacian
#     L1 = polarx(polarx(u, rho, phi), rho, phi) + polary(polary(u, rho, phi), rho, phi)
#     L2 = polarx(polarx(v, rho, phi), rho, phi) + polary(polary(v, rho, phi), rho, phi)

#     # Advection
#     A1 = u*polarx(u, rho, phi)+v*polary(u, rho, phi)
#     A2 = u*polarx(v, rho, phi)+v*polary(v, rho, phi)

#     # Pressure gradient
#     P1 = polarx(p, rho, phi)
#     P2 = polary(p, rho, phi)

#     # Curl-curl
#     C1 = polarx(polary(d, rho, phi), rho, phi) - polary(polary(b, rho, phi), rho, phi)
#     C2 = polarx(polary(b, rho, phi), rho, phi) - polary(polary(d, rho, phi), rho, phi)

#     # Multiplier gradient
#     R1 = sy.diff(r, rho)
#     R2 = sy.diff(r, rho)

#     # Coupling term for fluid variables
#     NS1 = -d*(polarx(d, rho, phi)-polary(b, rho, phi))
#     NS2 = b*(polarx(d, rho, phi)-polary(b, rho, phi))

#     # Coupling term for Magnetic variables
#     M1 = polary(u*d-v*b, rho, phi)
#     M2 = -polarx(u*d-v*b, rho, phi)

#     # Using https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates defintitions of the derivative operators      (sy.diff(u,rho) means partial derivative of u with respect to rho)

#     # Laplacian
#     L11 = (1./rho)*sy.diff(rho*sy.diff(u,rho),rho) + (1./(rho**2))*sy.diff(sy.diff(u,phi),phi) - (1./rho**2)*u - (2./rho**2)*sy.diff(v, phi)
#     L22 = (1./rho)*sy.diff(rho*sy.diff(v,rho),rho) + (1./(rho**2))*sy.diff(sy.diff(v,phi),phi) - (1./rho**2)*v + (2./rho**2)*sy.diff(u, phi)


#     # Advection
#     A11 = u*sy.diff(u, rho) + (1./rho)*v*sy.diff(u, phi) - u**2/rho
#     A22 = u*sy.diff(v, rho) + (1./rho)*v*sy.diff(v, phi) + v*u/rho

#     # Pressure gradient
#     P11 = sy.diff(p, rho)
#     P22 = (1./rho)*sy.diff(p, phi)

#     # Curl-curl
#     c = (1./rho)*(sy.diff(rho*d, rho) - sy.diff(b, phi))
#     C11 = (1./rho)*sy.diff(c, phi)
#     C22 = -sy.diff(c, rho)

#     # Multiplier gradient
#     R11 = sy.diff(r, rho)
#     R22 = sy.diff(r, rho)

#     # Coupling term for fluid variables
#     NS11 = -c*d
#     NS22 = c*b

#     # Coupling term for Magnetic variables
#     c = u*d-v*b
#     M11 = (1./rho)*sy.diff(c, phi)
#     M22 = -sy.diff(c, rho)
#     FF = sy.diff(u, rho) + (1./rho)*sy.diff(v, phi)

#     graduu0 = Expression((sy.ccode(sy.diff(u, rho)),sy.ccode(sy.diff(v, rho))), degree=4)
#     Laplacian = Expression((sy.ccode(L11),sy.ccode(L22)), degree=4)
#     Advection = Expression((sy.ccode(A11),sy.ccode(A22)), degree=4)
#     gradPres = Expression((sy.ccode(P11),sy.ccode(P22)), degree=4)
#     CurlCurl = Expression((sy.ccode(C11),sy.ccode(C22)), degree=4)
#     gradR = Expression((sy.ccode(R11).replace('M_PI','pi'),sy.ccode(R22).replace('M_PI','pi')), degree=4)
#     NS_Couple = Expression((sy.ccode(NS11),sy.ccode(NS22)), degree=4)
#     M_Couple = Expression((sy.ccode(M11),sy.ccode(M22)), degree=4)

#     # ignore this! Just removes the singularity (atan2(0,0) = NaN) and makes all functions zero at the origin
#     class u0(Expression):
#         def __init__(self, mesh, uu0):
#             self.mesh = mesh
#             self.u0 = uu0
#         def eval_cell(self, values, x, ufc_cell):
#             if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
#                 values[0] = 0.0
#                 values[1] = 0.0
#             else:
#                 r = sqrt(x[0]**2 + x[1]**2)
#                 theta = np.arctan2(x[1],x[0])
#                 # print theta
#                 if theta < 0:
#                     theta += 2*np.pi
#                 values[0] = self.u0(r, theta)[0]
#                 values[1] = self.u0(r,theta)[1]
#         def value_shape(self):
#             return (2,)

#     class gradu0(Expression):
#         def __init__(self, mesh, graduu0):
#             self.mesh = mesh
#             self.gradu0 = graduu0
#         def eval_cell(self, values, x, ufc_cell):
#             # if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
#             #     values =  0.0
#             # else:
#                 r = sqrt(x[0]**2 + x[1]**2)
#                 theta = np.arctan2(x[1],x[0])
#                 if theta < 0:
#                     theta += 2*np.pi
#                 values = self.gradu0(r,theta)
#     class p0(Expression):
#         def __init__(self, mesh, pu0):
#             self.mesh = mesh
#             self.p0 = pu0
#         def eval_cell(self, values, x, ufc_cell):
#             if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
#                 values[0] = 0.0
#             else:
#                 r = sqrt(x[0]**2 + x[1]**2)
#                 theta = np.arctan2(x[1],x[0])
#                 if theta < 0:
#                     theta += 2*np.pi
#                 values[0] = self.p0(r,theta)
#         def eval(self, values, x):
#             # if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
#             #     values = 0.0
#             # else:
#                 r = sqrt(x[0]**2 + x[1]**2)
#                 theta = np.arctan2(x[1],x[0])
#                 if theta < 0:
#                     theta += 2*np.pi
#                 values[0] = self.p0(r,theta)
#     class p0Vec(Expression):
#         def __init__(self, mesh, pu0):
#             self.mesh = mesh
#             self.p0 = pu0
#         def eval_cell(self, values, x, ufc_cell):
#             # if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
#             #     values[0] = 0.0
#             # else:
#                 r = sqrt(x[0]**2 + x[1]**2)
#                 theta = np.arctan2(x[1],x[0])
#                 if theta < 0:
#                     theta += 2*np.pi
#                 values[0] = self.p0(r,theta)
#                 values[1] = self.p0(r,theta)
#         def eval(self, values, x):
#             # if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
#             #     values = 0.0
#             # else:
#                 r = sqrt(x[0]**2 + x[1]**2)
#                 theta = np.arctan2(x[1],x[0])
#                 if theta < 0:
#                     theta += 2*np.pi
#                 values[0] = self.p0(r,theta)
#                 values[1] = self.p0(r,theta)
#         # def value_shape(self):
#         #     return (1,)
#     class b0(Expression):
#         def __init__(self, mesh, bu0):
#             self.mesh = mesh
#             self.b0 = bu0
#         def eval_cell(self, values, x, ufc_cell):
#             if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
#                 values[0] = 0.0
#                 values[1] = 0.0
#             else:
#                 r = sqrt(x[0]**2 + x[1]**2)
#                 theta = np.arctan2(x[1],x[0])
#                 if theta < 0:
#                     theta += 2*np.pi
#                 values[0] = self.b0(r, theta)[0]
#                 values[1] = self.b0(r,theta)[1]
#                 # print values
#         def value_shape(self):
#             return (2,)

#     class r0(Expression):
#         def __init__(self, mesh, element=None):
#             self.mesh = mesh
#         def eval(self, values, x):
#             values[0] = 1.0
#         # def value_shape(self):
#         #     return ( )


#     class F_NS(Expression):
#         def __init__(self, mesh, Laplacian, Advection, gradPres, NS_Couple, params):
#             self.mesh = mesh
#             self.Laplacian = Laplacian
#             self.Advection = Advection
#             self.gradPres = gradPres
#             self.NS_Couple = NS_Couple
#             self.params = params
#         def eval_cell(self, values, x, ufc_cell):
#             if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
#                 values[0] = 0.0
#                 values[1] = 0.0
#             else:
#                 r = sqrt(x[0]**2 + x[1]**2)
#                 theta = np.arctan2(x[1],x[0])
#                 if theta < 0:
#                     theta += 2*np.pi

#                 values[0] =  self.Advection(r,theta)[0] - self.params[0]*self.NS_Couple(r,theta)[0]
#                 values[1] =  self.Advection(r,theta)[1] - self.params[0]*self.NS_Couple(r,theta)[1]
#                 # ssss
#                 # print values

#         def value_shape(self):
#             return (2,)

#     class F_S(Expression):
#         def __init__(self, mesh, Laplacian, gradPres, params):
#             self.mesh = mesh
#             self.Laplacian = Laplacian
#             self.gradPres = gradPres
#             self.params = params
#         def eval_cell(self, values, x, ufc_cell):
#                 values[0] = 0
#                 values[1] = 0
#                 # print r, theta, self.Laplacian(r,theta)

#         def value_shape(self):
#             return (2,)


#         # params[1]*params[0]*CurlCurl+gradR -params[0]*M_Couple
#     class F_M(Expression):
#         def __init__(self, mesh, CurlCurl, gradR ,M_Couple, params):
#             self.mesh = mesh
#             self.CurlCurl = CurlCurl
#             self.gradR = gradR
#             self.M_Couple = M_Couple
#             self.params = params
#         def eval_cell(self, values, x, ufc_cell):
#             if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
#                 values[0] = 0.0
#                 values[1] = 0.0
#             else:
#                 r = sqrt(x[0]**2 + x[1]**2)
#                 theta = np.arctan2(x[1],x[0])
#                 if theta < 0:
#                     theta += 2*np.pi
#                 values[0] = - self.params[0]*self.M_Couple(r,theta)[0]
#                 values[1] = - self.params[0]*self.M_Couple(r,theta)[1]

#         def value_shape(self):
#             return (2,)
#     class F_MX(Expression):
#         def __init__(self, mesh):
#             self.mesh = mesh
#         def eval_cell(self, values, x, ufc_cell):
#             values[0] = 0.0
#             values[1] = 0.0


#         def value_shape(self):
#             return (2,)


#     class Neumann(Expression):
#         def __init__(self, mesh, pu0, graduu0, params, n):
#             self.mesh = mesh
#             self.p0 = pu0
#             self.gradu0 = graduu0
#             self.params = params
#             self.n = n
#         def eval_cell(self, values, x, ufc_cell):
#             if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
#                 values[0] = 2.0
#                 values[1] = 0.0
#             else:
#                 # print x[0], x[1]
#                 r = sqrt(x[0]**2 + x[1]**2)
#                 theta = np.arctan2(x[1],x[0])
#                 if theta < 0:
#                     theta += 2*np.pi
#                 # cell = Cell(self.mesh, ufc_cell.index)
#                 # print ufc_cell
#                 # n = cell.normal(ufc_cell.local_facet)
#                 # n = FacetNormal(self.mesh)
#                 # print self.n
#                 # sss
#                 values[0] = (self.p0(r,theta) - self.params[0]*self.gradu0(r,theta)[021])
#                 # print -(self.p0(r,theta) - self.params[0]*self.gradu0(r,theta))
#                 values[1] = -(self.params[0]*self.gradu0(r,theta)[1])

#         def value_shape(self):
#             return (2,)

#     # class NeumannGrad(Expression):
#     #     def __init__(self, mesh, pu0, graduu0, params, n):
#     #         self.mesh = mesh
#     #         self.p0 = pu0
#     #         self.gradu0 = graduu0
#     #         self.params = params
#     #         self.n = n
#     #     def eval_cell(self, values, x, ufc_cell):
#     #         if abs(x[0]) < 1e-3 and abs(x[1]) < 1e-3:
#     #             values[0] = 2.0
#     #             values[1] = 0.0
#     #         else:
#     #             # print x[0], x[1]
#     #             r = sqrt(x[0]**2 + x[1]**2)
#     #             theta = np.arctan2(x[1],x[0])
#     #             if theta < 0:
#     #                 theta += 2*np.pi
#     #             # cell = Cell(self.mesh, ufc_cell.index)
#     #             # print ufc_cell
#     #             # n = cell.normal(ufc_cell.local_facet)
#     #             # n = FacetNormal(self.mesh)
#     #             # print self.n
#     #             # sss
#     #             values[0] = -(self.p0(r,theta) - self.params[0]*self.gradu0(r,theta)[0])
#     #             # print -(self.p0(r,theta) - self.params[0]*self.gradu0(r,theta))
#     #             values[1] = (self.params[0]*self.gradu0(r,theta)[1])

#     #     def value_shape(self):
#     #         return (2,)

#     u0 = u0(mesh, uu0)
#     p0 = p0(mesh, pu0)
#     p0vec = p0Vec(mesh, pu0)
#     b0 = b0(mesh, bu0)
#     r0 = r0(mesh)
#     F_NS = F_NS(mesh, Laplacian, Advection, gradPres, NS_Couple, params)
#     F_M = F_M(mesh, CurlCurl, gradR, M_Couple, params)
#     F_MX = F_MX(mesh)
#     F_S = F_S(mesh, Laplacian, gradPres, params)
#     gradu0 = gradu0(mesh, graduu0)
#     Neumann = Neumann(mesh, pu0, graduu0, params, FacetNormal(mesh))
#     # NeumannGrad = NeumannGrad(mesh, p0, graduu0, params, FacetNormal(mesh))
#     return u0, p0, b0, r0, F_NS, F_M, F_MX, F_S, gradu0, Neumann, p0vec









# def ReadInSol():
#     tic()
#     velocity = open('Solution/velocity.t','r')
#     magnetic = open('Solution/magnetic.t','r')
#     pressure = open('Solution/pressure.t','r')
#     advection = open('Solution/advection.t','r')
#     laplacian = open('Solution/laplacian.t','r')
#     pressureGrad = open('Solution/pressureGrad.t','r')
#     curlcurl = open('Solution/curlcurl.t','r')
#     Mcouple = open('Solution/Mcouple.t','r')
#     NScouple = open('Solution/NScouple.t','r')

#     u = velocity.readline()
#     v = velocity.readline()
#     b = magnetic.readline()
#     d = magnetic.readline()
#     p = pressure.readline()

#     L1 = laplacian.readline()
#     L2 = laplacian.readline()
#     A1 = advection.readline()
#     A2 = advection.readline()
#     P1 = pressureGrad.readline()
#     P2 = pressureGrad.readline()

#     C1 = curlcurl.readline()
#     C2 = curlcurl.readline()
#     M1 = Mcouple.readline()
#     M2 = Mcouple.readline()
#     NS1 = NScouple.readline()
#     NS2 = NScouple.readline()

#     uu0 = Expression(((u),(v)), degree=4)
#     ub0 = Expression((str((u)).replace('atan2(x[1], x[0])','(atan2(x[1], x, [0])+2*pi)'),str((v)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)')), degree=4)
#     pu0 = Expression((p), degree=4)
#     pb0 = Expression(str((p)).replace('atan2(x[1], x[0])','(atan2(x[1], x, [0])+2*pi)'), degree=4)

#     bu0 = Expression(((b),(d)), degree=4)
#     bb0 = Expression((str((b)).replace('atan2(x[1], x[0])','(atan2(x[1], x, [0])+2*pi)'),str((d)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)')), degree=4)
#     ru0 = Expression('0.0', degree=4)

#     Laplacian = Expression(((L1),(L2)), degree=4)
#     Advection = Expression(((A1),(A2)), degree=4)
#     gradPres = Expression(((P1),(P2)), degree=4)
#     CurlCurl = Expression(((C1),(C2)), degree=4)
#     gradR = Expression(('0.0','0.0'), degree=4)
#     NS_Couple = Expression(((NS1),(NS2)), degree=4)
#     M_Couple = Expression(((M1),(M2)), degree=4)
#     print '                                             ', toc()
#     ssss
#     return uu0, ub0, pu0, pb0, bu0, bb0, ru0, Laplacian, Advection, gradPres, CurlCurl, gradR, NS_Couple, M_Couple







def CombineExpression(*values):
    if len(values) == 2:
        u = values[0]
        v = values[1]
        return Expression((("std::abs(x[0]) <= 1e-3 & std::abs(x[1]) <= 1e-3 ? 0 : x[1] < 0.0 ? " + sy.ccode(u) + " : "+ str(sy.ccode(u)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)')), ("std::abs(x[0]) <= 1e-3 & std::abs(x[1]) <= 1e-3 ? 0 : x[1] <= 0.0 ? " + sy.ccode(v) + " : "+ str(sy.ccode(v)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)'))), degree=4)
    else:
        u = values[0]
        return Expression(("std::abs(x[0]) <= 1e-3 & std::abs(x[1]) <= 1e-3 ? 0 : x[1] < 0.0 ? " + sy.ccode(u) + " : "+ str(sy.ccode(u)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)')), degree=4)

def SolutionSetUp():
    tic()
    l = 0.54448373678246
    omega = (3./2)*np.pi

    z = sy.symbols('z')

    x = sy.symbols('x[0]')
    y = sy.symbols('x[1]')
    rho = sy.sqrt(x**2 + y**2)
    phi = sy.atan2(y,x)

    psi = (sy.sin((1+l)*phi)*sy.cos(l*omega))/(1+l) - sy.cos((1+l)*phi) - (sy.sin((1-l)*phi)*sy.cos(l*omega))/(1-l) + sy.cos((1-l)*phi)

    psi_prime = polart(psi, x, y)
    psi_3prime = polart(polart(psi_prime, x, y), x, y)

    u = rho**l*((1+l)*sy.sin(phi)*psi + sy.cos(phi)*psi_prime)
    v = rho**l*(-(1+l)*sy.cos(phi)*psi + sy.sin(phi)*psi_prime)

    uu0 = Expression((sy.ccode(u),sy.ccode(v)), degree=4)
    ub0 = Expression((str(sy.ccode(u)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)'),str(sy.ccode(v)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)')), degree=4)

    p = -rho**(l-1)*((1+l)**2*psi_prime + psi_3prime)/(1-l)
    pu0 = Expression(sy.ccode(p), degree=4)
    pb0 = Expression(str(sy.ccode(p)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)'), degree=4)

    f = rho**(2./3)*sy.sin((2./3)*phi)
    b = sy.diff(f,x)
    d = sy.diff(f,y)
    bu0 = Expression((sy.ccode(b),sy.ccode(d)), degree=4)
    bb0 = Expression((str(sy.ccode(b)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)'),str(sy.ccode(d)).replace('atan2(x[1], x[0])','(atan2(x[1], x[0])+2*pi)')), degree=4)


    # b0 = CombineExpression(b, d)
    # u0 = CombineExpression(u, v)
    # p0 = CombineExpression(p)
    ru0 = Expression('0.0', degree=4)

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
    # graduu0 = Expression(sy.ccode(sy.diff(u, rho) + (1./rho)*sy.diff(u,, degree=4 phi)))
    # graduu0 = Expression((sy.ccode(sy.diff(u, rho)),sy.ccode(sy.diff(v,, degree=4 rho))))

    tic()
    Laplacian = Expression((sy.ccode(L1),sy.ccode(L2)), degree=4)
    Advection = Expression((sy.ccode(A1),sy.ccode(A2)), degree=4)
    gradPres = Expression((sy.ccode(P1),sy.ccode(P2)), degree=4)
    CurlCurl = Expression((sy.ccode(C1),sy.ccode(C2)), degree=4)
    gradR = Expression(('0.0','0.0'), degree=4)
    NS_Couple = Expression((sy.ccode(NS1),sy.ccode(NS2)), degree=4)
    M_Couple = Expression((sy.ccode(M1),sy.ccode(M2)), degree=4)
    print '                                             ', toc()

    # tic()
    # Laplacian = CombineExpression(L1, L2)
    # Advection = CombineExpression(A1, A2)
    # gradPres = CombineExpression(P1, P2)
    # CurlCurl = CombineExpression(C1, C2)
    # gradR = Expression(('0.0','0.0'), degree=4)
    # NS_Couple = CombineExpression(NS1, NS2)
    # M_Couple = CombineExpression(M1, M2)
    # print '                                             ', toc()

    # return u0, p0, b0, r0, Laplacian, Advection, gradPres, CurlCurl, gradR, NS_Couple, M_Couple

    return uu0, ub0, pu0, pb0, bu0, bb0, ru0, Laplacian, Advection, gradPres, CurlCurl, gradR, NS_Couple, M_Couple



def SolutionMeshSetup(mesh, params,uu0, ub0, pu0, pb0, bu0, bb0, ru0, Laplacian, Advection, gradPres, CurlCurl, gradR, NS_Couple, M_Couple):


    class u0(Expression):
        def __init__(self, mesh, uu0, ub0, **kwargs):
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
        def __init__(self, mesh, pu0, pb0, **kwargs):
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
        def __init__(self, mesh, bu0, bb0, **kwargs):
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
        def __init__(self, mesh, bu0, bb0, **kwargs):
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
        def __init__(self, mesh, element=None, **kwargs):
            self.mesh = mesh
        def eval(self, values, x):
            values[0] = 1.0
        # def value_shape(self):
        #     return ( )


    class F_NS(Expression):
        def __init__(self, mesh, Laplacian, Advection, gradPres, NS_Couple,  params, **kwargs):
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
        def __init__(self, mesh, Laplacian, gradPres, params, **kwargs):
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
        def __init__(self, mesh, CurlCurl, gradR ,M_Couple, params, **kwargs):
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
        def __init__(self, mesh, **kwargs):
            self.mesh = mesh
        def eval_cell(self, values, x, ufc_cell):
            values[0] = 0.0
            values[1] = 0.0


        def value_shape(self):
            return (2,)


    class Neumann(Expression):
        def __init__(self, mesh, pu0, graduu0, params, n, **kwargs):
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

    u0 = u0(mesh, uu0, ub0, degree=4)
    p0 = p0(mesh, pu0, pb0, degree=4)
    bNone = bNone(mesh, bu0, bb0, degree=4)
    # p0vec = p0Vec(mesh, pu0, degree=4)
    b0 = b0(mesh, bu0, bb0, degree=4)
    r0 = r0(mesh, degree=4)
    F_NS = F_NS(mesh, Laplacian, Advection, gradPres, NS_Couple, params, degree=4)
    F_M = F_M(mesh, CurlCurl, gradR, M_Couple, params, degree=4)
    F_MX = F_MX(mesh, degree=4)
    F_S = F_S(mesh, Laplacian, gradPres, params, degree=4)
    # gradu0 = gradu0(mesh, graduu0)
    # Neumann = Neumann(mesh, pu0, graduu0, params, FacetNormal(mesh))
    # NeumannGrad = NeumannGrad(mesh, p0, graduu0, params, FacetNormal(mesh))
    return u0, p0, b0, r0, F_NS, F_M, F_MX, F_S, 1, 1, 1, bNone









# Sets up the initial guess for the MHD problem
def Stokes(V, Q, F, u0, p0, gradu0, params,boundaries, domains, mesh):
    parameters['reorder_dofs_serial'] = False

    W = FunctionSpace(mesh, MixedElement([V, Q]))

    IS = MO.IndexSet(W)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    dx = Measure('dx', domain=mesh)
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    n = FacetNormal(W.mesh())

    a11 = params[2]*inner(grad(v), grad(u))*dx('everywhere')
    a12 = -div(v)*p*dx('everywhere')
    a21 = -div(u)*q*dx('everywhere')
    a = a11+a12+a21

    L = inner(v, F)*dx('everywhere') #+ inner(gradu0,v)*ds(2)

    pp = params[2]*inner(grad(v), grad(u))*dx(0) + (1./params[2])*p*q*dx(0)
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

    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    pc = ksp.getPC()
    ksp.setType('preonly')
    pc.setType('lu')
    OptDB = PETSc.Options()
    # if __version__ != '1.6.0':
    OptDB['pc_factor_mat_solver_package']  = "pastix"
    OptDB['pc_factor_mat_ordering_type']  = "rcm"
    ksp.setFromOptions()
    ksp.setOperators(A,A)

    # ksp = PETSc.KSP().create()
    # ksp.setTolerances(1e-8)
    # ksp.max_it = 200
    # pc = ksp.getPC()
    # pc.setType(PETSc.PC.Type.PYTHON)
    # ksp.setType('minres')
    # pc.setPythonContext(StokesPrecond.Approx(W, 1))
    # ksp.setOperators(A,P)

    scale = b.norm()
    b = b/scale
    del A
    start_time = time.time()
    ksp.solve(b,u)
    # Mits +=dodim
    u = u*scale
    print ("{:40}").format("Stokes solve, time: "), " ==>  ",("{:4f}").format(time.time() - start_time),("{:9}").format("   Its: "), ("{:4}").format(ksp.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    # Mits +=dodim
    u = u*scale
    u_k = Function(FunctionSpace(mesh, V))
    p_k = Function(FunctionSpace(mesh, Q))
    u_k.vector()[:] = u.getSubVector(IS[0]).array
    p_k.vector()[:] = u.getSubVector(IS[1]).array
    ones = Function(FunctionSpace(mesh, Q))
    ones.vector()[:]=(0*ones.vector().array()+1)
    p_k.vector()[:] += -assemble(p_k*dx('everywhere'))/assemble(ones*dx('everywhere'))
    return u_k, p_k


def Maxwell(V, Q, F, b0, r0, params, boundaries, bNone, mesh, HiptmairMatrices, Hiptmairtol):
    parameters['reorder_dofs_serial'] = False

    W = V*Q
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

    bcb1 = DirichletBC(W.sub(0), b0, boundaries,1)
    bcb2 = DirichletBC(W.sub(0), Expression(("0.0","0.0"), degree=4), boundaries,2)
    bcb3 = DirichletBC(W.sub(0), bNone, boundaries,2)
    bcb4 = DirichletBC(W.sub(0), b0, boundaries,2)

    bcr = DirichletBC(W.sub(1), r0, boundary)
    bc = [bcb1,bcb2, bcr]

    A, b = assemble_system(a, L, bc)
    A, b = CP.Assemble(A, b)
    u = b.duplicate()

    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    pc = ksp.getPC()
    ksp.setType('preonly')
    pc.setType('lu')
    OptDB = PETSc.Options()
    OptDB['pc_factor_mat_solver_package']  = "umfpack"
    OptDB['pc_factor_mat_ordering_type']  = "rcm"
    ksp.setFromOptions()

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
    u = u*scale
    b_k = Function(FunctionSpace(mesh, V))
    r_k = Function(FunctionSpace(mesh, Q))
    b_k.vector()[:] = u.getSubVector(IS[0]).array
    r_k.vector()[:] = u.getSubVector(IS[1]).array

    return b_k, r_k

