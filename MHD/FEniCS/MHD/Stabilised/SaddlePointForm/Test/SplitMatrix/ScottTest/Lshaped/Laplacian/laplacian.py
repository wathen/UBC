#!/usr/bin/python

import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
from dolfin import *
from dolfin import plot as dplot

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
import NSprecondSetup
import memory_profiler
import gc
import sympy as sy
#@profile
m = 7

set_log_active(False)
errL2u =np.zeros((m-1,1))
errH1u =np.zeros((m-1,1))


l2uorder =  np.zeros((m-1,1))
H1uorder =np.zeros((m-1,1))

NN = np.zeros((m-1,1))
Vdim = np.zeros((m-1,1))
level = np.zeros((m-1,1))

nn = 2

dim = 2
ShowResultPlots = 'yes'
split = 'Linear'

for xx in xrange(1,m):
    print xx
    # parameters["form_compiler"]["quadrature_degree"] = -1

    level[xx-1] = xx + 2
    nn = 2**(level[xx-1])



    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn/2

    # domain = mshr.Rectangle(Point(-1., -1.), Point(1., 1.)) - mshr.Rectangle(Point(0., -1.), Point(1., 0.) )
    # mesh = mshr.generate_mesh(domain, nn)


        # set_log_level(WARNING)
    mesh = RectangleMesh(-1,-1,1,1,nn,nn, 'left')
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

    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    origin = Point(0., 0.)
    for cell in cells(mesh):
        p = cell.midpoint()
        if abs(p.distance(origin)) < 0.6:
            cell_markers[cell] = True

    mesh = refine(mesh, cell_markers)


    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    origin = Point(0., 0.)
    for cell in cells(mesh):
        p = cell.midpoint()
        if abs(p.distance(origin)) < 0.4:
            cell_markers[cell] = True

    mesh = refine(mesh, cell_markers)


    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    origin = Point(0., 0.)
    for cell in cells(mesh):
        p = cell.midpoint()
        if abs(p.distance(origin)) < 0.2:
            cell_markers[cell] = True

    mesh = refine(mesh, cell_markers)

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
            return near(x[1], 0.0)

    class CornerLeft(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0)


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


    order = 1
    parameters['reorder_dofs_serial'] = False
    V = FunctionSpace(mesh, "CG", order)
    Vdim[xx-1] = V.dim()

    phi = sy.symbols('x[1]')
    rho = sy.symbols('x[0]')
    u =  pow(rho,2./3)*sy.sin(2*phi/3)
    v =  pow(rho,2./3)*sy.sin(2*phi/3)
    u = Expression((sy.ccode(u),sy.ccode(v)))
    # v = Expression(sy.ccode(v))

    class u0(Expression):
        def __init__(self, mesh, u, v):
            self.mesh = mesh
            self.u = u
            self.v = v
        def eval_cell(self, values, x, ufc_cell):
            if abs(x[0]) < 1e-8 and abs(x[1]) < 1e-8:
                values[0] = 0.0
                # values[1] = 0.0

            else:
                r = sqrt(x[0]**2 + x[1]**2)
                theta = np.arctan2(x[1],x[0])
                # print theta
                if theta < 0:
                    theta += 2*np.pi
                values[0] = self.u(r, theta)[0]
                # values[1] = self.u(r, theta)[1]
        # def value_shape(self):
        #     return (2,)

    u0 = u0(mesh, u, v)

    b = TrialFunction(V)
    c = TestFunction(V)
    a = inner(grad(b), grad(c))*dx
    f = Expression(("0.0"))
    L = inner(f,c)*dx

    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, u0, boundary)
    # bc2 = DirichletBC(V, f, boundaries, 2)
    # bc = [bc1, bc2]
    A, b = assemble_system(a, L, bc)
    A, b = CP.Assemble(A, b)
    u = b.duplicate()

    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    pc = ksp.getPC()
    ksp.setType('cg')
    pc.setType('hypre')
    ksp.setTolerances(1e-8)
    OptDB = PETSc.Options()
    # OptDB['pc_factor_mat_solver_package']  = "mumps"
    # OptDB['pc_factor_mat_ordering_type']  = "rcm"
    ksp.setFromOptions()
    scale = b.norm()
    b = b/scale
    ksp.setOperators(A,A)
    del A
    ksp.solve(b,u)
    # Mits +=dodim
    u = u*scale
    # parameters["form_compiler"]["quadrature_degree"] = 10

    Ve = FunctionSpace(mesh,"CG",order+3)
    ue = interpolate(u0, Ve)

    ua = Function(V)
    ua.vector()[:] = u.array
    ErrorU = ua-ue
    errL2u[xx-1] = sqrt(abs(assemble(inner(ErrorU, ErrorU)*dx)))
    errH1u[xx-1] = sqrt(abs(assemble(inner(grad(ErrorU), grad(ErrorU))*dx)))

    # errL2u[xx-1] = errornorm(ue, ua, norm_type='L2', degree_rise=8)
    # errH1u[xx-1] = errornorm(ue, ua, norm_type='H10', degree_rise=8)

    if xx > 1:
       l2uorder[xx-1] =  np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1])/np.log2(sqrt(float(Vdim[xx-1][0])/Vdim[xx-2][0])))
       H1uorder[xx-1] =  np.abs(np.log2(errH1u[xx-2]/errH1u[xx-1])/np.log2(sqrt(float(Vdim[xx-1][0])/Vdim[xx-2][0])))



import pandas as pd

# dplot(ua, interactive=True)

LatexTitles = ["l","DoFu","V-L2","L2-order","V-H1","H1-order"]
LatexValues = np.concatenate((level,Vdim,errL2u,l2uorder,errH1u,H1uorder), axis=1)
LatexTable = pd.DataFrame(LatexValues, columns = LatexTitles)
pd.set_option('precision',3)
LatexTable = MO.PandasFormat(LatexTable,"V-L2","%2.4e")
LatexTable = MO.PandasFormat(LatexTable,'V-H1',"%2.4e")
LatexTable = MO.PandasFormat(LatexTable,"H1-order","%1.2f")
LatexTable = MO.PandasFormat(LatexTable,'L2-order',"%1.2f")
print LatexTable

