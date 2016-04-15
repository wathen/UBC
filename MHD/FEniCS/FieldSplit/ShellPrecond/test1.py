
#!/usr/bin/python
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
Print = PETSc.Sys.Print
# from MatrixOperations import *
from dolfin import *
import numpy as np
import matplotlib.pylab as plt
import os
import scipy.io
from PyTrilinos import Epetra, EpetraExt, AztecOO, ML, Amesos
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
import PETScIO as IO

m = 8
errL2u = np.zeros((m-1,1))
errL2p = np.zeros((m-1,1))
l2uorder = np.zeros((m-1,1))
l2porder = np.zeros((m-1,1))
NN = np.zeros((m-1,1))
DoF = np.zeros((m-1,1))
Vdim = np.zeros((m-1,1))
Qdim = np.zeros((m-1,1))
Wdim = np.zeros((m-1,1))
iterations = np.zeros((m-1,1))
SolTime = np.zeros((m-1,1))
udiv = np.zeros((m-1,1))
nn = 2

dim = 2
Solving = 'Iterative'
ShowResultPlots = 'no'
ShowErrorPlots = 'no'
EigenProblem = 'no'
SavePrecond = 'no'
case = 3
parameters['linear_algebra_backend'] = 'PETSc'

xx = 2
for xx in xrange(1,m):

    nn = 2**(xx)
    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn

    mesh = RectangleMesh(0, 0, 1, 1, nn, nn,'right')

    parameters['reorder_dofs_serial'] = False
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    parameters['reorder_dofs_serial'] = False
    W = V*Q
    Vdim[xx-1] = V.dim()
    Qdim[xx-1] = Q.dim()
    Wdim[xx-1] = W.dim()
    print "\n\nV:  ",Vdim[xx-1],"Q:  ",Qdim[xx-1],"W:  ",Wdim[xx-1],"\n\n"
    def boundary(x, on_boundary):
        return on_boundary


    if case == 1:
        u0 = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
        p0 = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")
    elif case == 2:
        u0 = Expression(("sin(pi*x[1])","sin(pi*x[0])"))
        p0 = Expression("sin(x[1]*x[0])")
    elif case == 3:
        u0 =Expression(("sin(x[1])*exp(x[0])","cos(x[1])*exp(x[0])"))
        p0 = Expression("sin(x[0])*cos(x[1])")
    elif case == 4:
        u0 = Expression(("sin(x[1])*exp(x[0])","cos(x[1])*exp(x[0])"))
        p0 = Expression("sin(x[0])*cos(x[1])")




    bc = DirichletBC(W.sub(0),u0, boundary)
    bcs = [bc]

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    if case == 1:
        f = Expression(("120*x[0]*x[1]*(1-mu)","60*(pow(x[0],2)-pow(x[1],2))*(1-mu)"), mu = 1e0)
    elif case == 2:
        f = Expression(("pi*pi*sin(pi*x[1])+x[1]*cos(x[1]*x[0])","pi*pi*sin(pi*x[0])+x[0]*cos(x[1]*x[0])"))
    elif case == 3:
        f = Expression(("cos(x[0])*cos(x[1])","-sin(x[0])*sin(x[1])"))
    elif case == 4:
        f = Expression(("cos(x[1])*cos(x[0])","-sin(x[1])*sin(x[0])"))




    N = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    alpha = 10.0
    gamma =10.0
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    d = 0
    a11 = inner(grad(v), grad(u))*dx
    a12 = div(v)*p*dx
    a21 = div(u)*q*dx
    L1  =  inner(v,f)*dx
    a = a11-a12-a21

    # (u) = TrialFunctions(V)
    # (v) = TestFunctions(V)

    # (p) = TrialFunctions(Q)
    # (q) = TestFunctions(Q)
    # p11 = inner(grad(v), grad(u))*dx
    i = p*q*dx

    tic()
    AA, bb = assemble_system(a, L1, bcs)

    A = as_backend_type(AA).mat()
    print toc()
    b = bb.array()
    zeros = 0*b
    del bb
    bb = IO.arrayToVec(b)
    x = IO.arrayToVec(zeros)

    PP, Pb = assemble_system(a11+i,L1,bcs)
    P = as_backend_type(PP).mat()



    class ApplyPC(object):

        def __init__(self, W):
            self.W = W


        def create(self, pc):
            self.diag = None
            ksp = PETSc.KSP()
            ksp.create(comm=PETSc.COMM_WORLD)
            pc = ksp.getPC()
            ksp.setType('preonly')
            pc.setType('hypre')
            ksp.setFromOptions()
            self.ksp = ksp
            print ksp.view()
            print W.dim()

        def setUp(self, pc):
            A, B, flag = ksp.getOperators()
            self.B = B
            self.ksp.setOperators(self.B)

        def apply(self, pc, x, y):
            LOG('PCapply.apply()')
            # self.ksp.setOperators(self.B)
            self.ksp.solve(x, y)


    class PC(object):

        def __init__(self, W):
            self.W = W
            self.u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
            self.p_is = PETSc.IS().createGeneral(range(W.sub(0).dim(),W.sub(0).dim()+W.sub(1).dim()))

        def create(self, pc):
            self.diag = None
            kspAMG = PETSc.KSP()
            kspAMG.create(comm=PETSc.COMM_WORLD)
            pc = kspAMG.getPC()
            kspAMG.setType('preonly')
            pc.setType('hypre')
            OptDB = PETSc.Options()
            OptDB["pc_hypre_boomeramg_grid_sweeps_down"] = 2
            OptDB["pc_hypre_boomeramg_grid_sweeps_up"] = 2
            OptDB["pc_hypre_boomeramg_grid_sweeps_coarse"] = 2
            kspAMG.setFromOptions()
            self.kspAMG = kspAMG

            kspCG = PETSc.KSP()
            kspCG.create(comm=PETSc.COMM_WORLD)
            pc = kspCG.getPC()
            kspCG.setType('cg')
            pc.setType('icc')
            kspCG.setFromOptions()
            self.kspCG = kspCG


        def setUp(self, pc):


            A, P, flag = ksp.getOperators()
            self.P11 = P.getSubMatrix(self.u_is,self.u_is)
            self.P22 = P.getSubMatrix(self.p_is,self.p_is)

            self.kspAMG.setOperators(self.P11,self.P11 )
            self.kspCG.setOperators(self.P22,self.P22)


        def apply(self, pc, x, y):
            # LOG('PCapply.apply()')
            # self.kspCG.setOperators(self.B)
            x1 = x.getSubVector(self.u_is)
            y1 = x1.duplicate()
            x2 = x.getSubVector(self.p_is)
            y2 = x2.duplicate()

            self.kspAMG.solve(x1, y1)
            self.kspCG.solve(x2, y2)

            y.array = np.concatenate([y1.array, y2.array])


    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    ksp.setTolerances(1e-10)
    ksp.setType('minres')
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setPythonContext(PC(W))
    ksp.setOperators(A,P)

    # OptDB['pc_python_type'] = '%s.%s' % (module, factory)
    ksp.setFromOptions()

    ksp.solve(bb, x)



    print ksp.its
    r = bb.duplicate()
    A.mult(x, r)
    r.aypx(-1, bb)
    rnorm = r.norm()
    PETSc.Sys.Print('error norm = %g' % rnorm,comm=PETSc.COMM_WORLD)


    if (Solving == 'Iterative' or Solving == 'Direct'):
        if case == 1:
            ue = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
            pe = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)+5")
        elif case == 2:
            ue = Expression(("sin(pi*x[1])","sin(pi*x[0])"))
            pe = Expression("sin(x[1]*x[0])")
        elif case == 3:
            ue =Expression(("sin(x[1])*exp(x[0])","cos(x[1])*exp(x[0])"),cell=triangle)
            pe = Expression("sin(x[0])*cos(x[1])",cell=triangle)
        elif case == 4:
            ue = Expression(("sin(x[1])*exp(x[0])","cos(x[1])*exp(x[0])"))
            pe = Expression("sin(x[0])*cos(x[1])")


        u = interpolate(ue,V)
        p = interpolate(pe,Q)

        Nv  = u.vector().array().shape

        X = IO.vecToArray(x)
        x = X[0:Vdim[xx-1][0]]
        # x = x_epetra[0:Nv[0]]
        ua = Function(V)
        ua.vector()[:] = x
        udiv[xx-1] = assemble(div(ua)*dx)
        pp = X[Nv[0]:]
        pa = Function(Q)

        pa.vector()[:] = pp

        pend = assemble(pa*dx)

        ones = Function(Q)
        ones.vector()[:]=(0*pp+1)
        pp = Function(Q)
        pp.vector()[:] = pa.vector().array()- assemble(pa*dx)/assemble(ones*dx)

        pInterp = interpolate(pe,Q)
        pe = Function(Q)
        pe.vector()[:] = pInterp.vector().array()
        const = - assemble(pe*dx)/assemble(ones*dx)
        pe.vector()[:] = pe.vector()[:]+const

        errL2u[xx-1] = errornorm(ue,ua,norm_type="L2", degree_rise=6,mesh=mesh)
        errL2p[xx-1] = errornorm(pe,pp,norm_type="L2", degree_rise=6,mesh=mesh)



        print errL2u[xx-1]
        print errL2p[xx-1]



# plot(ua)
# plot(interpolate(ue,V))

# plot(pp)
# plot(interpolate(pe,Q))
# interactive()