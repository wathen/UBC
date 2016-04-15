#!/opt/local/bin/python

from dolfin import *
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

# from MatrixOperations import *
import numpy as np
import matplotlib.pylab as plt
import os
import scipy.io
#from PyTrilinos import Epetra, EpetraExt, AztecOO, ML, Amesos
#from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
import PETScIO as IO
import time
import common
import CheckPetsc4py as CP
import NSprecond

parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
#MO.SwapBackend('epetra')
#os.system("echo $PATH")
m = 2
errL2u = np.zeros((m-1,1))
errL2p = np.zeros((m-1,1))
NN = np.zeros((m-1,1))
DoF = np.zeros((m-1,1))
Vdim = np.zeros((m-1,1))
Qdim = np.zeros((m-1,1))
Wdim = np.zeros((m-1,1))
l2uorder = np.zeros((m-1,1))
l2porder = np.zeros((m-1,1))
nonlinear = np.zeros((m-1,1))
AvIt = np.zeros((m-1,1))
nn = 2

dim = 2
Solving = 'No'
Saving = 'no'
case = 1
# parameters['linear_algebra_backend'] = 'uBLAS'
parameters = CP.ParameterSetup()
def LOG(arg):
    if INFO:
        print(arg)




for xx in xrange(1,m):
    print xx
    nn = 2**(xx+4)
    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn

    mesh = RectangleMesh(-1, -1, 1, 1, nn, nn,'crossed')
    # tic()

    parameters['reorder_dofs_serial'] = False
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    parameters['reorder_dofs_serial'] = False
    # print 'time to create function spaces', toc(),'\n\n'
    W = V*Q
    Vdim[xx-1] = V.dim()
    Qdim[xx-1] = Q.dim()
    Wdim[xx-1] = W.dim()
    print "\n\nV:  ",Vdim[xx-1],"Q:  ",Qdim[xx-1],"W:  ",Wdim[xx-1],"\n\n"

    class DirichletL(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], -1.0) and between(x[1], (-1.0,1.0))

    class DirichletR(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0],1.0) and between(x[1], (-1.0,1.0))

    class DirichletB(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1],-1.0) and between(x[0], (-1.0,1.0))


    class DirichletT(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1],1.0) and between(x[0], (-1.0,1.0))

    dirichletL = DirichletL()
    dirichletR = DirichletR()
    dirichletB = DirichletB()
    dirichletT = DirichletT()
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)

    dirichletT.mark(boundaries, 1)
    dirichletL.mark(boundaries, 2)
    dirichletR.mark(boundaries, 2)
    dirichletB.mark(boundaries, 2)

    def boundary(x, on_boundary):
        return on_boundary

    if case == 1:
        u0 = Expression(("1-pow(x[0],4)","0"))
        uu0 = Expression(("0","0"))
    elif case == 2:
        u0 = Expression(("sin(x[1])*exp(x[0])","cos(x[1])*exp(x[0])"))
        p0 = Expression("sin(x[0])*cos(x[1])")


    MU = Constant(1e-1)

    bc = DirichletBC(W.sub(0),u0, boundaries,1)
    bc1 = DirichletBC(W.sub(0),uu0, boundaries,2)
    bcs = [bc,bc1]

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    if case == 1:
        f = Expression(("0","0"), mu = 1e0)
    elif case == 2:
        Laplacian = -MU*Expression(("0","0"))
        Advection = Expression(("pow(exp(x[0]),2)","0"))
        gradPres = Expression(("cos(x[1])*cos(x[0])","-sin(x[1])*sin(x[0])"))
        f = Laplacian+Advection+gradPres




    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    d = 0
    u_k,p_k = common.Stokes(V,Q,u0,Expression(("0","0")),[1,1,MU])
    p_k.vector()[:] = p_k.vector().array()
    # u_k = Function(V)
    # p_k = Function(Q)
    uOld = np.concatenate((u_k.vector().array(),p_k.vector().array()), axis=0)
    r = IO.arrayToVec(uOld)

    a11 = MU*inner(grad(v), grad(u))*dx + inner((grad(u)*u_k),v)*dx+(1/2)*div(u_k)*inner(u,v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds
    a12 = div(v)*p*dx
    a21 = -div(u)*q*dx
    L1  = inner(v, f)*dx
    a = a11-a12-a21


    r11 = MU*inner(grad(v), grad(u_k))*dx + inner((grad(u_k)*u_k),v)*dx+(1/2)*div(u_k)*inner(u_k,v)*dx- (1/2)*inner(u_k,n)*inner(u_k,v)*ds
    r12 = div(v)*p_k*dx
    r21 = div(u_k)*q*dx
    RHSform = r11-r12-r21


    p11 = inner(u,v)*dx
    p12 = div(v)*p*dx
    p21 = -div(u)*q*dx
    p22 = inner(p,q)*dx
    prec = p11-p12-p21 +p22
    bc = DirichletBC(W.sub(0),Expression(("0","0")), boundary)
    bcs = [bc]

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-6       # tolerance
    iter = 0            # iteration counter
    maxiter = 10        # max no of iterations allowed
    parameters = CP.ParameterSetup()
    outerit = 0


    (p) = TrialFunction(Q)
    (q) = TestFunction(Q)

    Mass = assemble(inner(p,q)*dx)
    LL = assemble(inner(grad(p),grad(q))*dx)
    Mass = CP.Assemble(Mass)
    L = CP.Assemble(LL)

    # u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
    t_is = PETSc.IS().createGeneral(range(W.dim()-1))
    while eps > tol and iter < maxiter:
        iter += 1
        x = Function(W)

        uu = Function(W)
        tic()
        AA, bb = assemble_system(a, L1-RHSform, bcs)
        A,b = CP.Assemble(AA,bb)
        print toc()
        # A =A.getSubMatrix(t_is,t_is)

        PP = assemble(prec)
        P = CP.Assemble(PP)
        del AA, PP
        FF = assemble(MU*inner(grad(p), grad(q))*dx+inner(inner(grad(p),u_k),q)*dx)
        F = CP.Assemble(FF)


        b = bb.array()
        zeros = 0*b
        bb = IO.arrayToVec(b)
        x = IO.arrayToVec(zeros)
        ksp = PETSc.KSP()
        ksp.create(comm=PETSc.COMM_WORLD)
        ksp.setTolerances(1e-8)
        ksp.setType('gmres')
        pc = ksp.getPC()



        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(NSprecond.PCD(W,A,Mass,F,L))
        ksp.setOperators(A,P)
        OptDB = PETSc.Options()
        # OptDB['pc_factor_shift_amount'] = 1

        ksp.setFromOptions()



        start = time.time()
        ksp.solve(bb, x)

        print time.time() - start
        print ksp.its
        outerit += ksp.its
        # r = bb.duplicate()
        # A.MUlt(x, r)
        # r.aypx(-1, bb)
        # rnorm = r.norm()
        # PETSc.Sys.Print('error norm = %g' % rnorm,comm=PETSc.COMM_WORLD)

        uu = IO.vecToArray(x)
        UU = uu[0:Vdim[xx-1][0]]
        # time = time+toc()
        u1 = Function(V)
        u1.vector()[:] = u1.vector()[:] + UU

        pp = uu[Vdim[xx-1][0]:]
        # time = time+toc()
        p1 = Function(Q)
        n = pp.shape

        p1.vector()[:] = p1.vector()[:] + pp
        diff = u1.vector().array()
        eps = np.linalg.norm(diff, ord=np.Inf)

        print '\n\n\niter=%d: norm=%g' % (iter, eps)
        print np.linalg.norm(p1.vector().array(),ord=np.inf)

        u2 = Function(V)
        u2.vector()[:] = u1.vector().array() + u_k.vector().array()
        p2 = Function(Q)
        p2.vector()[:] = p1.vector().array() + p_k.vector().array()
        u_k.assign(u2)
        p_k.assign(p2)

        uOld = np.concatenate((u_k.vector().array(),p_k.vector().array()), axis=0)
        r = IO.arrayToVec(uOld)

        del A, P, pc, ksp, F, FF



    AvIt[xx-1] = np.ceil(outerit/iter)


    ua = Function(V)
    ua.vector()[:] = u_k.vector().array()
    # nonlinear[xx-1] = assemble(inner((grad(ua)*ua),ua)*dx+(1/2)*div(ua)*inner(ua,ua)*dx- (1/2)*inner(ua,n)*inner(ua,ua)*ds)


    Nv  = ua.vector().array().shape

    X = IO.vecToArray(r)
    x = X[0:Vdim[xx-1][0]]
    # x = x_epetra[0:Nv[0]]
    ua = Function(V)
    ua.vector()[:] = x

    pp = X[Nv[0]:]
    n = pp.shape
    print n
    # pp = np.insert(pp,n,0)
    pa = Function(Q)
    pa.vector()[:] = pp

    pend = assemble(pa*dx)

    ones = Function(Q)
    ones.vector()[:]=(0*pp+1)
    pp = Function(Q)
    pp.vector()[:] = pa.vector().array()- assemble(pa*dx)/assemble(ones*dx)


# plt.loglog(N,erru)
# plt.title('Error plot for P2 elements - convergence = %f' % np.log2(np.average((erru[0:m-2]/erru[1:m-1]))))
# plt.xlabel('N')
# plt.ylabel('L2 error')

# plt.figure()
# plt.loglog(N,errp)
# plt.title('Error plot for P1 elements - convergence = %f' % np.log2(np.average((errp[0:m-2]/errp[1:m-1]))))
# plt.xlabel('N')
# plt.ylabel('L2 error')


plot(ua)

plot(pp)

interactive()

plt.show()

print AvIt
