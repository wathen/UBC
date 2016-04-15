
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
from scipy.sparse import  spdiags
import CavityDomain3d as domain
import CavityInitial as initial

parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
#MO.SwapBackend('epetra')
#os.system("echo $PATH")
m = 4
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
Solver = 'PCD'
Saving = 'no'
case = 1
# parameters['linear_algebra_backend'] = 'uBLAS'
parameters = CP.ParameterSetup()
def LOG(arg):
    if INFO:
        print(arg)




for xx in xrange(1,m):
    print xx
    nn = 2**(xx)
    # Create mesh and define function space
    nn = int(nn)
    # print nn
    NN[xx-1] = nn

    mesh, boundaries = domain.CavityMesh3d(nn)
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

    def boundary(x, on_boundary):
        return on_boundary

    if case == 1:
        u0 =Expression(("sin(x[1])*exp(x[0])","cos(x[1])*exp(x[0])","0"))
        p0 = Expression("sin(x[0])*cos(x[1])")
        # u0 = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
        # p0 = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")
    # elif case == 2:
    #     u0 = Expression(("sin(x[1])*exp(x[0])","cos(x[1])*exp(x[0])"))
    #     p0 = Expression("sin(x[0])*cos(x[1])")

    R = 100
    MU = Constant(1e0)
    # MU = 2/R
    bcc = DirichletBC(W.sub(0),u0, boundary)
    bcs = [bcc]

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    if case == 1:
        Laplacian = -MU*Expression(("0","0"))
        Advection = Expression(("pow(exp(x[0]),2)","0"))
        gradPres = Expression(("cos(x[1])*cos(x[0])","-sin(x[1])*sin(x[0])"))

        f = Expression(("0","0","0"))
        # f = Expression(("120*x[0]*x[1]*(1-mu)+ 400*x[0]*pow(x[1],6)+(5*pow(x[0],4)-5*pow(x[1],4))*60*x[0]*x[1]*x[1]","60*(pow(x[0],2)-pow(x[1],2))*(1-mu)+400*pow(x[0],4)*pow(x[1],3)-(5*pow(x[0],4)-5*pow(x[1],4))*20*x[1]*x[1]*x[1]"), mu = MU)
    # elif case == 2:
    #     Laplacian = -MU*Expression(("0","0"))
    #     Advection = Expression(("pow(exp(x[0]),2)","0"))
    #     gradPres = Expression(("cos(x[1])*cos(x[0])","-sin(x[1])*sin(x[0])"))
    #     f = Laplacian+Advection+gradPres
    u0=[Expression(("1","0","0")),Expression(("0","0","0"))]



    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    d = 0
    u_k,p_k = initial.Stokes(V,Q,u0,f,MU,boundaries)
    # p_k.vector()[:] = p_k.vector().array()
    # u_k = Function(V)
    # p_k = Function(Q)
    uOld = np.concatenate((u_k.vector().array(),p_k.vector().array()), axis=0)
    r = IO.arrayToVec(uOld)
    # ufile = File("u.pvd")
    # ufile << u_k
    # plot(u_k)
    a11 = MU*inner(grad(v), grad(u))*dx + inner((grad(u)*u_k),v)*dx + (1/2)*div(u_k)*inner(u,v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds
    a12 = div(v)*p*dx
    a21 = div(u)*q*dx
    L1  = inner(v, f)*dx
    a = a11-a12-a21


    r11 = MU*inner(grad(v), grad(u_k))*dx + inner((grad(u_k)*u_k),v)*dx + (1/2)*div(u_k)*inner(u_k,v)*dx- (1/2)*inner(u_k,n)*inner(u_k,v)*ds
    r12 = div(v)*p_k*dx
    r21 = div(u_k)*q*dx
    RHSform = r11-r12-r21


    p11 = inner(u,v)*dx
    # p12 = div(v)*p*dx
    # p21 = div(u)*q*dx
    p22 = inner(p,q)*dx
    prec = p11 +p22
    bc = DirichletBC(W.sub(0),Expression(("0","0","0")), boundary)
    bcs = [bc]

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-5       # tolerance
    iter = 0            # iteration counter
    maxiter = 10        # max no of iterations allowed
    parameters = CP.ParameterSetup()
    outerit = 0

    if Solver == "LSC":
        parameters['linear_algebra_backend'] = 'uBLAS'
        BQB = assemble(inner(u,v)*dx - div(v)*p*dx-div(u)*q*dx)
        bc.apply(BQB)
        BQB = BQB.sparray()
        X = BQB[0:V.dim(),0:V.dim()]
        Xdiag = X.diagonal()
        # Xdiag = X.sum(1).A
        # print Xdiag
        B = BQB[V.dim():W.dim(),0:V.dim()]
        Bt = BQB[0:V.dim(),V.dim():W.dim()]
        d = spdiags(1.0/Xdiag, 0, len(Xdiag), len(Xdiag))
        L = B*d*Bt
        Bd = B*d
        dBt = d*Bt
        L = PETSc.Mat().createAIJ(size=L.shape,csr=(L.indptr, L.indices, L.data))
        Bd = PETSc.Mat().createAIJ(size=Bd.shape,csr=(Bd.indptr, Bd.indices, Bd.data))
        dBt = PETSc.Mat().createAIJ(size=dBt.shape,csr=(dBt.indptr, dBt.indices, dBt.data))
        parameters['linear_algebra_backend'] = 'PETSc'
    elif Solver == "PCD":
        (pQ) = TrialFunction(Q)
        (qQ) = TestFunction(Q)
        Mass = assemble(inner(pQ,qQ)*dx)
        L = assemble(inner(grad(pQ),grad(qQ))*dx)

        F = assemble(MU*inner(grad(qQ), grad(pQ))*dx+inner((u_k[0]*grad(pQ)[0]+u_k[1]*grad(pQ)[1]+u_k[2]*grad(pQ)[2]),qQ)*dx + (1/2)*div(u_k)*inner(pQ,qQ)*dx - (1/2)*(u_k[0]*n[0]+u_k[1]*n[1]+u_k[2]*n[2])*inner(pQ,qQ)*ds)
        # print "hi"
        F = CP.Assemble(F)
        L = CP.Assemble(L)
        Mass = CP.Assemble(Mass)


    while eps > tol and iter < maxiter:
        iter += 1
        x = Function(W)

        uu = Function(W)
        tic()
        AA, bb = assemble_system(a, L1-RHSform, bcs)
        A,b = CP.Assemble(AA,bb)
        print toc()
        print A
        # b = b.getSubVector(t_is)
        PP = assemble(prec)
        bcc.apply(PP)
        P = CP.Assemble(PP)


        b = bb.array()
        zeros = 0*b
        bb = IO.arrayToVec(b)
        x = IO.arrayToVec(zeros)
        ksp = PETSc.KSP()
        ksp.create(comm=PETSc.COMM_WORLD)
        ksp.setTolerances(1e-5)
        ksp.setType('gmres')
        pc = ksp.getPC()



        pc.setType(PETSc.PC.Type.PYTHON)
        if Solver == "LSC":
            pc.setPythonContext(NSprecond.LSCnew(W,A,L,Bd,dBt))
        elif Solver == "PCD":
            pc.setPythonContext(NSprecond.PCD(W, A, Mass, F, L))

        ksp.setOperators(A)
        OptDB = PETSc.Options()
        # OptDB['ksp_gmres_restart'] = 200
        # OptDB['pc_factor_mat_ordering_type'] = 'rcm'
        # OptDB['pc_factor_mat_solver_package']  = 'umfpack'
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

        p1.vector()[:] = p1.vector()[:] +  pp
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

    if case == 1:
        ue = u0
        pe = p0
    elif case == 2:
        ue = u0
        pe = p0

    AvIt[xx-1] = np.ceil(outerit/iter)
    # u = interpolate(ue,V)
    # p = interpolate(pe,Q)

    # ua = Function(V)
    # ua.vector()[:] = u_k.vector().array()
    # # nonlinear[xx-1] = assemble(inner((grad(ua)*ua),ua)*dx+(1/2)*div(ua)*inner(ua,ua)*dx- (1/2)*inner(ua,n)*inner(ua,ua)*ds)


    Nv  = u_k.vector().array().shape

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
#
    pend = assemble(pa*dx)

    ones = Function(Q)
    ones.vector()[:]=(0*pp+1)
    pp = Function(Q)
    pp.vector()[:] = pa.vector().array()- assemble(pa*dx)/assemble(ones*dx)

    # pInterp = interpolate(pe,Q)
    # pe = Function(Q)
    # pe.vector()[:] = pInterp.vector().array()
    # const = - assemble(pe*dx)/assemble(ones*dx)
    # pe.vector()[:] = pe.vector()[:]+const

    # errL2u[xx-1] = errornorm(ue,ua,norm_type="L2", degree_rise=4,mesh=mesh)
    # errL2p[xx-1] = errornorm(pe,pp,norm_type="L2", degree_rise=4,mesh=mesh)
    if xx == 1:
        l2uorder[xx-1] = 0
        l2porder[xx-1] = 0
    else:
        l2uorder[xx-1] =  np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1]))
        l2porder[xx-1] =  np.abs(np.log2(errL2p[xx-2]/errL2p[xx-1]))

    print errL2u[xx-1]
    print errL2p[xx-1]
    # del  solver




print nonlinear



print "Velocity Elements rate of convergence ", np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1])))
print "Pressure Elements rate of convergence ", np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1])))


import pandas as pd
tableTitles = ["Total DoF","V DoF","Q DoF","AvIt","V-L2","V-order","P-L2","P-order"]
tableValues = np.concatenate((Wdim,Vdim,Qdim,AvIt,errL2u,l2uorder,errL2p,l2porder),axis=1)
df = pd.DataFrame(tableValues, columns = tableTitles)
pd.set_option('precision',3)
print df
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
# plot(interpolate(ue,V))

plot(pp)
# plot(interpolate(pe,Q))

interactive()

# plt.show()

