from dolfin import *
import numpy as np
import time
import CheckPetsc4py as CP
import NSprecond
from scipy.sparse import  spdiags
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import PETScIO as IO

m =6
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



for xx in xrange(1,m):
    print xx
    nn = 2**(xx)
    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn

    mesh = RectangleMesh(-1, -1, 1, 1, nn, nn,'right')
    # tic()

    parameters['reorder_dofs_serial'] = False
    V = FunctionSpace(mesh, "BDM", 2)
    Q = FunctionSpace(mesh, "DG", 1)
    parameters['reorder_dofs_serial'] = False
    # print 'time to create function spaces', toc(),'\n\n'
    W = V*Q
    Vdim[xx-1] = V.dim()
    Qdim[xx-1] = Q.dim()
    Wdim[xx-1] = W.dim()
    print "\n\nV:  ",Vdim[xx-1],"Q:  ",Qdim[xx-1],"W:  ",Wdim[xx-1],"\n\n"

    def boundary(x, on_boundary):
        return on_boundary


    u0 =Expression(("sin(x[1])*exp(x[0])","cos(x[1])*exp(x[0])"))
    p0 = Expression("sin(x[0])*cos(x[1])")


    bc = DirichletBC(W.sub(0),u0, boundary)
    bcs = [bc]

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    MU = Constant(1e0)

    Laplacian = -MU*Expression(("0","0"))
    Advection = Expression(("pow(exp(x[0]),2)","0"))
    gradPres = Expression(("cos(x[1])*cos(x[0])","-sin(x[1])*sin(x[0])"))

    f = Laplacian+Advection+gradPres



    u_k = Function(V)
    p_k = Function(Q)

    u_k.vector()[:] = u_k.vector()[:]*0
    N = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    alpha = 10.0
    gamma =10.0
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)

    a11 = inner(grad(v), grad(u))*dx \
        - inner(avg(grad(v)), outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
        - inner(outer(v('+'),N('+'))+outer(v('-'),N('-')), avg(grad(u)))*dS \
        + alpha/h_avg*inner(outer(v('+'),N('+'))+outer(v('-'),N('-')),outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
        - inner(outer(v,N), grad(u))*ds \
        - inner(grad(v), outer(u,N))*ds \
        + gamma/h*inner(v,u)*ds

    O = inner((grad(u)*u_k),v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds \
     -(1/2)*(inner(u_k('+'),N('+'))+inner(u_k('-'),N('-')))*avg(inner(u,v))*ds \
    -dot(avg(v),dot(outer(u('+'),N('+'))+outer(u('-'),N('-')),avg(u_k)))*dS

    a12 = div(v)*p*dx
    a21 = div(u)*q*dx
    L1  = inner(v, f)*dx + gamma/h*inner(u0,v)*ds - inner(grad(v),outer(u0,N))*ds
    a = a11+O-a12-a21

    r11 = inner(grad(v), grad(u_k))*dx \
        - inner(avg(grad(v)), outer(u_k('+'),N('+'))+outer(u_k('-'),N('-')))*dS \
        - inner(outer(v('+'),N('+'))+outer(v('-'),N('-')), avg(grad(u_k)))*dS \
        + alpha/h_avg*inner(outer(v('+'),N('+'))+outer(v('-'),N('-')),outer(u_k('+'),N('+'))+outer(u_k('-'),N('-')))*dS \
        - inner(outer(v,N), grad(u_k))*ds \
        - inner(grad(v), outer(u_k,N))*ds \
        + gamma/h*inner(v,u_k)*ds \
        +inner((grad(u_k)*u_k),v)*dx- (1/2)*inner(u_k,n)*inner(u_k,v)*ds \
     -(1/2)*(inner(u_k('+'),N('+'))+inner(u_k('-'),N('-')))*avg(inner(u_k,v))*ds \
    -dot(avg(v),dot(outer(u_k('+'),N('+'))+outer(u_k('-'),N('-')),avg(u_k)))*dS

    r12 = div(v)*p_k*dx
    r21 = div(u_k)*q*dx
    RHSform = r11-r12-r21


    p11 = inner(u,v)*dx
    # p12 = div(v)*p*dx
    # p21 = div(u)*q*dx
    p22 = inner(p,q)*dx
    prec = p11 +p22
    bc = DirichletBC(W.sub(0),Expression(("0","0")), boundary)
    bcs = [bc]

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-5       # tolerance
    iter = 0            # iteration counter
    maxiter = 10        # max no of iterations allowed
    parameters = CP.ParameterSetup()
    outerit = 0


    parameters['linear_algebra_backend'] = 'uBLAS'
    BQB = assemble(inner(u,v)*dx- div(v)*p*dx-div(u)*q*dx)
    bc.apply(BQB)
    BQB = BQB.sparray()
    X = BQB[0:V.dim(),0:V.dim()]
    Xdiag = X.diagonal()
    # Xdiag = X.sum(0)[0]
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



    while eps > tol and iter < maxiter:
        iter += 1
        x = Function(W)

        uu = Function(W)
        tic()
        AA, bb = assemble_system(a, L1-RHSform, bcs)
        A,b = CP.Assemble(AA,bb)
        print toc()
        print A
        # # b = b.getSubVector(t_is)
        # PP = assemble(prec)
        # bcc.apply(PP)
        # P = CP.Assemble(PP)


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
        pc.setPythonContext(NSprecond.LSCnew(W,A,L,Bd,dBt))
        ksp.setOperators(A)
        OptDB = PETSc.Options()
        OptDB['ksp_gmres_restart'] = 200
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

#

    ue = u0
    pe = p0


    AvIt[xx-1] = np.ceil(outerit/iter)
    u = interpolate(ue,V)
    p = interpolate(pe,Q)

    ua = Function(V)
    ua.vector()[:] = u_k.vector().array()
    # nonlinear[xx-1] = assemble(inner((grad(ua)*ua),ua)*dx+(1/2)*div(ua)*inner(ua,ua)*dx- (1/2)*inner(ua,n)*inner(ua,ua)*ds)


    Nv  = u.vector().array().shape

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

    pInterp = interpolate(pe,Q)
    pe = Function(Q)
    pe.vector()[:] = pInterp.vector().array()
    const = - assemble(pe*dx)/assemble(ones*dx)
    pe.vector()[:] = pe.vector()[:]+const

    errL2u[xx-1] = errornorm(ue,ua,norm_type="L2", degree_rise=4,mesh=mesh)
    errL2p[xx-1] = errornorm(pe,pp,norm_type="L2", degree_rise=4,mesh=mesh)
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
plot(interpolate(ue,V))

plot(pp)
plot(interpolate(pe,Q))

interactive()

# plt.show()
