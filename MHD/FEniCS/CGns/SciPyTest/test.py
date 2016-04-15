#!/usr/bin/python
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

# from MatrixOperations import *
from dolfin import *
import numpy as np
import matplotlib.pylab as plt
import os
import scipy.io
from PyTrilinos import Epetra, EpetraExt, AztecOO, ML, Amesos
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
import PETScIO as IO
import time
import common
import scipy.sparse.linalg as la

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
nn = 2

dim = 2
Solving = 'No'
Saving = 'no'
case = 1
parameters['linear_algebra_backend'] = 'uBLAS'


for xx in xrange(1,m):
    print xx
    nn = 2**(xx)
    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn

    mesh = RectangleMesh(-1, -1, 1, 1, nn, nn,'right')
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
        u0 = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
        p0 = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")
        gradu0 = Expression((("20*pow(x[1],3)","60*x[0]*pow(x[1],2)"),("20*pow(x[0],3)","-20*pow(x[1],3)")))
    elif case == 2:
        u0 = Expression(("pow(x[1],2)-x[1]","pow(x[0],2)-x[0]"))
        p0 = Expression("x[1]+x[0]-1")
        # gradu0 = Expression(("",""))
    elif case == 3:
        u0 = Expression(("cos(2*pi*x[1])*sin(2*pi*x[0]) ","-cos(2*pi*x[0])*sin(2*pi*x[1]) "))
        p0 = Expression("sin(2*pi*x[0])*sin(2*pi*x[1]) ")



    bc = DirichletBC(W.sub(0),u0, boundary)
    bcs = [bc]

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    if case == 1:
        f = Expression(("120*x[0]*x[1]*(1-mu)+ 400*x[0]*pow(x[1],6)+(5*pow(x[0],4)-5*pow(x[1],4))*60*x[0]*x[1]*x[1]","60*(pow(x[0],2)-pow(x[1],2))*(1-mu)+400*pow(x[0],4)*pow(x[1],3)-(5*pow(x[0],4)-5*pow(x[1],4))*20*x[1]*x[1]*x[1]"), mu = 1e0)
    elif case == 2:
        f = -Expression(("-1","-1"))
    elif case == 3:
        f = -Expression(("8*pi*pi*cos(2*pi*x[1])*sin(2*pi*x[0]) + 2*pi*cos(2*pi*x[0])*sin(2*pi*x[1])","2*pi*cos(2*pi*x[0])*sin(2*pi*x[1]) - 8*pi*pi*cos(2*pi*x[0])*sin(2*pi*x[1])"))



    mu = Constant(1e0)

    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    d = 0
    u_k,p_k = common.Stokes(V,Q,u0,Expression(("0","0")),[1,1,mu])
    p_k.vector()[:] = p_k.vector().array() - np.max(p_k.vector().array())
    uOld = np.concatenate((u_k.vector().array(),p_k.vector().array()), axis=0)
    r = IO.arrayToVec(uOld[:-1])

    a11 = mu*inner(grad(v), grad(u))*dx + inner((grad(u)*u_k),v)*dx+(1/2)*div(u_k)*inner(u,v)*dx - (1/2)*inner(u_k,n)*inner(u,v)*ds
    a12 = div(v)*p*dx
    a21 = div(u)*q*dx
    g = (gradu0*n)-p0*n
    L1  = inner(v, f)*dx + inner( g ,v)*ds
    a = a11-a12-a21
    r11 = mu*inner(grad(v), grad(u_k))*dx + inner((grad(u_k)*u_k),v)*dx+(1/2)*div(u_k)*inner(u_k,v)*dx- (1/2)*inner(u_k,n)*inner(u_k,v)*ds
    r12 = div(v)*p_k*dx
    r21 = div(u_k)*q*dx
    RHSform = r11-r12-r21

    bc = DirichletBC(W.sub(0),Expression(("0","0")), boundary)
    bcs = [bc]

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-10       # tolerance
    iter = 0            # iteration counter
    maxiter = 10        # max no of iterations allowed
    parameters['linear_algebra_backend'] = 'uBLAS'
    while eps > tol and iter < maxiter:
        iter += 1
        x = Function(W)

        uu = Function(W)
        tic()
        AA, bb = assemble_system(a, L1-RHSform, bcs)
        As = AA.sparray()
        AA = PETSc.Mat().createAIJ(size=As.shape,csr=(As.indptr, As.indices, As.data))
        print toc()

        b = bb.array()

        def Amatvec(v):
            x = IO.arrayToVec(v)
            B = x.duplicate()
            AA.mult(x,B)
            b = IO.vecToArray(B)
            return b

        gmresiter = 0

        # def precApply(v):
        # print 1
        tic()
        bb = IO.arrayToVec(bb.array())
        x = bb.duplicate()
        ksp = PETSc.KSP().create()
        pc = PETSc.PC().create()
        ksp.setOperators(AA)
        OptDB = PETSc.Options()
        OptDB["pc_type"] = "ilu"
        OptDB["ksp_type"] = "gmres"
        # OptDB["ksp_max_it"] = 1
        # OptDB["ksp_view"] = " "
        ksp.setFromOptions()
        # print ksp.view()
        ksp.solve(bb, x)
        X = IO.vecToArray(x)
        # del ksp,pc
        # return X
        print toc()

        A = la.LinearOperator((W.dim(),W.dim()),Amatvec,dtype='float64')

        # prec = la.LinearOperator((W.dim(),W.dim()),precApply,dtype='float64' )

        # def applyPrec(v):        # Define action of preconditioner
        #     return spsolve(-D2f[1:-1,1:-1],v)
        # P = LinearOperator((n,n),matvec=applyPrec,dtype=float)
        import pyamg.krylov as pyiter
        residuals = []
        def callback(x):
            residuals.append( np.linalg.norm(b - As*x) )
        # tic()
        # uu ,info= la.gmres(A, b,M=prec, tol=1e-09,maxiter=1000,callback=callback)
        # print toc()
        residuals1 = []
        def callback1(x):
            residuals1.append( np.linalg.norm(b - As*x) )


        tic()
        P = scipy.sparse.linalg.spilu(As)
        M_x = lambda x: P.solve(x)
        prec = la.LinearOperator((W.dim(),W.dim()), M_x,dtype='float64')
        uu = la.gmres(A, b,M=prec,tol=1e-09,maxiter=1000, callback=callback1)
        print toc()

        print ksp.its, len(residuals1)
        # print uu

        uu = IO.vecToArray(x)

        # uu = uu[:][0]
        UU = uu[0:Vdim[xx-1][0]]
        # time = time+toc()
        u1 = Function(V)
        u1.vector()[:] = u1.vector()[:] + UU

        pp = uu[Vdim[xx-1][0]:]
        # time = time+toc()
        p1 = Function(Q)
        n = pp.shape

        p1.vector()[:] = p1.vector()[:] + pp #np.insert(pp,n,0)
        diff = u1.vector().array()
        eps = np.linalg.norm(diff, ord=np.Inf)

        print '\n\n\niter=%d: norm=%g' % (iter, eps)
        print np.linalg.norm(p1.vector().array(),ord=np.inf)

        u = Function(V)
        u.vector()[:] = u1.vector().array() + u_k.vector().array()
        p = Function(Q)
        p.vector()[:] = p1.vector().array() + p_k.vector().array()
        u_k.assign(u)
        p_k.assign(p)

        uOld = np.concatenate((u_k.vector().array(),p_k.vector().array()), axis=0)
        r = IO.arrayToVec(uOld)
        del As, AA
#


    if case == 1:
        ue = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
        pe = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)+5")
    elif case == 2:
        ue = Expression(("pow(x[1],2)-x[1]","pow(x[0],2)-x[0]"))
        pe = Expression("x[1]+x[0]-1")
    elif case == 3:
        ue = Expression(("cos(2*pi*x[1])*sin(2*pi*x[0]) ","-cos(2*pi*x[0])*sin(2*pi*x[1]) "))
        pe = Expression("sin(2*pi*x[0])*sin(2*pi*x[1]) ")





    u = interpolate(ue,V)
    p = interpolate(pe,Q)

    ua = Function(V)
    ua.vector()[:] = u_k.vector().array()


    Nv  = u.vector().array().shape

    X = IO.vecToArray(r)
    x = X[0:Vdim[xx-1][0]]
    # x = x_epetra[0:Nv[0]]
    ua = Function(V)
    ua.vector()[:] = x

    pp = X[Nv[0]:]
    n = pp.shape
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
tableTitles = ["Total DoF","V DoF","Q DoF","V-L2","V-order","P-L2","P-order"]
tableValues = np.concatenate((Wdim,Vdim,Qdim,errL2u,l2uorder,errL2p,l2porder),axis=1)
df = pd.DataFrame(tableValues, columns = tableTitles)
pd.set_option('precision',3)
print df


# plot(ua)
# plot(interpolate(ue,V))

# plot(pp)
# plot(interpolate(pe,Q))

# interactive()

# plt.show()

