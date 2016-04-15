#!/usr/bin/env python
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



#MO.SwapBackend('epetra')
#os.system("echo $PATH")
m = 6
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
SolTime = np.zeros((m-1,1))
iterations = np.zeros((m-1,1))
nn = 2

dim = 2
Solving = 'No'
Saving = 'no'
case = 1
parameters['linear_algebra_backend'] = 'PETSc'


for xx in xrange(1,m):
    print xx
    nn = 2**(xx)
    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn

    mesh = RectangleMesh(-1, -1, 1, 1, nn, nn,'right')
    # tic()

    parameters['reorder_dofs_serial'] = False
    Vlagrange = VectorFunctionSpace(mesh, "CG", 1)
    Vbubble = VectorFunctionSpace(mesh,"B",3)
    Q = FunctionSpace(mesh, "CG", 1)
    parameters['reorder_dofs_serial'] = False
    # print 'time to create function spaces', toc(),'\n\n'
    V = Vlagrange+Vbubble
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
        Su0 = Expression(("pow(x[1],2)-x[1]","pow(x[0],2)-x[0]"))
        p0 = Expression("x[1]+x[0]-1")
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


    u_k = Function(V)
    mu = Constant(1e0)
    u_k.vector()[:] = u_k.vector()[:]*0
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    d = 0


    a11 = mu*inner(grad(v), grad(u))*dx + inner((grad(u)*u_k),v)*dx+(1/2)*div(u_k)*inner(u,v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds
    a12 = div(v)*p*dx
    a21 = div(u)*q*dx
    L1  = inner(v, f)*dx
    a = a11-a12-a21

    p11 = mu*inner(grad(v), grad(u))*dx + inner((grad(u)*u_k),v)*dx+(1/2)*div(u_k)*inner(u,v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds
    p22 = inner(grad(q),grad(p))*dx

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-4       # tolerance
    iter = 0            # iteration counter
    maxiter = 100        # max no of iterations allowed

    while eps > tol and iter < maxiter:
            iter += 1
            x = Function(W)

            uu = Function(W)
            tic()
            AA, bb = assemble_system(a, L1, bcs)
            A = as_backend_type(AA).mat()
            print toc()

            tic()
            PP, Pb = assemble_system(p11+p22, L1, bcs)
            P = as_backend_type(PP).mat()
            print toc()


            b = bb.array()
            zeros = 0*b
            bb = IO.arrayToVec(b)
            x = IO.arrayToVec(zeros)

            tic()
            u_is = PETSc.IS().createGeneral(range(V.dim()))
            p_is = PETSc.IS().createGeneral(range(V.dim(),V.dim()+Q.dim()))

            ksp = PETSc.KSP().create()
            ksp.setOperators(A,P)
            ksp.setTolerances(1e-6)
            ksp.setType('fgmres')
            pc = ksp.getPC()
            pc.setFromOptions()
            pc.setType(pc.Type.FIELDSPLIT)
            fields = [ ("field1", u_is), ("field2", p_is)]
            pc.setFieldSplitIS(*fields)
            pc.setFieldSplitType(4)
            ksp.setFromOptions()
            # PETSc.Object.compose("LSC_L",P)
            # ksp.view()
            # subksp = pc.getFieldSplitSubKSP()

            # ksp1 = subksp[0]
            # ksp2 = subksp[1]

            # ksp1.setType(ksp1.Type.CG)
            # ksp1.pc.setType(ksp1.pc.Type.HYPRE)
            # ksp1.setFromOptions()
            # ksp1.view()

            # ksp2.setType(ksp2.Type.PREONLY)
            # ksp2.pc.setType(ksp2.pc.Type.LSC)
            # # ksp2.pc.setType(ksp2.pc.Type.JACOBI)
            # ksp2.setFromOptions()
            # ksp2.view()

            # print ksp1.its

            # ksp2.setType(ksp2.Type.PREONLY)
            # ksp2.pc.setType(ksp2.pc.Type.HYPRE)


            # Pscpipy = IO.matToSparse(P)
            # P = Pscpipy.tocsr()
            # ksp1.pc.setOperators(IO.arrayToMat(P[:V.dim(),:V.dim()]))
            # ksp2.pc.setOperators(IO.arrayToMat(P[V.dim():W.dim(),V.dim():W.dim()]))
            # # ksp2.view()
            # ksp2.getMonitor()
            # pc.view()


            print " time to create petsc field split preconditioner", toc(),"\n\n"

            tic()
            ksp.solve(bb, x)
            SolTime[xx-1] = toc()
            print "time to solve: ",SolTime[xx-1]
            iterations[xx-1] =  ksp.its
            print "\n\nouter iterations = ", iterations[xx-1]
            # print "Inner itations, field 1 = ", ksp1.its, " field 2 = ", ksp2.it
            uu = IO.vecToArray(x)
            uu = uu[0:Vdim[xx-1][0]]
            # time = time+toc()
            u1 = Function(V)
            u1.vector()[:] = u1.vector()[:] + uu
            diff = u1.vector().array() - u_k.vector().array()
            eps = np.linalg.norm(diff, ord=np.Inf)

            print '\n\n\niter=%d: norm=%g' % (iter, eps)
            u_k.assign(u1)

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
    nonlinear[xx-1] = assemble(inner((grad(ua)*ua),ua)*dx+(1/2)*div(ua)*inner(ua,ua)*dx- (1/2)*inner(ua,n)*inner(ua,ua)*ds)


    Nv  = u.vector().array().shape

    X = IO.vecToArray(x)
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





# scipy.io.savemat('Vdim.mat', {'VDoF':Vdim})
# scipy.io.savemat('DoF.mat', {'DoF':DoF})

# plt.loglog(NN,errL2u)
# plt.title('Error plot for CG2 elements - Velocity L2 convergence = %f' % np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1]))))
# plt.xlabel('N')
# plt.ylabel('L2 error')


# plt.figure()

# plt.loglog(NN,errL2p)
# plt.title('Error plot for CG1 elements - Pressure L2 convergence = %f' % np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1]))))
# plt.xlabel('N')
# plt.ylabel('L2 error')

# plt.show()

print nonlinear



print "Velocity Elements rate of convergence ", np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1])))
print "Pressure Elements rate of convergence ", np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1])))


import pandas as pd
tableTitles = ["Total DoF","V DoF","Q DoF","V-L2","V-order","P-L2","P-order"]
tableValues = np.concatenate((Wdim,Vdim,Qdim,errL2u,l2uorder,errL2p,l2porder),axis=1)
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

plt.show()

