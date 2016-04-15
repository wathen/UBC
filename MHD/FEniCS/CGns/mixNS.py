#!/usr/bin/python

# from MatrixOperations import *
from dolfin import *
import numpy as np
import matplotlib.pylab as plt
import os
import scipy.io
from PyTrilinos import Epetra, EpetraExt, AztecOO, ML, Amesos

def SaveEpertaMatrix(A,name):
     from PyTrilinos import EpetraExt
     from numpy import array,loadtxt
     import scipy.sparse as sps
     import scipy.io
     test ="".join([name,".txt"])
     EpetraExt.RowMatrixToMatlabFile(test,A)
     data = loadtxt(test)
     col,row,values = data[:,0]-1,data[:,1]-1,data[:,2]
     Asparse = sps.csr_matrix((values, (row, col)))
     testmat ="".join([name,".mat"])
     scipy.io.savemat( testmat, {name: Asparse},oned_as='row')



#MO.SwapBackend('epetra')
#os.system("echo $PATH")
m = 5
errL2u = np.zeros((m-1,1))
errL2p = np.zeros((m-1,1))
NN = np.zeros((m-1,1))
DoF = np.zeros((m-1,1))
Vdim = np.zeros((m-1,1))
nn = 2

dim = 2
Solving = 'No'
Saving = 'no'
case = 1


for xx in xrange(1,m):
    print xx
    # parameters['linear_algebra_backend'] = 'Epetra'
    nn = 2**(xx+1)
    # N[xx-1] = nn
    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn
    # nn = 32
    # mesh = UnitSquareMesh(16,16)
    # mesh = UnitSquareMesh(nn, nn)
    mesh = RectangleMesh(-1, -1, 1, 1, nn, nn,'left')
    # tic()

    parameters['reorder_dofs_serial'] = False
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    parameters['reorder_dofs_serial'] = False
    # print 'time to create function spaces', toc(),'\n\n'
    W = V*Q
    Vdim[xx-1] = V.dim()
    print Vdim[xx-1]
    def boundary(x, on_boundary):
        return on_boundary

    # u0 = Expression(("sin(pi*x[1])","sin(pi*x[0])"))
    # u0 = Expression(("pow(x[1],2)-1","pow(x[0],2)-1"))
    # u0 = Expression(("1","1"))
    # u0 = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
    u0 = Expression(("x[1]*x[1]",'x[0]*x[0]'))
    # p0 = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")

    bc = DirichletBC(W.sub(0),u0, boundary)
    # bc1 = DirichletBC(W.sub(1), p0, boundary)
    bcs = [bc]
    # v, u = TestFunction(V), TrialFunction(V)
    # q, p = TestFunction(Q), TrialFunction(Q)
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    # f = -Expression(('(2*x[0]-1)*(x[1]*x[1]-x[1])','(2*x[1]-1)*(x[0]*x[0]-x[0])'))
    # f = -Expression(("400*x[0]*pow(x[1],6)+(5*pow(x[0],4)-5*pow(x[1],4))*60*x[0]*x[1]*x[1]","400*pow(x[0],4)*pow(x[1],3)-(5*pow(x[0],4)-5*pow(x[1],4))*20*x[1]*x[1]*x[1]"))
    f = -Expression(("-2 +2*x[1]*x[0]*x[0]","-2+2*x[0]*x[1]*x[1]"))

    u_k = Function(V)
    mu = Constant(1e-0)

    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    d = 0
    u_k = Function(V)
    a11 = -mu*inner(grad(v), grad(u))*dx - inner(dolfin.dot(u_k,grad(u)),v)*dx
    a12 = div(v)*p*dx
    a21 = div(u)*q*dx
    L1  = inner(v, f)*dx
    a = a11+a12+a21

    i = p*q*dx
    # AA = assemble(a11)

    U = Function(W)     # new unknown function
    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-4        # tolerance
    iter = 0            # iteration counter
    maxiter = 100        # max no of iterations allowed

    while eps > tol and iter < maxiter:
        iter += 1

        solve(a==L1,U,bcs)
        uu, pp = U.split()
        u1 = Function(V,uu)
        diff = u1.vector().array() - u_k.vector().array()
        eps = np.linalg.norm(diff, ord=np.Inf)

        print '\n\n\niter=%d: norm=%g' % (iter, eps)
        # u_k.assign(uu)   # update for next iteration
        u_k.assign(u1)


    # ue = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
    # pe = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")
    ue = Expression(("x[1]*x[1]",'x[0]*x[0]'))
    pe = Expression('0')

    erru = ue - Function(V,uu)
    errp = pe - Function(Q,pp)
    pa = Function(Q)
    pa1 = Function(Q)
    pa2 = Function(Q)
    pa1.vector()[:] = pp.vector().array()
    pa2.vector()[:] = 0*pp.array+1
    pa2.vector().array()
    pa.vector()[:] = pp.array + assemble(pa1*dx)/assemble(pa2*dx)

    errL2u[xx-1]=errornorm(ue,uu,norm_type="L2", degree_rise=4,mesh=mesh)
    errL2p[xx-1]=errornorm(pe,pp,norm_type="L2", degree_rise=4,mesh=mesh)
    print "\n\n\n",errL2u[xx-1],errL2p[xx-1],"\n\n\n"



plt.loglog(NN,errL2u)
plt.title('Error plot for CG2 elements - Velocity L2 convergence = %f' % np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1]))))
plt.xlabel('N')
plt.ylabel('L2 error')


plt.figure()

plt.loglog(NN,errL2p)
plt.title('Error plot for CG1 elements - Pressure L2 convergence = %f' % np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1]))))
plt.xlabel('N')
plt.ylabel('L2 error')

plt.show()


plot(uu)
plot(interpolate(ue,V))
plot(pp)
plot(interpolate(pe,Q))
interactive()
