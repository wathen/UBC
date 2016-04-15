#!/usr/bin/python


# from MatrixOperations import *
from dolfin import *
import numpy as np
import matplotlib.pylab as plt
import os
import scipy.io
from PyTrilinos import Epetra, EpetraExt, AztecOO, ML, Amesos
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix

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


def NullSpace(A,name):
     from PyTrilinos import EpetraExt, Epetra
     from numpy import array,loadtxt
     import scipy.sparse as sps
     import scipy.io
     import matplotlib.pylab as plt
     test ="".join([name,".txt"])
     EpetraExt.RowMatrixToMatlabFile(test,A)
     data = loadtxt(test)
     col,row,values = data[:,0]-1,data[:,1]-1,data[:,2]
     Asparse = sps.csr_matrix((values, (row, col)))
     (Nb,Mb) = Asparse.shape
     Aublas1 = Asparse[0:Nb-1,0:Mb-1]
     # plt.spy(Aublas1)
     # if (Nb < 1000):
        # plt.show()

     comm = Epetra.PyComm()
     Ap = scipy_csr_matrix2CrsMatrix(Aublas1, comm)
     return Ap


#MO.SwapBackend('epetra')
#os.system("echo $PATH")
m =7
errL2u = np.zeros((m-1,1))
errL2p = np.zeros((m-1,1))
NN = np.zeros((m-1,1))
DoF = np.zeros((m-1,1))
Vdim = np.zeros((m-1,1))
Qdim = np.zeros((m-1,1))
Wdim = np.zeros((m-1,1))
nn = 2

dim = 2
Solving = 'No'
Saving = 'no'
case = 1
parameters['linear_algebra_backend'] = 'Epetra'


for xx in xrange(1,m):
    print xx
    nn = 2**xx
    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn
    # nn = 32
    # mesh = UnitSquareMesh(16,16)
    # mesh = UnitSquareMesh(nn, nn)
    mesh = RectangleMesh(-1, -1, 1, 1, nn, nn,'right')
    # tic()

    parameters['reorder_dofs_serial'] = False
    V = FunctionSpace(mesh, "BDM", 1)
    Q = FunctionSpace(mesh, "DG", 0)
    parameters['reorder_dofs_serial'] = False
    # print 'time to create function spaces', toc(),'\n\n'
    W = V*Q
    Vdim[xx-1] = V.dim()
    Qdim[xx-1] = Q.dim()
    Wdim[xx-1] = W.dim()
    print "\n\n\V:  ",Vdim[xx-1],"Q:  ",Qdim[xx-1],"W:  ",Wdim[xx-1],"\n\n"
    def boundary(x, on_boundary):
        return on_boundary

    # u0 = Expression(("sin(pi*x[1])","sin(pi*x[0])"))
    # u0 = Expression(("pow(x[1],2)-1","pow(x[0],2)-1"))

    if case == 1:
        u0 = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
        p0 = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")
    elif case == 2:
        u0 = Expression(("pow(x[1],2)-x[1]","pow(x[0],2)-x[0]"))
        p0 = Expression("x[1]+x[0]-1")
    elif case == 3:
        u0 = Expression(("cos(2*pi*x[1])*sin(2*pi*x[0]) ","-cos(2*pi*x[0])*sin(2*pi*x[1]) "))
        p0 = Expression("sin(2*pi*x[0])*sin(2*pi*x[1]) ")



    bc = DirichletBC(W.sub(0),u0, boundary)
    bc1 = DirichletBC(W.sub(1), p0, boundary)
    bcs = [bc]
    # v, u = TestFunction(V), TrialFunction(V)
    # q, p = TestFunction(Q), TrialFunction(Q)
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    # f = Expression(("-pi*pi*sin(pi*x[1])+2*x[1]","-pi*pi*sin(pi*x[0])"))
    if case == 1:
        f = -Expression(("0","0"))
    elif case == 2:
        f = -Expression(("-1","-1"))
    elif case == 3:
        f = -Expression(("8*pi*pi*cos(2*pi*x[1])*sin(2*pi*x[0]) + 2*pi*cos(2*pi*x[0])*sin(2*pi*x[1])","2*pi*cos(2*pi*x[0])*sin(2*pi*x[1]) - 8*pi*pi*cos(2*pi*x[0])*sin(2*pi*x[1])"))


    u_k = Function(V)
    mu = Constant(1e-0)
    N = FacetNormal(mesh)
    t = as_vector((-N[0], N[1]))
    h = CellSize(mesh)
    h_avg =avg(h)
    alpha = 10.0
    gamma =10.0
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    d = 0
    a11 = inner(grad(v), grad(u))*dx \
        - inner(avg(grad(v)), outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
        - inner(outer(v('+'),N('+'))+outer(v('-'),N('-')), avg(grad(u)))*dS \
        + alpha/h_avg*inner(outer(v('+'),N('+'))+outer(v('-'),N('-')),outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
        - inner(outer(v,N), grad(u))*ds \
        - inner(grad(v), outer(u,N))*ds \
        + gamma/h*inner(v,u)*ds
    a12 = div(v)*p*dx
    a21 = div(u)*q*dx
    L1  =  inner(v,f)*dx - gamma/h*inner(u0,v)*ds + inner(grad(v),outer(u0,N))*ds
    a = -a11+a12+a21

    i = p*q*dx
    # AA = assemble(a11)
    x = Function(W)

    uu = Function(W)
    # tic()
    AA, bb = assemble_system(a, L1, bcs)
    PP, btmp = assemble_system(i+a11, L1, bcs)
    DoF = V.dim() + Q.dim()
    x_epetra = Epetra.Vector(0*bb.array())
    A_epetra = down_cast(AA).mat()
    P_epetra = down_cast(PP).mat()
    b_epetra = down_cast(bb).vec()
    # x_epetra = down_cast(uu.vector()).vec()
    A_epetra =NullSpace(A_epetra,"A_epetra")
    P_epetra =NullSpace(P_epetra,"P_epetra")

    bbb =bb.array()
    Nb = bbb.shape
    b =bbb[0:Nb[0]-1]
    b_epetra = Epetra.Vector(b)
    xxx = x.vector().array()
    x =xxx[0:Nb[0]-1]
    x_epetra = Epetra.Vector(x)

    # mlList = {"max levels"        : 200,
    #       "output"            : 10,
    #       "smoother: type"    : "symmetric Gauss-Seidel",
    #       "aggregation: type" : "Uncoupled"
    # }

    # prec = ML.MultiLevelPreconditioner(P_epetra, False)
    # prec.SetParameterList(mlList)
    # prec.ComputePreconditioner()

    # solver = AztecOO.AztecOO(A_epetra, x_epetra, b_epetra)
    # solver.SetPrecOperator(prec)
    # solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres);
    # solver.SetAztecOption(AztecOO.AZ_output, 100);
    # err = solver.Iterate(20000, 1e-10)


    tic()
    problem = Epetra.LinearProblem(A_epetra,x_epetra,b_epetra)
    print '\n\n\n\n\n\n'
    factory = Amesos.Factory()
    solver = factory.Create("Amesos_Umfpack", problem)
    # solver = factory.Create("MUMPS", problem)
    amesosList = {"PrintTiming" : True, "PrintStatus" : True }
    solver.SetParameters(amesosList)
    solver.SymbolicFactorization()
    solver.NumericFactorization()
    solver.Solve()
    soln = problem.GetLHS()
    print "||x_computed||_2 =", soln.Norm2()
    # solver.PrintTiming()
    print '\n\n\n\n\n\n'

    if case == 1:
        ue = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
        pe = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)+5")
    elif case == 2:
        ue = Expression(("pow(x[1],2)-x[1]","pow(x[0],2)-x[0]"))
        pe = Expression("x[1]+x[0]-1")
    elif case == 3:
        ue = Expression(("cos(2*pi*x[1])*sin(2*pi*x[0]) ","-cos(2*pi*x[0])*sin(2*pi*x[1]) "))
        pe = Expression("sin(2*pi*x[0])*sin(2*pi*x[1]) ")




    # pp = x_epetra[Vdim[xx-1][0]:]
    # pa = Function(Q)
    # pa1 = Function(Q)
    # pa2 = Function(Q)
    # pa1.vector()[:] = pp.array
    # pa2.vector()[:] = 0*pp.array+1
    # pa2.vector().array()
    # pa.vector()[:] = pp.array + assemble(pa1*dx)/assemble(pa2*dx)

    # uu = x_epetra[0:Vdim[xx-1][0]]
    # ua = Function(V)
    # ua.vector()[:] = uu.array

    u = interpolate(ue,V)
    p = interpolate(pe,Q)

    Nv  = u.vector().array().shape

    x = x_epetra[0:Nv[0]]
    ua = Function(V)
    ua.vector()[:] = x.array

    pp = x_epetra[Nv[0]:]
    pp = pp.array
    n = pp.shape
    pp = np.insert(pp,n,0)
    pa = Function(Q)
    pa.vector()[:] = pp

    pend = assemble(pa*dx)

    ones = Function(Q)
    ones.vector()[:]=(0*pp+1)
    pp = Function(Q)
    pp.vector()[:] = pa.vector().array() - assemble(pa*dx)/assemble(ones*dx)

    pInterp = interpolate(pe,Q)
    pe = Function(Q)
    pe.vector()[:] = pInterp.vector().array()
    const = - assemble(pe*dx)/assemble(ones*dx)
    pe.vector()[:] = pe.vector()[:]+const
    errL2u[xx-1] = errornorm(ue,ua,norm_type="L2", degree_rise=4,mesh=mesh)
    errL2p[xx-1] = errornorm(pe,pp,norm_type="L2", degree_rise=4,mesh=mesh)

    print "\n\n"
    print errL2u[xx-1], errL2p[xx-1]
    print "\n\n"
    del  solver





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




print "Velocity Elements rate of convergence ", np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1])))
print "Pressure Elements rate of convergence ", np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1])))

plt.loglog(NN,errL2u)
plt.title('Error plot for P2 elements - convergence = %f' % np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1]))))
plt.xlabel('N')
plt.ylabel('L2 error')

plt.figure()
plt.loglog(NN,errL2p)
plt.title('Error plot for P1 elements - convergence = %f' % np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1]))))
plt.xlabel('N')
plt.ylabel('L2 error')



# plot(ua)
# plot(interpolate(ue,V))

plot(pp)
plot(interpolate(pe,Q))

interactive()

plt.show()

