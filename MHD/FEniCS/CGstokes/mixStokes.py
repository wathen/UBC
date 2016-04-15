#!/usr/bin/python


# from MatrixOperations import *
from dolfin import *
import numpy as np
import matplotlib.pylab as plt
import os
import scipy.io
from PyTrilinos import Epetra, EpetraExt, AztecOO, ML
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
def SaveEpertaMatrix(A,name,xdim,ydim):
     from PyTrilinos import EpetraExt
     from numpy import array,loadtxt
     import scipy.sparse as sps
     import scipy.io
     test ="".join([name,".txt"])
     EpetraExt.RowMatrixToMatlabFile(test,A)
     data = loadtxt(test)
     col,row,values = data[:,0]-1,data[:,1]-1,data[:,2]
     Asparse = sps.csr_matrix((values, (row, col)))
     As = Asparse[0:xdim,0:ydim]
     comm = Epetra.PyComm()
     Ap = scipy_csr_matrix2CrsMatrix(Aublas1, comm)

     return Ap



def NullSpace(Arank):
    from PyTrilinos import Epetra, EpetraExt
    from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
    from scipy.sparse import csr_matrix

    Aublas = Arank.sparray()
    # Aublas = sps.csr_matrix((values, (row, col)))
    scipy.io.savemat( "before.mat", {"Aublas": Aublas},oned_as='row')
    (Nb,Mb) = Aublas.shape
    Aublas1 = Aublas[0:Nb-1,0:Mb-1]
    scipy.io.savemat( "after.mat", {"Aublas1": Aublas1},oned_as='row')

    comm = Epetra.PyComm()
    tic()
    Ap = scipy_csr_matrix2CrsMatrix(Aublas1, comm)
    print toc()

    return Ap


#MO.SwapBackend('epetra')
#os.system("echo $PATH")
m = 2
erru = np.zeros((m-1,1))
errp = np.zeros((m-1,1))
N = np.zeros((m-1,1))
DoF = np.zeros((m-1,1))
Vdim = np.zeros((m-1,1))
nn = 2

dim = 2
Solving = 'No'
Saving = 'no'
case = 1

# if Saving == 'yes':
#     parameters['linear_algebra_backend'] = 'Epetra'
# else:
parameters['linear_algebra_backend'] = 'Epetra'
# parameters['linear_algebra_backend'] = 'uBLAS'
for xx in xrange(1,m):
    print xx
    # parameters['linear_algebra_backend'] = 'uBLAS'
    nn = 2**2

    N[xx-1] = nn
    # Create mesh and define function space
    nn = int(nn)
    N[xx-1] = nn
    # nn = 32
    # mesh = UnitSquareMesh(16,16)
    mesh = UnitSquareMesh(nn, nn)
    # mesh = RectangleMesh(-1, -1, 1, 1, nn, nn,'crossed')
    # tic()

    parameters['reorder_dofs_serial'] = False
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    # parameters['reorder_dofs_serial'] = False
    # print 'time to create function spaces', toc(),'\n\n'
    W = V*Q
    Vdim[xx-1] = V.dim()
    print Vdim[xx-1]
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

    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    d = 0
    a11 = -inner(grad(v), grad(u))*dx
    a12 = div(v)*p*dx
    a21 = div(u)*q*dx
    L1  = inner(v, f)*dx
    a = a11+a12+a21
    A = assemble(a11)
    B = assemble(a12)
    Bt = assemble(a21)

    i = p*q*dx
    # AA = assemble(a11)
    SaveEpertaMatrix(A.down_cast().mat(),"A")
    SaveEpertaMatrix(B.down_cast().mat(),"B")
    SaveEpertaMatrix(Bt.down_cast().mat(),"Bt")
    uu = Function(W)
    # tic()
    AA, bb = assemble_system(a, L1, bcs)
    PP, btmp = assemble_system(i+a11, L1, bcs)
    SaveEpertaMatrix(AA.down_cast().mat(),"KKT")
    # print 'time to create linear system', toc(),'\n\n'
    # tic()
    # Aublas = AA.sparray()
    # (Nb,Mb) = Aublas.shape
    # Aublas1 = Aublas[0:Nb-1,0:Mb-1]
    # comm = Epetra.PyComm()
    # print Nb, toc()
    # tic()
    # A = scipy_csr_matrix2CrsMatrix(Aublas1, comm)
    # print toc()
    x = Function(W)
    # parameters['linear_algebra_backend'] = 'Epetra'

    # AAe, bbe = assemble_system(a, L1, bcs)
    # PP, btmp = assemble_system(i+a11, L1, bcs)

    # SaveEpertaMatrix(AA,"before")
    A_epetra = NullSpace(AA)
    P_epetra = NullSpace(PP)
    # SaveEpertaMatrix(AA,"after")


    bbb =bb.array()
    Nb = bbb.shape
    b =bbb[0:Nb[0]-1]
    b_epetra = Epetra.Vector(b)
    xxx = x.vector().array()
    x =xxx[0:Nb[0]-1]
    x_epetra = Epetra.Vector(x)


    Aname ="".join(["A",str(Nb[0]-1)])
    Pname = "".join(["P",str(Nb[0]-1)])
    # SaveEpertaMatrix(A_epetra,Aname)
    # SaveEpertaMatrix(P_epetra,Pname)


    # b_epetra = NullSpace(bb,'vec')
    # x_epetra = NullSpace(x.vector(),'vec')
    print '\n\n\n DoF = ', Nb[0],'\n\n\n'
    DoF[xx-1] = Nb[0]-1
    mlList = {"max levels"        : 200,
          "output"            : 10,
          "smoother: type"    : "symmetric Gauss-Seidel",
          "aggregation: type" : "Uncoupled"
    }

    prec = ML.MultiLevelPreconditioner(P_epetra, False)
    prec.SetParameterList(mlList)
    prec.ComputePreconditioner()

    solver = AztecOO.AztecOO(A_epetra, x_epetra, b_epetra)
    solver.SetPrecOperator(prec)
    solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres);
    solver.SetAztecOption(AztecOO.AZ_output, 100);
    err = solver.Iterate(1550, 1e-5)


    print 'done'

    # Ap = sp.save(A)

    # SaveEpertaMatrix(A,"A")
    # SaveEpertaMatrix(P.down_cast().mat(),"P")
    # tic()


    # solve(a==L1,uu,bcs)
    # # solver = KrylovSolver("gmres", "amg")
    # # solver.set_operators(AA, P)
    # # solver.solve(uu.vector(), bb)
    # # print 'time to solve linear system', toc(),'\n\n'



    # # ue = Expression(("sin(pi*x[1])","sin(pi*x[0])"))
    # # pe = Expression("x[1]*x[1]*2")


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
    pa = Function(Q)
    pa.vector()[:] = pp
    pend = assemble(pa*dx)

    mesh_cells = mesh.cells()
    meshcoords = mesh.coordinates()
    endc = meshcoords[mesh_cells[-1]]
    A = endc[0]
    B = endc[1]
    C   = endc[2]
    # print A,B,C
    areac = np.abs((A[0]*(B[1]-C[1])+B[0]*(C[1]-A[1])+C[0]*(A[1]-B[1]))/2)
    # pa.vector()[n[0]] = -pend*areac
    # x = x_epetra[0:Nv[0]]
    # ua = Function(V)
    # ua.vector()[:] = x.array

    # ue = Expression(("pow(x[0],2)*x[1]","-pow(x[1],2)*x[0]"))
    # pe = Expression("x[0]+x[1]-1.0")



    # erru = ue - ua
    # errp = pe - pa?

    erru[xx-1] = errornorm(ue,ua,norm_type="L2", degree_rise=3,mesh=mesh)
    errp[xx-1] = errornorm(pe,Function(W.sub(1),pa),norm_type="L2", degree_rise=3,mesh=mesh)


    print erru[xx-1]
    print errp[xx-1]




# scipy.io.savemat('Vdim.mat', {'VDoF':Vdim})
# scipy.io.savemat('DoF.mat', {'DoF':DoF})




# plt.loglog(N,erru)
# plt.title('Error plot for P2 elements - convergence = %f' % np.log2(np.average((erru[0:m-2]/erru[1:m-1]))))
# plt.xlabel('N')
# plt.ylabel('L2 error')

# plt.figure()
# plt.loglog(N,errp)
# plt.title('Error plot for P1 elements - convergence = %f' % np.log2(np.average((errp[0:m-2]/errp[1:m-1]))))
# plt.xlabel('N')
# plt.ylabel('L2 error')



# plot(ua)
# #plot(interpolate(ue,V))

# plot(pa)
# #plot(interpolate(pe,Q))

# interactive()

# plt.show()

