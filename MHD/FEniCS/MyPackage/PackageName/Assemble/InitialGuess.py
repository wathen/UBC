from dolfin import *
from PackageName.Forms import Stokes
from PackageName.GeneralFunc import PrintFuncs
from PackageName.PETScFunc import PETScMatOps
from PackageName.Preconditioners import StokesSetup
from PackageName.Solvers import solve
from numpy.linalg import norm

def StokesIG(FS, F, U, nu, Stab = 'No'):

    W = FS['Velocity']*FS['Pressure']
    a, L = Stokes(W, F, nu, Stab)

    def boundary(x, on_boundary):
        return on_boundary

    PrintFuncs.PrintStr('Matrix assembly',4,'-','\n','\n')

    bcu = DirichletBC(W.sub(0), U['u0'], boundary)
    bc = [bcu]
    tic()
    A, b = assemble_system(a, L, bc)
    PrintFuncs.StrTimePrint('Matrix Assembly, time:',toc())

    tic()
    A, b = PETScMatOps.Assemble(A, b)
    PrintFuncs.StrTimePrint('Convert to PETSc format, time:',toc())

    tic()
    P = StokesSetup.StokesMatrixSetup(FS, A, nu)
    PrintFuncs.StrTimePrint('Precondition matrix setup, time:',toc())

    tic()
    StokesKsp = StokesSetup.StokesKspSetup(P)
    PrintFuncs.StrTimePrint('Precondition ksp setup, time:',toc())

    PrintFuncs.PrintStr('Solving system',4,'-','\n','\n')

    Type = {'eq': 'Stokes', 'ksp':'minres', 'pc': 'python', 'tol': 1e-10, 'W': W, 'precondKsp': StokesKsp}

    tic()
    u, its = solve(A, b, SolveType = 'Iterative', SolverSetup = Type)
    PrintFuncs.StrTimePrint('Solve system, time:',toc())

    x = u[0:FS['Velocity'].dim()]
    p =  u[FS['Velocity'].dim():]

    u_k = Function(FS['Velocity'])
    u_k.vector()[:] = x

    p_k = Function(FS['Pressure'])
    n = p.shape
    p_k.vector()[:] = p

    ones = Function(FS['Pressure'])
    ones.vector()[:]=(0*ones.vector().array()+1)
    p_k.vector()[:] += -assemble(p_k*dx)/assemble(ones*dx)

    return u_k, p_k

def MaxwellIG(W, F, U, params):
    return 1



