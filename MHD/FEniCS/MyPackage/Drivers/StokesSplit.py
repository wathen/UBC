from PackageName import *
from dolfin import *
import numpy as np
import pandas as pd

m = 5
errL2u = np.zeros((m,1))
errH1u = np.zeros((m,1))
errL2p = np.zeros((m,1))
ordL2u = np.zeros((m,1))
ordH1u = np.zeros((m,1))
ordL2p = np.zeros((m,1))

level = np.zeros((m,1))
Velocitydim = np.zeros((m,1))
Pressuredim = np.zeros((m,1))

GeneralFunc.PrintStr('Stokes example',4,'=','\n\n','\n\n')
u0, p0, Laplacian, Advection, Grad = GeneralFunc.Fluid2D(4)
u = {'u0': u0, 'p0': p0}

for xx in range(m):
    print '\n\n\n'

    level[xx] = xx + 1
    GeneralFunc.PrintStr('Level '+str(int(level[xx][0])),4,'+','\n','\n')

    n = int(2**(level[xx]))

    GeneralFunc.PrintStr('Mesh and function space assembly',4,'-','\n','\n')

    tic()
    mesh = UnitSquareMesh(n, n)
    GeneralFunc.StrTimePrint('Mesh generated, time:',toc())

    tic()
    parameters['reorder_dofs_serial'] = False
    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)

    FS = {'Velocity': V, 'Pressure': Q}
    W = V*Q
    GeneralFunc.StrTimePrint('Function Space assembly, time:',toc())

    GeneralFunc.PrintDOF(FS)

    Velocitydim[xx] = V.dim()
    Pressuredim[xx] = Q.dim()

    nu = 1.0
    F = -nu*Laplacian + Grad
    a, L = Forms.Stokes(W, F, nu)

    def boundary(x, on_boundary):
        return on_boundary

    GeneralFunc.PrintStr('Matrix assembly',4,'-','\n','\n')

    bcu = DirichletBC(W.sub(0), u0, boundary)
    bc = [bcu]

    tic()
    b, BC = Assemble.RHSAssemble(W, L, u, 'Fluid')
    GeneralFunc.StrTimePrint('RHS assemble, time:',toc())

    tic()
    A = Assemble.FluidAssemble(W, a, L, u, BC, 'Stokes')
    GeneralFunc.StrTimePrint('Matrix assembly, time:',toc())

    tic()
    A, b = Assemble.PETScAssemble(W, A, b, opt = 'Matrix')
    GeneralFunc.StrTimePrint('Convert to PETSc format, time:',toc())

    tic()
    P = Preconditioners.StokesMatrixSetup(FS, A, nu)
    GeneralFunc.StrTimePrint('Precondition matrix setup, time:',toc())

    tic()
    StokesKsp = Preconditioners.StokesKspSetup(P)
    GeneralFunc.StrTimePrint('Precondition ksp setup, time:',toc())

    GeneralFunc.PrintStr('Solving system',4,'-','\n','\n')

    Type = {'eq': 'Stokes', 'ksp':'minres', 'pc': 'python', 'tol': 1e-8, 'W': W, 'precondKsp': StokesKsp}

    tic()
    x = Solvers.solve(A, b, SolveType = 'Iterative', SolverSetup = Type)
    GeneralFunc.StrTimePrint('Solve system, time:',toc())

    tic()
    errL2u[xx], errH1u[xx], errL2p[xx] = Errors.Fluid(x, [V, Q], [u0, p0])
    GeneralFunc.StrTimePrint('Error calculations, time:',toc())
    print '\n\n\n'

ordL2u[1:] = np.abs(np.log2(errL2u[1:]/errL2u[:-1]))
ordH1u[1:] = np.abs(np.log2(errH1u[1:]/errH1u[:-1]))
ordL2p[1:] = np.abs(np.log2(errL2p[1:]/errL2p[:-1]))



LatexTitles = ["l","DoFu","Dofp","V-L2","L2-order","V-H1","H1-order","P-L2","PL2-order"]
LatexValues = np.concatenate((level,Velocitydim,Pressuredim,errL2u,ordL2u,errH1u,ordH1u,errL2p,ordL2p), axis=1)
LatexTable = pd.DataFrame(LatexValues, columns = LatexTitles)
pd.set_option('precision',3)
LatexTable = GeneralFunc.PandasFormat(LatexTable,"V-L2","%2.4e")
LatexTable = GeneralFunc.PandasFormat(LatexTable,'V-H1',"%2.4e")
LatexTable = GeneralFunc.PandasFormat(LatexTable,"H1-order","%1.2f")
LatexTable = GeneralFunc.PandasFormat(LatexTable,'L2-order',"%1.2f")
LatexTable = GeneralFunc.PandasFormat(LatexTable,"P-L2","%2.4e")
LatexTable = GeneralFunc.PandasFormat(LatexTable,'PL2-order',"%1.2f")
print LatexTable


