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

AvIters = np.zeros((m,1))

GeneralFunc.PrintStr('NS example',4,'=','\n\n','\n\n')
u0, p0, Laplacian, Advection, Grad = GeneralFunc.Fluid2D(4)
U = {'u0': u0, 'p0': p0}

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
    F = -nu*Laplacian + Advection + Grad

    Type = 'Updates'
    Stab = 'P1P1'
    u_k, p_k = Assemble.StokesIG(FS, F, U, nu)
    a, L = Forms.NS(W, F, nu, u_k)

    def boundary(x, on_boundary):
        return on_boundary

    GeneralFunc.PrintStr('Matrix assembly',4,'-','\n','\n')
    if Type != 'Update':
        bcu = DirichletBC(W.sub(0), u0, boundary)
    else:
        bcu = DirichletBC(W.sub(0), Expression(('0.0','0.0')), boundary)

    bc = [bcu]

    maxit = 7
    tol = 1e-4
    eps = 1
    iter = 0
    P = {}
    GlobalIt = 0
    while eps > tol  and iter < maxit:
        iter += 1

        GeneralFunc.PrintStr("Iter "+str(iter),4,"=","\n\n\n","\n\n")
        tic()
        A, b = assemble_system(a, L, bc)
        GeneralFunc.StrTimePrint('Matrix Assembly, time:',toc())

        tic()
        A, b = PETScFunc.Assemble(A, b)
        GeneralFunc.StrTimePrint('Convert to PETSc format, time:',toc())

        GeneralFunc.PrintStr('Solving system',4,'-','\n','\n')

        SolveType = {'precond': 'PCD', 'u_k': u_k}
        tic()
        P = Preconditioners.NSMatrixSetup(P, FS, A, nu, iter, SolveType)
        GeneralFunc.StrTimePrint('NS precond ' + SolveType['precond'] +' matrix setup, time:',toc())

        tic()
        P = Preconditioners.NSKspSetup(P, iter, SolveType)
        GeneralFunc.StrTimePrint('NS precond ' + SolveType['precond'] +' ksp setup, time:',toc())

        P['Bt'] = True
        Type = {'eq': 'NS', 'ksp':'gmres', 'pc': 'python', 'tol': 1e-8, 'W': W, 'precondKsp': P}

        tic()
        x, iters = Solvers.solve(A, b, SolveType = 'Iterative', SolverSetup = Type)
        GeneralFunc.StrTimePrint('Solve system, time:',toc())

        xOld = PETScFunc.arrayToVec(np.concatenate([u_k.vector().array(), p_k.vector().array()]))
        xNew, eps = Assemble.NLtol(x, xOld, FS, Type)

        v = PETScFunc.PETScToNLiter(xNew,FS)
        u_k.assign(v['Velocity'])
        p_k.assign(v['Pressure'])
        GeneralFunc.PrintStr(str(iters['GlobalIters']),40,"=","\n\n\n","\n\n")
        tic()
        GlobalIt += iters['GlobalIters']
    AvIters[xx] = float(GlobalIt)/iter
    tic()
    errL2u[xx], errH1u[xx], errL2p[xx] = Errors.Fluid(xNew, [V, Q], [u0, p0])
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

print AvIters
