from PackageName import *
from dolfin import *
import numpy as np
import pandas as pd

m = 6


errL2b = np.zeros((m,1))
errCurlb = np.zeros((m,1))
errL2r = np.zeros((m,1))
errH1r = np.zeros((m,1))
ordL2b = np.zeros((m,1))
ordCurlb = np.zeros((m,1))
ordL2p = np.zeros((m,1))
ordH1r = np.zeros((m,1))

level = np.zeros((m,1))
Magneticdim = np.zeros((m,1))
Multiplierdim = np.zeros((m,1))

GlobalIts = np.zeros((m,1))
HXits = np.zeros((m,1))
CGits = np.zeros((m,1))


GeneralFunc.PrintStr('Maxwell example',4,'=','\n\n','\n\n')
b0, r0, CurlCurl, gradPres = GeneralFunc.Mag2D(1)

for xx in range(m):
    u = {'b0': b0, 'r0': r0}
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
    V = FunctionSpace(mesh, 'N1curl', 1)
    Q = FunctionSpace(mesh, 'CG', 1)
    FS = {'Magnetic': V, 'Multiplier': Q}
    W = V*Q
    GeneralFunc.StrTimePrint('Function Space assembly, time:',toc())

    GeneralFunc.PrintDOF(FS)

    Magneticdim[xx] = V.dim()
    Multiplierdim[xx] = Q.dim()

    GeneralFunc.PrintStr('Matrix assembly',4,'-','\n','\n')

    nu_m = 1.0
    F = nu_m*CurlCurl + gradPres
    a, L = Forms.Maxwell(FS, F, nu_m)

    tic()
    b, markers = Assemble.RHSAssemble(FS, L, u, 'Maxwell')
    GeneralFunc.StrTimePrint('RHS assemble, time:',toc())

    tic()
    A = Assemble.MagAssemble(FS, a, L, u, markers)
    GeneralFunc.StrTimePrint('Matrix assembly, time:',toc())

    tic()
    A, b = Assemble.PETScAssemble(FS, A, b, opt = 'Matrix')
    GeneralFunc.StrTimePrint('Convert to PETSc format, time:',toc())

    tic()
    P = Preconditioners.MaxwellMatrixSetup(FS, nu_m, u, opt = 'Hiptmair')
    GeneralFunc.StrTimePrint('Precondition matrix setup, time:',toc())
    P['tol'] = 1e-5
    tic()
    MaxwellKsp = Preconditioners.MaxwellKspSetup(P, 'Iterative')
    GeneralFunc.StrTimePrint('Precondition ksp setup, time:',toc())


    Type = {'eq': 'Maxwell', 'ksp':'minres', 'pc': 'python', 'tol': 1e-6, 'W': W, 'precondKsp': MaxwellKsp}

    GeneralFunc.PrintStr('Solving system',4,'-','\n','\n')

    tic()
    u, its = Solvers.solve(A, b, SolveType = 'Iterative', SolverSetup = Type)
    GeneralFunc.StrTimePrint('Solve system, time:',toc())

    GlobalIts[xx] = its['GlobalIters']
    if its.has_key('HX'):
        HXits[xx] = its['HX']
        CGits[xx] = its['CG']
    tic()
    errL2b[xx], errCurlb[xx], errL2r[xx], errH1r[xx] = Errors.Magnetic(u, [V, Q], [b0, r0])
    GeneralFunc.StrTimePrint('Error calculations, time:',toc())
    print '\n\n\n'

ordL2b[1:] = np.abs(np.log2(errL2b[1:]/errL2b[:-1]))
ordCurlb[1:] = np.abs(np.log2(errCurlb[1:]/errCurlb[:-1]))
ordL2p[1:] = np.abs(np.log2(errL2r[1:]/errL2r[:-1]))
ordH1r[1:] = np.abs(np.log2(errH1r[1:]/errH1r[:-1]))


print "\n\n   Magnetic convergence"
MagneticTitles = ["l","B DoF","R DoF","B-L2","L2-order","B-Curl","HCurl-order"]
MagneticValues = np.concatenate((level,Magneticdim,Multiplierdim,errL2b,ordL2b,errCurlb,ordCurlb),axis=1)
MagneticTable= pd.DataFrame(MagneticValues, columns = MagneticTitles)
pd.set_option('precision',3)
MagneticTable = GeneralFunc.PandasFormat(MagneticTable,"B-Curl","%2.4e")
MagneticTable = GeneralFunc.PandasFormat(MagneticTable,'B-L2',"%2.4e")
MagneticTable = GeneralFunc.PandasFormat(MagneticTable,"L2-order","%1.2f")
MagneticTable = GeneralFunc.PandasFormat(MagneticTable,'HCurl-order',"%1.2f")
print MagneticTable

print "\n\n   Lagrange convergence"
LagrangeTitles = ["l","B DoF","R DoF","R-L2","L2-order","R-H1","H1-order"]
LagrangeValues = np.concatenate((level,Magneticdim,Multiplierdim,errL2r,ordL2p,errH1r,ordH1r),axis=1)
LagrangeTable= pd.DataFrame(LagrangeValues, columns = LagrangeTitles)
pd.set_option('precision',3)
LagrangeTable = GeneralFunc.PandasFormat(LagrangeTable,"R-L2","%2.4e")
LagrangeTable = GeneralFunc.PandasFormat(LagrangeTable,'R-H1',"%2.4e")
LagrangeTable = GeneralFunc.PandasFormat(LagrangeTable,"H1-order","%1.2f")
LagrangeTable = GeneralFunc.PandasFormat(LagrangeTable,'L2-order',"%1.2f")
print LagrangeTable

print "\n\n   Lagrange convergence"
LagrangeTitles = ["l","DoF","Global iterations","HX iterations","CG iterations"]
LagrangeValues = np.concatenate((level,Magneticdim + Multiplierdim,GlobalIts,HXits,CGits),axis=1)
LagrangeTable= pd.DataFrame(LagrangeValues, columns = LagrangeTitles)
pd.set_option('precision',3)
print LagrangeTable

