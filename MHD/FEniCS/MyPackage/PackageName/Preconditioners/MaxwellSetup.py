from dolfin import *
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from PackageName import PETScFunc
from PackageName.GeneralFunc import common, PrintFuncs
import os.path, inspect
import numpy as np
from scipy.sparse import csr_matrix, spdiags

def GradProlongation(FS):

    mesh = FS['Magnetic'].mesh()
    N = FS['Magnetic'].dim()
    M = FS['Multiplier'].dim()

    path = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), ".."))
    gradient_code = open(os.path.join(path, 'HXsetup.cpp'), 'r').read()
    compiled_gradient_module = compile_extension_module(code=gradient_code)

    column =  np.zeros(2*mesh.num_edges(), order="C")
    row =  np.zeros(2*mesh.num_edges(), order="C")
    data =  np.zeros(2*mesh.num_edges(), order="C")

    dataX =  np.zeros(2*mesh.num_edges(), order="C")
    dataY =  np.zeros(2*mesh.num_edges(), order="C")
    dataZ =  np.zeros(2*mesh.num_edges(), order="C")

    tic()
    c = compiled_gradient_module.GradProlong(mesh, dataX,dataY,dataZ, data, row, column)

    C = csr_matrix((data,(row,column)), shape=(N, M))
    Px = csr_matrix((dataX,(row,column)), shape=(N, M))
    Py = csr_matrix((dataY,(row,column)), shape=(N, M))
    Pz = csr_matrix((dataZ,(row,column)), shape=(N, M))
    end = toc()
    PrintFuncs.StrTimePrint("HX operators C and P created, time: ",end)

    if mesh.geometry().dim() == 2:
        P = {'Px': Px, 'Py': Py}
    else:
        P = {'Px': Px, 'Py': Py, 'Pz': Pz}
    HXoperators = {'DiscreteGrad': C, 'InterOper': P}

    return HXoperators

def Boundary(Space,BoundaryMarkers):
    key = BoundaryMarkers.keys()
    BC = np.zeros(0)
    for i in range(len(key)):
        BC = np.append(BC,int(str(key[i])))
    Boundary = np.ones(Space.dim())
    Boundary[BC.astype('int')] = 0
    BoundaryMarkers = spdiags(Boundary,0,Space.dim(),Space.dim())

    return BoundaryMarkers



def MaxwellMatrixSetup(FS, nu_m, U, opt = None):

    def boundary(x, on_boundary):
        return on_boundary
    bcb = DirichletBC(FS['Magnetic'], U['b0'], boundary)
    bcr = DirichletBC(FS['Multiplier'], U['r0'], boundary)

    u  = TrialFunction(FS['Magnetic'])
    v = TestFunction(FS['Magnetic'])
    p = TrialFunction(FS['Multiplier'])
    q = TestFunction(FS['Multiplier'])

    CurlCurlShift,b = assemble_system(inner(curl(u),curl(v))*dx + inner(u,v)*dx, inner(U['b0'],v)*dx, bcb)
    L = assemble(inner(grad(p),grad(q))*dx)
    bcr.apply(L)

    L = PETScFunc.Assemble(L)
    CurlCurlShift = PETScFunc.Assemble(CurlCurlShift)
    if opt != 'Hiptmair':
        A = {'CurlShift': CurlCurlShift, 'Laplacian': L}
        return A
    else:
        PrintFuncs.PrintStr('Auxillary space preconditioner setup',4,'-','\n\n','\n\n')
        if FS['Magnetic'].ufl_element().degree() == 1:

            HXoperators = GradProlongation(FS)

            BC = {}
            tic()
            BC['Magnetic'] = Boundary(FS['Magnetic'],bcb.get_boundary_values())
            BC['Multiplier'] = Boundary(FS['Multiplier'],bcr.get_boundary_values())
            PrintFuncs.StrTimePrint("Boundary markers assembled, time: ",toc())

            tic()
            VectorLaplacian = assemble(inner(grad(p),grad(q))*dx + inner(p,q)*dx)
            bcr.apply(VectorLaplacian)
            HXoperators['VectorLaplacian'] = PETScFunc.Scipy2PETSc(VectorLaplacian.sparray())
            HXoperators['ScalarLaplacian'] = L
            PrintFuncs.StrTimePrint("Vector Laplacian assembled, time: ",toc())

            tic()
            HXoperators['DiscreteGrad'] = PETScFunc.Scipy2PETSc(BC['Magnetic']*HXoperators['DiscreteGrad']*BC['Multiplier'])
            HXoperators['InterOper']['Px'] = PETScFunc.Scipy2PETSc(BC['Magnetic']*HXoperators['InterOper']['Px'] *BC['Multiplier'])
            HXoperators['InterOper']['Py']  = PETScFunc.Scipy2PETSc(BC['Magnetic']*HXoperators['InterOper']['Py'] *BC['Multiplier'])
            if len(HXoperators['InterOper']) == 3:
                HXoperators['InterOper']['Pz'] = PETScFunc.Scipy2PETSc(BC['Magnetic']*HXoperators['InterOper']['Pz'] *BC['Multiplier'])
            PrintFuncs.StrTimePrint("BC applied to C and P, time: ",toc())
            HXoperators['CurlShift'] = CurlCurlShift
            return HXoperators
        else:
            PrintFuncs.Error('Error: Auxillary space precondition only implemented for first order Nedelec elements')
            return


def MaxwellKspSetup(P, opt = 'Direct'):

    if P.has_key('DiscreteGrad'):

        kspVL = PETSc.KSP()
        kspVL.create(comm=PETSc.COMM_WORLD)
        pcVL = kspVL.getPC()

        if opt == 'Iterative':
            kspVL.setType('preonly')
            pcVL.setType('hypre')
        elif opt == 'Direct':
            kspVL.setType('preonly')
            pcVL.setType('lu')
        else:
            PrintFuncs.Error('Error: No Krylov solver set')

        kspVL.setOperators(P['VectorLaplacian'],P['VectorLaplacian'])

        kspSL = PETSc.KSP()
        kspSL.create(comm=PETSc.COMM_WORLD)
        pcSL = kspSL.getPC()

        if opt == 'Iterative':
            kspSL.setType('preonly')
            pcSL.setType('hypre')
        elif opt == 'Direct':
            kspSL.setType('preonly')
            pcSL.setType('lu')
        else:
            PrintFuncs.Error('Error: No Krylov solver set')

        kspSL.setOperators(P['ScalarLaplacian'],P['ScalarLaplacian'])

        kspScalar = PETSc.KSP()
        kspScalar.create(comm=PETSc.COMM_WORLD)
        pcScalar = kspScalar.getPC()

        if opt == 'Iterative':
            kspScalar.setType('cg')
            pcScalar.setType('hypre')
            kspScalar.setTolerances(P['tol'])
        elif opt == 'Direct':
            kspScalar.setType('preonly')
            pcScalar.setType('lu')
        else:
            PrintFuncs.Error('Error: No Krylov solver set')

        kspScalar.setOperators(P['ScalarLaplacian'],P['ScalarLaplacian'])

        P['kspVL'] = kspVL
        P['kspSL'] = kspSL
        P['kspScalar'] = kspScalar
        return P
    else:
        kspCurl = PETSc.KSP()
        kspCurl.create(comm=PETSc.COMM_WORLD)
        pcCurl = kspCurl.getPC()

        if opt == 'Iterative':
            kspCurl.setType('preonly')
            pcCurl.setType('hypre')
            kspCurl.setOperators(P['CurlShift'],P['CurlShift'])
        elif opt == 'Direct':
            kspCurl.setType('preonly')
            pcCurl.setType('lu')
            kspCurl.setOperators(P['CurlShift'],P['CurlShift'])
        else:
            PrintFuncs.Error('Error: No Krylov solver set')


        kspL = PETSc.KSP()
        kspL.create(comm=PETSc.COMM_WORLD)
        pcL = kspL.getPC()

        if opt == 'Iterative':
            kspL.setType('cg')
            pcL.setType('jacobi')
            kspL.setOperators(P['Laplacian'],P['Laplacian'])
        elif opt == 'Direct':
            kspL.setType('preonly')
            pcL.setType('lu')
            kspL.setOperators(P['Laplacian'],P['Laplacian'])
        else:
            PrintFuncs.Error('Error: No Krylov solver set')


        kspP = {'kspCurl': kspCurl, 'kspL': kspL}
        return kspP




