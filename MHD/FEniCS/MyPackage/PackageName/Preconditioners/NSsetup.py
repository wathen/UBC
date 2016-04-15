from dolfin import *
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from PackageName import PETScFunc
from PackageName.GeneralFunc import common
from PackageName.GeneralFunc import PrintFuncs

def NSMatrixSetup(P, FS, A, nu, iter, Type):

    if Type['precond'] == 'PCD':
        u  = TrialFunction(FS['Velocity'])
        v = TestFunction(FS['Velocity'])
        p = TrialFunction(FS['Pressure'])
        q = TestFunction(FS['Pressure'])
        IS = common.IndexSet([FS['Velocity']])
        mesh = FS['Velocity'].mesh()
        n = FacetNormal(mesh)
        if iter == 1:
            if FS['Pressure'].ufl_element().degree() == 0 or FS['Pressure'].ufl_element().family() == 'DG':
                PrintFuncs.Errors('Navier-Stokes PCD preconditioner not implemented for DG pressure elements')

            P['L'] = PETScFunc.Assemble(assemble(nu*inner(grad(p),grad(q))*dx))

            P['M'] = PETScFunc.Assemble(assemble((1./nu)*inner(p,q)*dx))

        P['Fp'] = PETScFunc.Assemble(assemble(nu*inner(grad(q), grad(p))*dx(mesh)+inner((Type['u_k'][0]*grad(p)[0]+Type['u_k'][1]*grad(p)[1]),q)*dx(mesh) + (1./2)*div(Type['u_k'])*inner(p,q)*dx(mesh) - (1./2)*(Type['u_k'][0]*n[0]+Type['u_k'][1]*n[1])*inner(p,q)*ds(mesh)))

        P['F'] = A.getSubMatrix(IS[0],IS[0])

    elif Type['precond'] == 'LSC':
        u  = TrialFunction(FS['Velocity'])
        v = TestFunction(FS['Velocity'])
        IS = common.IndexSet(FS)
        if iter == 1:
            Qdiag = PETScFunc.Assemble(assemble(inner(u,v)*dx)).getDiagonal()
            Bt = A.getSubMatrix(IS['Velocity'], IS['Pressure'])
            B = A.getSubMatrix(IS['Pressure'], IS['Velocity'])
            Bt.diagonalScale(Qdiag)

            P['scaledBt'] = Bt
            P['L'] = B*Bt
        P['F'] = A.getSubMatrix(IS['Velocity'],IS['Velocity'])
    else:
        PrintFuncs.Errors('Navier-Stokes preconditioner has to be LSC or PCD')

    P['precond']  = Type['precond']

    return P

def NSKspSetup(P, iter, Type, Opt = 'Iterative'):
    if Type['precond'] == 'PCD':
        if iter == 1:
            kspL = PETSc.KSP()
            kspL.create(comm=PETSc.COMM_WORLD)
            pcL = kspL.getPC()
            if Opt == 'Iterative':
                kspL.setType('preonly')
                pcL.setType('hypre')
            else:
                kspL.setType('preonly')
                pcL.setType('lu')
            kspL.setOperators(P['L'],P['L'])
            P['kspL'] = kspL

            kspM = PETSc.KSP()
            kspM.create(comm=PETSc.COMM_WORLD)
            pcM = kspM.getPC()
            if Opt == 'Iterative':
                kspM.setType('cg')
                pcM.setType('jacobi')
            else:
                kspM.setType('preonly')
                pcM.setType('lu')
            kspM.setOperators(P['M'],P['M'])
            P['kspM'] = kspM

        kspF = PETSc.KSP()
        kspF.create(comm=PETSc.COMM_WORLD)
        pcF = kspF.getPC()
        if Opt == 'Iterative':
            kspF.setType('preonly')
            pcF.setType('hypre')
        else:
            kspF.setType('preonly')
            pcF.setType('lu')
        kspF.setOperators(P['F'],P['F'])
        P['kspF'] = kspF

    elif Type['precond'] == 'LSC':
        if iter == 1:
            kspL = PETSc.KSP()
            kspL.create(comm=PETSc.COMM_WORLD)
            pcL = kspL.getPC()
            if Opt == 'Iterative':
                kspL.setType('preonly')
                pcL.setType('hypre')
            else:
                kspL.setType('preonly')
                pcL.setType('lu')
            kspL.setOperators(P['L'],P['L'])
            P['kspL'] = kspL

        kspF = PETSc.KSP()
        kspF.create(comm=PETSc.COMM_WORLD)
        pcF = kspF.getPC()
        if Opt == 'Iterative':
            kspF.setType('preonly')
            pcF.setType('hypre')
        else:
            kspF.setType('preonly')
            pcF.setType('lu')
        kspF.setOperators(P['F'],P['F'])
        P['kspF'] = kspF

    return P




