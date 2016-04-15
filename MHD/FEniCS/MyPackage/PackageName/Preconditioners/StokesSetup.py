from dolfin import *
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from PackageName import PETScFunc
from PackageName.GeneralFunc import common

def StokesMatrixSetup(FS, A, nu):


    u  = TrialFunction(FS['Velocity'])
    v = TestFunction(FS['Velocity'])
    p = TrialFunction(FS['Pressure'])
    q = TestFunction(FS['Pressure'])
    IS = common.IndexSet([FS['Velocity']])
    Mass = (1./nu)*inner(p,q)*dx

    L = A.getSubMatrix(IS[0],IS[0])
    M = assemble(Mass)
    M = PETScFunc.Assemble(M)

    return [L,M]

def StokesKspSetup(P, Opt = 'Iterative'):

    kspL = PETSc.KSP()
    kspL.create(comm=PETSc.COMM_WORLD)
    pcL = kspL.getPC()
    if Opt == 'Iterative':
        kspL.setType('preonly')
        pcL.setType('hypre')
    else:
        kspL.setType('preonly')
        pcL.setType('lu')
    kspL.setOperators(P[0],P[0])

    kspM = PETSc.KSP()
    kspM.create(comm=PETSc.COMM_WORLD)
    pcM = kspM.getPC()
    if Opt == 'Iterative':
        kspM.setType('cg')
        pcM.setType('jacobi')
    else:
        kspM.setType('preonly')
        pcM.setType('lu')
    kspM.setOperators(P[1],P[1])

    return [kspL,kspM]




