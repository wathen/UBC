import scipy.sparse as sp

import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import CheckPetsc4py as CP
import MatrixOperations as MO

def Create(A, W, PCD, KSPL):
    IS = MO.IndexSet(W)
    F = A.getSubMatrix(IS[0],IS[0])
    Ct = A.getSubMatrix(IS[0],IS[1])
    Dt = A.getSubMatrix(IS[1],IS[3])
    M = A.getSubMatrix(IS[1],IS[1])
    Bt = A.getSubMatrix(IS[0],IS[2])
    L = KSPL.getOperators()
    F = CP.PETSc2Scipy(F)
    Ct = CP.PETSc2Scipy(Ct)
    Dt = CP.PETSc2Scipy(Dt)
    Bt = CP.PETSc2Scipy(Bt)
    L = CP.PETSc2Scipy(L[0])
    M = CP.PETSc2Scipy(M)
    Ap = CP.PETSc2Scipy(PCD[0])
    Mp = CP.PETSc2Scipy(PCD[1])
    Fp = CP.PETSc2Scipy(PCD[2])
    F = CP.PETSc2Scipy(F)
    F = CP.PETSc2Scipy(F)

