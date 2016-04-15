import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc


def PCDKSPsetup(F, Q, A):
    # OptDB = PETSc.Options()
    # OptDB['pc_hypre_type'] = 'boomeramg'
    # OptDB['pc_hypre_boomeramg_strong_threshold']  = 0.5
    # OptDB['pc_hypre_boomeramg_grid_sweeps_all']  = 1

    kspF = PETSc.KSP()
    kspF.create(comm=PETSc.COMM_WORLD)
    pcF = kspF.getPC()
    kspF.setType('preonly')
    pcF.setType('hypre')
    kspF.setFromOptions()

    kspA = PETSc.KSP()
    kspA.create(comm=PETSc.COMM_WORLD)
    pcA = kspA.getPC()
    kspA.setType('preonly')
    pcA.setType('hypre')
    kspA.setFromOptions()

    kspQ = PETSc.KSP()
    kspQ.create(comm=PETSc.COMM_WORLD)
    pcA = kspQ.getPC()
    kspQ.setType('cg')
    pcA.setType('jacobi')
    kspQ.setTolerances(tol)
    kspQ.setFromOptions()


    kspF.setOperators(F,F)
    kspA.setOperators(A,A)
    kspQ.setOperators(Q,Q)


    return kspF, kspA, kspQ

def LSCKSPsetup(F, QB, B):
    # OptDB = PETSc.Options()
    # OptDB['pc_hypre_type'] = 'boomeramg'
    # OptDB['pc_hypre_boomeramg_strong_threshold']  = 0.5
    # OptDB['pc_hypre_boomeramg_grid_sweeps_all']  = 1

    BQB = B*QB

    kspF = PETSc.KSP()
    kspF.create(comm=PETSc.COMM_WORLD)
    pcF = kspF.getPC()
    kspF.setType('preonly')
    pcF.setType('hypre')
    kspF.setFromOptions()

    kspBQB = PETSc.KSP()
    kspBQB.create(comm=PETSc.COMM_WORLD)
    pcBQB = kspBQB.getPC()
    kspBQB.setType('preonly')
    pcBQB.setType('hypre')
    kspBQB.setFromOptions()


    kspF.setOperators(F,F)
    kspBQB.setOperators(BQB,BQB)

    return kspF, kspBQB



