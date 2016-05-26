import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy
from dolfin import compile_extension_module, tic, toc, DirichletBC, Expression, TestFunctions, TrialFunctions, Function
from scipy.sparse import coo_matrix, spdiags
import time





def PCDKSPlinear(Q, A):
    OptDB = PETSc.Options()
    OptDB['pc_hypre_type'] = 'boomeramg'
    OptDB['pc_hypre_boomeramg_strong_threshold']  = 0.5
    OptDB['pc_hypre_boomeramg_grid_sweeps_all']  = 1
    # OptDB['pc_hypre_boomeramg_cycle_type']  = "W"
    # OptDB = PETSc.Options()
    # OptDB["pc_factor_mat_ordering_type"] = "rcm"
    # OptDB["pc_factor_mat_solver_package"] = "mumps"


    kspA = PETSc.KSP()
    kspA.create(comm=PETSc.COMM_WORLD)
    pcA = kspA.getPC()
    kspA.setType('preonly')
    kspA.max_it = 1
    pcA.setType('hypre')
    kspA.setFromOptions()

    kspQ = PETSc.KSP()
    kspQ.create(comm=PETSc.COMM_WORLD)
    pcA = kspQ.getPC()
    kspQ.setType('cg')
    pcA.setType('bjacobi')
    kspQ.setFromOptions()

    kspA.setOperators(A,A)
    kspQ.setOperators(Q,Q)

    return kspA, kspQ

def PCDKSPnonlinear(F):
    OptDB = PETSc.Options()
    OptDB['pc_hypre_type'] = 'boomeramg'
    OptDB['pc_hypre_boomeramg_strong_threshold']  = 0.5
    OptDB['pc_hypre_boomeramg_grid_sweeps_all']  = 1
    OptDB['pc_hypre_boomeramg_cycle_type']  = "W"
    kspF = PETSc.KSP()
    kspF.create(comm=PETSc.COMM_WORLD)
    pcF = kspF.getPC()
    kspF.setType('preonly')
    kspF.max_it = 1
    pcF.setType('hypre')
    kspF.setFromOptions()

    kspF.setOperators(F,F)


    return kspF




def Ksp(BQB):
    OptDB = PETSc.Options()
    OptDB['pc_hypre_type'] = 'boomeramg'
    OptDB['pc_hypre_boomeramg_strong_threshold']  = 0.5
    OptDB['pc_hypre_boomeramg_grid_sweeps_all']  = 1
    OptDB['pc_hypre_boomeramg_cycle_type']  = "W"
    # OptDB = PETSc.Options()
    # OptDB["pc_factor_mat_ordering_type"] = "rcm"
    # OptDB["pc_factor_mat_solver_package"] = "mumps"
    kspBQB = PETSc.KSP()
    kspBQB.create(comm=PETSc.COMM_WORLD)
    pcBQB = kspBQB.getPC()
    kspBQB.setType('preonly')
    pcBQB.setType('hypre')
    kspBQB.setFromOptions()


    kspBQB.setOperators(BQB,BQB)

    return kspBQB


def LSCKSPnonlinear(F):
    OptDB = PETSc.Options()
    OptDB['pc_hypre_boomeramg_cycle_type']  = "W"
    kspF = PETSc.KSP()
    kspF.create(comm=PETSc.COMM_WORLD)
    pcF = kspF.getPC()
    kspF.setType('preonly')
    pcF.setType('hypre')
    kspF.setFromOptions()

    kspF.setOperators(F,F)
    return kspF
