from dolfin import assemble, MixedFunctionSpace, tic,toc
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from  PackageName.Preconditioners import StokesApply, MaxwellApply, NSApply

def solve(A, b, SolveType = 'Direct', SolverSetup = {}):
    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)

    if SolveType == 'Direct':
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        OptDB = PETSc.Options()
        OptDB['pc_factor_mat_solver_package']  = "pastix"
        OptDB['pc_factor_mat_ordering_type']  = "rcm"
        ksp.setFromOptions()
        ksp.setTolerances(1e-10)
    else:
        ksp.setType(SolverSetup['ksp'])
        pc = ksp.getPC()
        pc.setType(SolverSetup['pc'])
        ksp.setTolerances(SolverSetup['tol'])

        if SolverSetup['eq'] == 'Stokes':
            pc.setPythonContext(StokesApply(SolverSetup['W'], SolverSetup['precondKsp']))
        elif SolverSetup['eq'] == 'Maxwell':
            pc.setPythonContext(MaxwellApply(SolverSetup['W'], SolverSetup['precondKsp']))
        elif SolverSetup['eq'] == 'NS':
            pc.setPythonContext(NSApply(SolverSetup['W'], SolverSetup['precondKsp']))

    ksp.max_it = 1000
    ksp.view()
    scale = b.norm()
    b = b/scale
    u = b.duplicate()

    ksp.setOperators(A,A)
    ksp.solve(b,u)

    u = u*scale

    iterations = {'GlobalIters': ksp.its}
    if iterations.has_key('HX'):
        iterations['HX'], iterations['CG'] = pc.getPythonContext().ReturnIters()
    return u, iterations

