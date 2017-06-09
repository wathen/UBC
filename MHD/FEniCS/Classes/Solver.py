from dolfin import *
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import CheckPetsc4py as CP
import StokesPrecond
import NSpreconditioner
import MaxwellPrecond as MP
import MatrixOperations as MO
import PETScIO as IO
import numpy as np
import P as PrecondMulti
import MHDprec
import scipy.sparse as sp
from scipy.linalg import svd
# import matplotlib.pylab as plt
from scipy.sparse.linalg.dsolve import spsolve


def solve(A, b, u, params, Fspace, SolveType, IterType, OuterTol, InnerTol, HiptmairMatrices, Hiptmairtol, KSPlinearfluids, Fp, kspF):

    if SolveType == "Direct":
        ksp = PETSc.KSP()
        ksp.create(comm=PETSc.COMM_WORLD)
        pc = ksp.getPC()
        ksp.setType('preonly')
        pc.setType('lu')
        OptDB = PETSc.Options()
        OptDB['pc_factor_mat_solver_package'] = "pastix"
        OptDB['pc_factor_mat_ordering_type'] = "rcm"
        ksp.setFromOptions()
        scale = b.norm()
        b = b/scale
        ksp.setOperators(A, A)
        del A
        ksp.solve(b, u)
        # Mits +=dodim
        u = u*scale
        MO.PrintStr("Number iterations = "+str(ksp.its),
                    60, "+", "\n\n", "\n\n")
        return u, ksp.its, 0
    elif SolveType == "Direct-class":
        ksp = PETSc.KSP()
        ksp.create(comm=PETSc.COMM_WORLD)
        pc = ksp.getPC()
        ksp.setType('gmres')
        pc.setType('none')
        ksp.setFromOptions()
        scale = b.norm()
        b = b/scale
        ksp.setOperators(A, A)
        del A
        ksp.solve(b, u)
        # Mits +=dodim
        u = u*scale
        MO.PrintStr("Number iterations = "+str(ksp.its),
                    60, "+", "\n\n", "\n\n")
        return u, ksp.its, 0

    else:

        # u = b.duplicate()
        if IterType == "Full":
            ksp = PETSc.KSP()
            ksp.create(comm=PETSc.COMM_WORLD)
            pc = ksp.getPC()
            ksp.setType('gmres')
            pc.setType('python')

            OptDB = PETSc.Options()
            OptDB['ksp_gmres_restart'] = 100
            # FSpace = [Velocity,Magnetic,Pressure,Lagrange]
            reshist = {}
            def monitor(ksp, its, fgnorm):
                reshist[its] = fgnorm
                print its, "    OUTER:", fgnorm
            # ksp.setMonitor(monitor)
            ksp.max_it = 100
            ksp.setTolerances(OuterTol)

            W = Fspace
            FFSS = [W.sub(0), W.sub(1), W.sub(2), W.sub(3)]

            pc.setPythonContext(MHDprec.ApproxInv(FFSS, kspF, KSPlinearfluids[0], KSPlinearfluids[1], Fp, HiptmairMatrices[
                                3], HiptmairMatrices[4], HiptmairMatrices[2], HiptmairMatrices[0], HiptmairMatrices[1], HiptmairMatrices[6], Hiptmairtol))

            # OptDB = PETSc.Options()
            # OptDB['pc_factor_mat_solver_package']  = "umfpack"
            # OptDB['pc_factor_mat_ordering_type']  = "rcm"
            ksp.setFromOptions()
            scale = b.norm()
            b = b/scale
            ksp.setOperators(A, A)
            del A
            ksp.solve(b, u)
            # Mits +=dodim
            u = u*scale
            MO.PrintStr("Number iterations = "+str(ksp.its),
                        60, "+", "\n\n", "\n\n")
            return u, ksp.its, 0

        IS = MO.IndexSet(Fspace, '2by2')
        M_is = IS[1]
        NS_is = IS[0]
        kspNS = PETSc.KSP().create()
        kspM = PETSc.KSP().create()
        kspNS.setTolerances(OuterTol)

        kspNS.setOperators(A[0])
        kspM.setOperators(A[1])
        # print P.symmetric
        if IterType == "MD":
            kspNS.setType('gmres')
            kspNS.max_it = 500

            pcNS = kspNS.getPC()
            pcNS.setType(PETSc.PC.Type.PYTHON)
            pcNS.setPythonContext(NSpreconditioner.NSPCD(MixedFunctionSpace(
                [Fspace.sub(0), Fspace.sub(1)]), kspF, KSPlinearfluids[0], KSPlinearfluids[1], Fp))
        elif IterType == "CD":
            kspNS.setType('minres')
            pcNS = kspNS.getPC()
            pcNS.setType(PETSc.PC.Type.PYTHON)
            Q = KSPlinearfluids[1].getOperators()[0]
            Q = 1./params[2]*Q
            KSPlinearfluids[1].setOperators(Q, Q)
            pcNS.setPythonContext(StokesPrecond.MHDApprox(MixedFunctionSpace(
                [Fspace.sub(0), Fspace.sub(1)]), kspF, KSPlinearfluids[1]))
        reshist = {}
        def monitor(ksp, its, fgnorm):
            reshist[its] = fgnorm
            print fgnorm
        # kspNS.setMonitor(monitor)

        uNS = u.getSubVector(NS_is)
        bNS = b.getSubVector(NS_is)
        # print kspNS.view()
        scale = bNS.norm()
        bNS = bNS/scale
        print bNS.norm()
        kspNS.solve(bNS, uNS)
        uNS = uNS*scale
        NSits = kspNS.its
        kspNS.destroy()
        # for line in reshist.values():
        #     print line
        kspM.setFromOptions()
        kspM.setType(kspM.Type.MINRES)
        kspM.setTolerances(InnerTol)
        pcM = kspM.getPC()
        pcM.setType(PETSc.PC.Type.PYTHON)
        pcM.setPythonContext(MP.Hiptmair(MixedFunctionSpace([Fspace.sub(2), Fspace.sub(3)]), HiptmairMatrices[3], HiptmairMatrices[
                             4], HiptmairMatrices[2], HiptmairMatrices[0], HiptmairMatrices[1], HiptmairMatrices[6], Hiptmairtol))

        uM = u.getSubVector(M_is)
        bM = b.getSubVector(M_is)
        scale = bM.norm()
        bM = bM/scale
        print bM.norm()
        kspM.solve(bM, uM)
        uM = uM*scale
        Mits = kspM.its
        kspM.destroy()
        u = IO.arrayToVec(np.concatenate([uNS.array, uM.array]))

        MO.PrintStr("Number of M iterations = " +
                    str(Mits), 60, "+", "\n\n", "\n\n")
        MO.PrintStr("Number of NS/S iterations = " +
                    str(NSits), 60, "+", "\n\n", "\n\n")
        return u, NSits, Mits
