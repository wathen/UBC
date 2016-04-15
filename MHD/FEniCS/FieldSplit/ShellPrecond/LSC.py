import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

class OC(object):

    def __init__(self, W):
        self.W = W
        self.u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
        self.p_is = PETSc.IS().createGeneral(range(W.sub(0).dim(),W.sub(0).dim()+W.sub(1).dim()))

    def create(self, pc):
        self.diag = None
        kspAMG = PETSc.KSP()
        kspAMG.create(comm=PETSc.COMM_WORLD)
        pc = kspAMG.getPC()
        kspAMG.setType('preonly')
        pc.setType('hypre')
        OptDB = PETSc.Options()
        OptDB["pc_hypre_boomeramg_grid_sweeps_down"] = 2
        OptDB["pc_hypre_boomeramg_grid_sweeps_up"] = 2
        OptDB["pc_hypre_boomeramg_grid_sweeps_coarse"] = 2
        kspAMG.setFromOptions()
        self.kspAMG = kspAMG

        kspCG = PETSc.KSP()
        kspCG.create(comm=PETSc.COMM_WORLD)
        pc = kspCG.getPC()
        kspCG.setType('cg')
        pc.setType('icc')
        kspCG.setFromOptions()
        self.kspCG = kspCG


    def setUp(self, pc):


        A, P, flag = ksp.getOperators()
        self.P11 = P.getSubMatrix(self.u_is,self.u_is)
        self.P22 = P.getSubMatrix(self.p_is,self.p_is)

        self.kspAMG.setOperators(self.P11,self.P11 )
        self.kspCG.setOperators(self.P22,self.P22)


    def apply(self, pc, x, y):
        # LOG('PCapply.apply()')
        # self.kspCG.setOperators(self.B)
        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()

        self.kspAMG.solve(x1, y1)
        self.kspCG.solve(x2, y2)

        y.array = np.concatenate([y1.array, y2.array])

