import petsc4py
import sys

petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
from dolfin import tic, toc
import HiptmairSetup
import PETScIO as IO
import scipy.sparse as sp
# import matplotlib.pylab as plt
import MatrixOperations as MO
import HiptmairSetup

class BaseMyPC(object):
    def setup(self, pc):
        pass
    def reset(self, pc):
        pass
    def apply(self, pc, x, y):
        raise NotImplementedError
    def applyT(self, pc, x, y):
        self.apply(pc, x, y)
    def applyS(self, pc, x, y):
        self.apply(pc, x, y)
    def applySL(self, pc, x, y):
        self.applyS(pc, x, y)
    def applySR(self, pc, x, y):
        self.applyS(pc, x, y)
    def applyRich(self, pc, x, y, w, tols):
        self.apply(pc, x, y)

class Matrix(object):
    def __init__(self):
        pass
    def create(self, mat):
        pass
    def destroy(self, mat):
        pass







class ShiftedCurlCurl(BaseMyPC):

    def __init__(self, Z, M, T):
        self.T = T
        self.Z = Z
        self.M = M

    def create(self, pc):
        print "Create"



    def setUp(self, pc):
        kspT = PETSc.KSP()
        kspT.create(comm=PETSc.COMM_WORLD)
        pcT = kspT.getPC()
        kspT.setType('preonly')
        pcT.setType('hypre')
        kspT.setOperators(self.T, self.T)
        options = PETSc.Options()
        # options['pc_hypre_boomeramg_cycle_type']  = "W"
        #     #
        options["pc_hypre_type"] = "boomeramg"
        options["pc_hypre_boomeramg_nodal_coarsen"] = 6
        options["pc_hypre_boomeramg_vec_interp_variant"] = 2
        kspT.setFromOptions()
        self.kspT = kspT

        kspM = PETSc.KSP()
        kspM.create(comm=PETSc.COMM_WORLD)
        pcM = kspM.getPC()
        kspM.setType('preonly')
        pcM.setType('hypre')
        kspM.setOperators(self.M, self.M)
        self.kspM = kspM

        print "setup"
    def apply(self, pc, x, y):
        x1 = x.duplicate()
        x2 = x.duplicate()
        x3 = x.duplicate()
        self.kspT.solve(x, x1)
        self.Z.mult(x1, x2)
        self.kspM.solve(x2, x3)
        y.array = x3.array









