import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from PackageName.GeneralFunc import common
from numpy import concatenate

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

class StokesApply(BaseMyPC):

    def __init__(self, W, KSP):
        self.W = W
        self.IS = common.IndexSet(W)
        self.KSP = KSP
    def create(self, pc):
        a = 1
    def setUp(self, pc):
        a = 1
    def apply(self, pc, x, y):

        xu = x.getSubVector(self.IS[0])
        yu = xu.duplicate()
        xp = x.getSubVector(self.IS[1])
        yp = xp.duplicate()

        self.KSP[0].solve(xu, yu)
        self.KSP[1].solve(xp, yp)

        y.array = (concatenate([yu.array, yp.array]))

