import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from PackageName.GeneralFunc import common
from numpy import concatenate
import HXapply

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

class MaxwellApply(BaseMyPC):

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

        if self.KSP.has_key('kspCurl'):
            self.KSP['kspCurl'].solve(xu, yu)
            self.KSP['kspL'].solve(xp, yp)
        else:
            self.KSP['kspScalar'].solve(xp, yp)
            yu, self.HXits = HXapply.solve(self.KSP, xu)
        self.CGits = self.KSP['kspScalar'].its
        y.array = (concatenate([yu.array, yp.array]))

    def ReturnIters(self):
        return self.HXits, self.CGits

