import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from PackageName.GeneralFunc import common, PrintFuncs
from numpy import concatenate
import NSapprox

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

class NSApply(BaseMyPC):

    def __init__(self, W, KSP):
        self.W = W
        self.IS = common.IndexSet(W)
        self.KSP = KSP
    def create(self, pc):
        a = 1
    def setUp(self, pc):
        A, A = pc.getOperators()
        if self.KSP['Bt'] == True:
            self.Bt = A.getSubMatrix(self.IS[0],self.IS[1])
    def apply(self, pc, x, y):

        xu = x.getSubVector(self.IS[0])
        yu = xu.duplicate()
        xp = x.getSubVector(self.IS[1])
        yp = xp.duplicate()

        if self.KSP['precond'] == 'PCD':
            yp = NSapprox.PCD(self.KSP, xp)
        elif self.KSP['precond'] == 'LSC':
            yp = NSapprox.LSC(self.KSP, xp)

        else:
            PrinntFuncs.Error('NS preconditioner type needs to be PCD or LSC')

        if self.KSP['Bt'] == True:
            x = xu.duplicate()
            self.Bt.mult(yp,x)
        self.KSP['kspF'].solve(xu-x,yu)


        y.array = (concatenate([yu.array, yp.array]))

    def ReturnIters(self):
        return self.HXits, self.CGits

