import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

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

class Apply(BaseMyPC):

    def __init__(self, A):
            self.A = A

    def apply(self, pc, x, y):
        ysave = x.duplicate()

        xhat = self.A['InterOper']['Px'].getVecRight()
        self.A['InterOper']['Px'].multTranspose(x,xhat)
        yp1 = self.A['InterOper']['Px'].getVecLeft()
        yhat = self.A['InterOper']['Px'].getVecRight()
        self.A['kspVL'].solve(xhat, yhat)
        self.A['InterOper']['Px'].mult(yhat,yp1)
        xhat.destroy()
        yhat.destroy()

        xhat = self.A['InterOper']['Py'].getVecRight()
        self.A['InterOper']['Py'].multTranspose(x,xhat)
        yp2 = self.A['InterOper']['Py'].getVecLeft()
        yhat = self.A['InterOper']['Py'].getVecRight()
        self.A['kspVL'].solve(xhat, yhat)
        self.A['InterOper']['Py'].mult(yhat,yp2)
        xhat.destroy()
        yhat.destroy()

        if len(self.A['InterOper']) == 3:
            xhat = self.A['InterOper']['Pz'].getVecRight()
            self.A['InterOper']['Pz'].multTranspose(x,xhat)
            yp3 = self.A['InterOper']['Pz'].getVecLeft()
            yhat = self.A['InterOper']['Pz'].getVecRight()
            self.A['kspVL'].solve(xhat, yhat)
            self.A['InterOper']['Pz'].mult(yhat,yp3)


        xhat = self.A['DiscreteGrad'].getVecRight()
        self.A['DiscreteGrad'].multTranspose(x,xhat)
        yg = self.A['DiscreteGrad'].getVecLeft()
        yhat = self.A['DiscreteGrad'].getVecRight()
        self.A['kspSL'].solve(xhat, yhat)
        self.A['DiscreteGrad'].mult(yhat,yg)
        xhat.destroy()
        yhat.destroy()

        diag = self.A['CurlShift'].getDiagonal()
        diag.reciprocal()
        xx = x.duplicate()
        xx.pointwiseMult(diag, x)

        if len(self.A['InterOper']) == 2:
            ysave.array = (xx.array+yp1.array+yp2.array+yg.array)
        else:
            ysave.array = (xx.array+yp1.array+yp2.array+yp3.array+yg.array)
        y.array = ysave.array


def solve(A,b):

    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    pc = ksp.getPC()
    x = b.duplicate()
    ksp.setType('cg')
    pc.setType('python')
    pc.setPythonContext(Apply(A))
    ksp.setTolerances(A['tol'])
    ksp.setOperators(A['CurlShift'],A['CurlShift'])
    ksp.solve(b,x)


    return x, ksp.its
