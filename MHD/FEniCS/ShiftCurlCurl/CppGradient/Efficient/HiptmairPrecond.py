import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import HiptmairSetup
from dolfin import interpolate, Function
import PETScIO as IO


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




class HiptmairApply(BaseMyPC):

    def __init__(self,Fspace , B, kspMass, kspVector, kspScalar, diag, BC):
        self.Fspace = Fspace
        self.B = B
        self.kspMass = kspMass
        self.kspVector = kspVector
        self.kspScalar = kspScalar
        self.diag = diag
        self.BC = BC

    def create(self, pc):
        print "created"

    def setUp(self, pc):
        print "setup"
    def apply(self, pc, x, y):
        ysave = x.duplicate()

        f = HiptmairSetup.BCapply(self.Fspace[0],self.BC[0],x,"dolfin")
        fVec = interpolate(f,self.Fspace[2])
        self.BC[2].apply(fVec.vector())
        uVec = HiptmairSetup.FuncToPETSc(fVec)

        u = uVec.duplicate()
        self.kspVector.solve(uVec, u)

        f = HiptmairSetup.BCapply(self.Fspace[2],self.BC[2],u,"dolfin")
        fMag = interpolate(f,self.Fspace[0])
        self.BC[0].apply(fMag.vector())
        xVec = HiptmairSetup.FuncToPETSc(fMag)


        xMag = HiptmairSetup.BCapply(self.Fspace[0],self.BC[0],x)
        uGrad = HiptmairSetup.TransGradOp(self.kspMass,self.B,xMag)
        xLag = HiptmairSetup.BCapply(self.Fspace[1],self.BC[1],uGrad)

        u = uGrad.duplicate()
        self.kspScalar.solve(xLag, u)

        uGrad = HiptmairSetup.BCapply(self.Fspace[1],self.BC[1],u)
        uGrad1 = HiptmairSetup.GradOp(self.kspMass,self.B,uGrad)
        xMag = HiptmairSetup.BCapply(self.Fspace[0],self.BC[0],uGrad1)


        xx = x.duplicate()
        xx.pointwiseMult(self.diag, x)

        # print xVec.array, xMag.array
        # sss
        y.array = xx.array+xVec.array+xMag.array


class GSvector(BaseMyPC):

    def __init__(self, G, P, kspVector, kspScalar, diag):
        self.G = G
        self.P = P
        self.kspVector = kspVector
        self.kspScalar = kspScalar
        self.diag = diag

    def create(self, pc):
        print "created"

    def setUp(self, pc):
        print "setup"
    def apply(self, pc, x, y):
        ysave = x.duplicate()

        xhat = self.P[0].getVecRight()
        self.P[0].multTranspose(x,xhat)
        yp1 =self.P[0].getVecLeft()
        yhat =self.P[0].getVecRight()
        self.kspVector.solve(xhat, yhat)
        self.P[0].mult(yhat,yp1)
        xhat.destroy()
        yhat.destroy()

        xhat = self.P[1].getVecRight()
        self.P[1].multTranspose(x,xhat)
        yp2 =self.P[1].getVecLeft()
        yhat =self.P[1].getVecRight()
        self.kspVector.solve(xhat, yhat)
        self.P[1].mult(yhat,yp2)
        xhat.destroy()
        yhat.destroy()

        if len(self.P) == 3:
            xhat = self.P[2].getVecRight()
            self.P[2].multTranspose(x,xhat)
            yp3 =self.P[2].getVecLeft()
            yhat =self.P[2].getVecRight()
            self.kspVector.solve(xhat, yhat)
            self.P[2].mult(yhat,yp3)


        xhat = self.G.getVecRight()
        self.G.multTranspose(x,xhat)
        yg =self.G.getVecLeft()
        yhat =self.G.getVecRight()
        self.kspScalar.solve(xhat, yhat)
        self.G.mult(yhat,yg)
        xhat.destroy()
        yhat.destroy()
        xx = x.duplicate()
        xx.pointwiseMult(self.diag, x)

        if len(self.P) == 2:
            ysave.array = (xx.array+yp1.array+yp2.array+yg.array)
        else:
            ysave.array = (xx.array+yp1.array+yp2.array+yp3.array+yg.array)
        y.array = ysave.array
