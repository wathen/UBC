#!/usr/bin/python
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np
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

class TensorMass(BaseMyPC):

    def __init__(self, W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol,SchurCompApprox):
        self.W = W
        self.kspF = kspF
        self.kspA = kspA
        self.kspQ = kspQ
        self.Fp = Fp
        self.kspScalar = kspScalar
        self.kspCGScalar = kspCGScalar
        self.kspVector = kspVector
        self.HiptmairIts = 0
        self.CGits = 0
        self.SchurCompApprox = SchurCompApprox

        self.P = P
        self.G = G
        self.AA = A
        self.tol = Hiptmairtol
        self.u_is = PETSc.IS().createGeneral(range(self.W[0].dim()))
        self.b_is = PETSc.IS().createGeneral(range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim()))
        self.p_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim(),
            self.W[0].dim()+self.W[1].dim()+self.W[2].dim()))
        self.r_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim()+self.W[2].dim(),
            self.W[0].dim()+self.W[1].dim()+self.W[2].dim()+self.W[3].dim()))


    def create(self, pc):
        print "Create"


    def setUp(self, pc):
        A, P, flag = pc.getOperators()
        self.Ct = A.getSubMatrix(self.u_is,self.b_is)
        self.Bt = A.getSubMatrix(self.u_is,self.p_is)
        self.Dt = A.getSubMatrix(self.b_is,self.r_is)
        
        kspMX = PETSc.KSP()
        kspMX.create(comm=PETSc.COMM_WORLD)
        pcMX = kspMX.getPC()
        kspMX.setType('preonly')
        pcMX.setType('lu')
        if self.SchurCompApprox.size[0] == self.W[0].dim():
            F = A.getSubMatrix(self.u_is,self.u_is)
            self.kspF.setOperators(F+self.SchurCompApprox,F+self.SchurCompApprox)
            kspMX.setOperators(self.AA,self.AA)
        else:
            print self.AA.size, self.SchurCompApprox.size
            kspMX.setOperators(self.AA+self.SchurCompApprox,self.AA+self.SchurCompApprox)
        
        self.kspF.setType('preonly')
        self.kspF.getPC().setType('lu')
        self.kspF.setFromOptions()
        self.kspF.setPCSide(0)

        self.kspA.setType('preonly')
        self.kspA.getPC().setType('lu')
        self.kspA.setFromOptions()
        self.kspA.setPCSide(0)

        self.kspQ.setType('preonly')
        self.kspQ.getPC().setType('lu')
        self.kspQ.setFromOptions()
        self.kspQ.setPCSide(0)

        self.kspScalar.setType('preonly')
        self.kspScalar.getPC().setType('lu')
        self.kspScalar.setFromOptions()
        self.kspScalar.setPCSide(0)

        self.kspMX = kspMX

        self.kspVector.setType('preonly')
        self.kspVector.getPC().setType('lu')
        self.kspVector.setFromOptions()
        self.kspVector.setPCSide(0)



        print "setup"
    def apply(self, pc, x, y):

        br = x.getSubVector(self.r_is)
        xr = br.duplicate()
        self.kspScalar.solve(br, xr)

        # print self.D.size
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()
        y3 = x2.duplicate()
        xp = x2.duplicate()
        self.kspA.solve(x2,y2)
        self.Fp.mult(y2,y3)
        self.kspQ.solve(y3,xp)


        # self.kspF.solve(bu1-bu4-bu2,xu)

        bb = x.getSubVector(self.b_is)
        bb = bb - self.Dt*xr
        xb = bb.duplicate()
        self.kspMX.solve(bb,xb)

        bu1 = x.getSubVector(self.u_is)
        bu2 = self.Bt*xp
        bu4 = self.Ct*xb
        XX = bu1.duplicate()
        xu = XX.duplicate()
        self.kspF.solve(bu1-bu4-bu2,xu)
        #self.kspF.solve(bu1,xu)

        y.array = (np.concatenate([xu.array, xb.array,xp.array,xr.array]))
    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime

