import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
from dolfin import tic, toc
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




class NSPCD(BaseMyPC):

    def __init__(self, W, kspF, kspA, kspQ,Fp):
        self.W = W
        self.kspF = kspF
        self.kspA = kspA
        self.kspQ = kspQ
        self.Fp = Fp
        self.u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
        self.p_is = PETSc.IS().createGeneral(range(W.sub(0).dim(),W.sub(0).dim()+W.sub(1).dim()))
        self.HiptmairIts = 0
        self.CGits = 0


    def create(self, pc):
        print "Create"


    def setUp(self, pc):
        A, P = pc.getOperators()

        if A.type != 'python':
            self.Bt = A.getSubMatrix(self.p_is,self.u_is)
        else:
            self.Bt = A.getPythonContext().getMatrix("Bt")
        print "setup"


    def apply(self, pc, x, y):
        # self.kspCurlCurl.setOperators(self.B)
        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        y11 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()
        y3 = x2.duplicate()
        y4 = x2.duplicate()

        # tic()

        self.kspA.solve(x2,y2)
        self.Fp.mult(y2,y3)
        self.kspQ.solve(y3,y4)
        self.Bt.multTranspose(y4,y11)
        self.kspF.solve(x1+y11,y1)
        y.array = (np.concatenate([y1.array, -y4.array]))

    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime










class NSLSC(BaseMyPC):

    def __init__(self, W, kspF, kspBQB,  QB):
        self.QB = QB
        self.W = W
        self.kspF = kspF
        self.kspBQB = kspBQB
        self.u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
        self.p_is = PETSc.IS().createGeneral(range(W.sub(0).dim(),W.sub(0).dim()+W.sub(1).dim()))

    def create(self, pc):
        print "Create"


    def setUp(self, pc):
        A, P= pc.getOperators()
        self.Bt = A.getSubMatrix(self.u_is,self.p_is)
        self.A = A.getSubMatrix(self.u_is,self.u_is)
        print "setup"


    def apply(self, pc, x, y):

        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()

        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()
        y3 = x1.duplicate()
        y4 = x1.duplicate()
        y5 = x2.duplicate()
        y6 = x2.duplicate()
        y7 = x1.duplicate()

        self.kspBQB.solve(x2, y2)
        self.QB.mult(y2,y3)
        self.A.mult(y3,y4)
        self.QB.multTranspose(y4,y5)
        self.kspBQB.solve(y5, y6)

        self.Bt.mult(y6,y7)
        self.kspF.solve(x1+y7,y1)

        y.array = (np.concatenate([y1.array, -y6.array]))




