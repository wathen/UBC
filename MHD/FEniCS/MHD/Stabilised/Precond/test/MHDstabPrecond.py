import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
from dolfin import tic, toc
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




class InnerOuter(BaseMyPC):

    def __init__(self, W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol,Bt):
        self.W = W
        self.kspF = kspF
        self.kspA = kspA
        self.kspQ = kspQ
        self.Fp = Fp
        self.kspScalar = kspScalar
        self.kspCGScalar = kspCGScalar
        self.kspVector = kspVector
        self.Bt = Bt
        self.HiptmairIts = 0
        self.CGits = 0



        self.P = P
        self.G = G
        self.A = A
        self.tol = Hiptmairtol
        self.u_is = PETSc.IS().createGeneral(range(self.W[0].dim()))
        self.p_is = PETSc.IS().createGeneral(range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim()))
        self.b_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim(),self.W[0].dim()+self.W[1].dim()+self.W[2].dim()))
        self.r_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim()+self.W[2].dim(),self.W[0].dim()+self.W[1].dim()+self.W[2].dim()+self.W[3].dim()))



    def create(self, pc):
        print "Create"


    # def setUp(self, pc):
    #     A, P, flag = pc.getOperators()
    #     print A.size
    #     self.Bt = A.getSubMatrix(self.u_is,self.p_is)
    #     print "setup"

    def apply(self, pc, x, y):


        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        y11 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()
        y3 = x2.duplicate()
        y4 = x2.duplicate()

        self.kspA.solve(x2,y2)
        self.Fp.mult(y2,y3)
        self.kspQ.solve(y3,y4)
        self.Bt.mult(y4,y11)
        self.kspF.solve(x1-y11,y1)

        x1 = x.getSubVector(self.b_is)
        yy1 = x1.duplicate()
        x2 = x.getSubVector(self.r_is)
        yy2 = x2.duplicate()

        # tic()
        yy1, its, self.HiptmairTime = HiptmairSetup.HiptmairApply(self.A, x1, self.kspScalar, self.kspVector, self.G, self.P, self.tol)
        # print "Hiptmair time: ", toc()
        self.HiptmairIts += its
        tic()
        self.kspCGScalar.solve(x2, yy2)
        self.CGtime = toc()

        y.array = (np.concatenate([y1.array, y4.array,yy1.array,yy2.array]))


    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime





class Test(BaseMyPC):

    def __init__(self, W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol,Bt,C):
        self.W = W
        self.kspF = kspF
        self.kspA = kspA
        self.kspQ = kspQ
        self.Fp = Fp
        self.kspScalar = kspScalar
        self.kspCGScalar = kspCGScalar
        self.kspVector = kspVector
        self.Bt = Bt
        self.HiptmairIts = 0
        self.CGits = 0
        self.C = C


        self.P = P
        self.G = G
        self.A = A
        self.tol = Hiptmairtol
        self.u_is = PETSc.IS().createGeneral(range(self.W[0].dim()))
        self.p_is = PETSc.IS().createGeneral(range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim()))
        self.b_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim(),self.W[0].dim()+self.W[1].dim()+self.W[2].dim()))
        self.r_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim()+self.W[2].dim(),self.W[0].dim()+self.W[1].dim()+self.W[2].dim()+self.W[3].dim()))



    def create(self, pc):
        print "Create"


    # def setUp(self, pc):
    #     A, P, flag = pc.getOperators()
    #     print A.size
    #     self.Bt = A.getSubMatrix(self.u_is,self.p_is)
    #     print "setup"

    # def test(self,ksp):
    #     self.norm = ksp.buildResidual()

    def apply(self, pc, x, y):
        # print self.norm
        x1 = x.getSubVector(self.b_is)
        yy1 = x1.duplicate()
        x2 = x.getSubVector(self.r_is)
        yy2 = x2.duplicate()
        # tic()
        yy1, its, self.HiptmairTime = HiptmairSetup.HiptmairApply(self.A, x1, self.kspScalar, self.kspVector, self.G, self.P, self.tol)
        # print "Hiptmair time: ", toc()
        self.HiptmairIts += its
        tic()
        self.kspCGScalar.solve(x2, yy2)
        self.CGtime = toc()

        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        y11 = x1.duplicate()
        y111 = x1.duplicate()

        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()
        y3 = x2.duplicate()
        y4 = x2.duplicate()

        self.kspA.solve(x2,y2)
        self.Fp.mult(y2,y3)
        self.kspQ.solve(y3,y4)
        self.Bt.mult(y4,y11)
        self.C.mult(yy1,y111)
        self.kspF.solve(x1-y11-y111,y1)



        y.array = (np.concatenate([y1.array, y4.array,yy1.array,yy2.array]))



    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime