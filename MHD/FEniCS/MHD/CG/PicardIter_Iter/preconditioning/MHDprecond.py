import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import P as PrecondMulti

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

class Direct(BaseMyPC):

    def __init__(self, Fspace, P,Q,fp,L):
        self.Fspace = Fspace
        self.P = P
        self.Q = Q
        self.fp = fp
        self.L = L
        self.A = A
        self.u_is = PETSc.IS().createGeneral(range(Fspace[0].dim()))
        self.p_is = PETSc.IS().createGeneral(range(Fspace[0].dim(),Fspace[0].dim()+Fspace[1].dim()))
        self.b_is = PETSc.IS().createGeneral(range(Fspace[0].dim()+Fspace[1].dim(),Fspace[0].dim()+Fspace[1].dim()+Fspace[2].dim()))
        self.r_is = PETSc.IS().createGeneral(range(Fspace[0].dim()+Fspace[1].dim()+Fspace[2].dim(),Fspace[0].dim()+Fspace[1].dim()+Fspace[2].dim()+Fspace[3].dim()))
        self.NS_is = PETSc.IS().createGeneral(range(Fspace[0].dim()+Fspace[1].dim()))
        self.M_is = PETSc.IS().createGeneral(range(Fspace[0].dim()+Fspace[1].dim(),WFspace[0].dim()+Fspace[1].dim()+Fspace[2].dim()+Fspace[3].dim()))
    def create(self, pc):
        self.diag = None
        kspNS = PETSc.KSP()
        kspNS.create(comm=PETSc.COMM_WORLD)
        pcNS = kspNS.getPC()
        kspNS.setType('gmres')
        pcNS.setType('python')

        kspNS.setFromOptions()
        self.kspNS = kspNS


        kspM = PETSc.KSP()
        kspM.create(comm=PETSc.COMM_WORLD)
        pcM = kspM.getPC()
        kspM.setType('gmres')
        pcM.setType('python')

        kspM.setFromOptions()
        self.kspM = kspM



    def setUp(self, pc):
        NSp = PrecondMulti.NSP(self.Fspace,self.P,self.Mass,self.L,self.fp)
        Mp = PrecondMulti.MP(self.Fspace,self.P)

        self.kspNS.setOperators(NSp,self.P.getSubMatrix(NS_is,NS_is))
        self.kspM.setOperators(Mp,self.P.getSubMatrix(self.M_is,self.M_is))


    def apply(self, pc, x, y):
        # self.kspCurlCurl.setOperators(self.B)
        x1 = x.getSubVector(NS_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(M_is)
        y2 = x2.duplicate()
        self.kspNS.solve(x1, y1)
        self.kspM.solve(x2, y2)

        y.array = (np.concatenate([y1.array,y2.array]))




