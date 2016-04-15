from dolfin import assemble, MixedFunctionSpace
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import P as PrecondMulti
import NSprecond
import MaxwellPrecond as MP
import CheckPetsc4py as CP

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

    def __init__(self, Fspace, P,Q,F,L):
        self.Fspace = Fspace
        self.P = P
        self.Q = Q

        self.F = F
        self.L = L



        self.NS_is = PETSc.IS().createGeneral(range(Fspace[0].dim()+Fspace[1].dim()))
        self.M_is = PETSc.IS().createGeneral(range(Fspace[0].dim()+Fspace[1].dim(),Fspace[0].dim()+Fspace[1].dim()+Fspace[2].dim()+Fspace[3].dim()))

    def create(self, pc):
        self.diag = None
        kspNS = PETSc.KSP()
        kspNS.create(comm=PETSc.COMM_WORLD)
        pcNS = kspNS.getPC()
        kspNS.setType('gmres')
        pcNS.setType('python')
        pcNS.setPythonContext(NSprecond.PCDdirect(MixedFunctionSpace([self.Fspace[0],self.Fspace[1]]), self.Q, self.F, self.L))
        kspNS.setTolerances(1e-3)
        kspNS.setFromOptions()
        self.kspNS = kspNS

        kspM = PETSc.KSP()
        kspM.create(comm=PETSc.COMM_WORLD)
        pcM = kspM.getPC()

        kspM.setType('gmres')
        pcM.setType('python')
        kspM.setTolerances(1e-3)
        pcM.setPythonContext(MP.Direct(MixedFunctionSpace([self.Fspace[2],self.Fspace[3]])))
        kspM.setFromOptions()
        self.kspM = kspM

    def setUp(self, pc):
        Ans = PETSc.Mat().createPython([self.Fspace[0].dim()+self.Fspace[1].dim(), self.Fspace[0].dim()+self.Fspace[1].dim()])
        Ans.setType('python')
        Am = PETSc.Mat().createPython([self.Fspace[2].dim()+self.Fspace[3].dim(), self.Fspace[2].dim()+self.Fspace[3].dim()])
        Am.setType('python')
        NSp = PrecondMulti.NSP(self.Fspace,self.P,self.Q,self.L,self.F)
        Mp = PrecondMulti.MP(self.Fspace,self.P)
        Ans.setPythonContext(NSp)
        Ans.setUp()
        Am.setPythonContext(Mp)
        Am.setUp()

        self.kspNS.setOperators(Ans,self.P.getSubMatrix(self.NS_is,self.NS_is))
        self.kspM.setOperators(Am,self.P.getSubMatrix(self.M_is,self.M_is))

        # print self.kspNS.view()

    def apply(self, pc, x, y):
        # self.kspCurlCurl.setOperators(self.B)
        x1 = x.getSubVector(self.NS_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(self.M_is)
        y2 = x2.duplicate()
        reshist = {}
        def monitor(ksp, its, fgnorm):
            reshist[its] = fgnorm
        self.kspM.setMonitor(monitor)
        self.kspNS.solve(x1, y1)

        self.kspM.solve(x2, y2)
        print reshist
        for line in reshist.values():
            print line
        y.array = (np.concatenate([y1.array,y2.array]))
