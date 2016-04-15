from dolfin import assemble, MixedFunctionSpace,tic,toc
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import P as PrecondMulti
import NSprecond
import MaxwellPrecond as MP
import CheckPetsc4py as CP
import MHDapply as MHD

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




class D(BaseMyPC):

    def __init__(self, Fspace, P,Q,F,L):
        self.Fspace = Fspace
        self.P = P
        self.Q = Q

        self.F = F
        self.L = L

        self.u_is = PETSc.IS().createGeneral(range(self.Fspace[0].dim()))
        self.p_is = PETSc.IS().createGeneral(range(self.Fspace[0].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()))
        self.b_is = PETSc.IS().createGeneral(range(self.Fspace[0].dim()+self.Fspace[1].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim()))
        self.r_is = PETSc.IS().createGeneral(range(self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim()+self.Fspace[3].dim()))

        self.NS_is = PETSc.IS().createGeneral(range(Fspace[0].dim()+Fspace[1].dim()))
        self.M_is = PETSc.IS().createGeneral(range(Fspace[0].dim()+Fspace[1].dim(),Fspace[0].dim()+Fspace[1].dim()+Fspace[2].dim()+Fspace[3].dim()))

    def create(self, pc):
        self.diag = None
        kspLAMG = PETSc.KSP()
        kspLAMG.create(comm=PETSc.COMM_WORLD)
        pc = kspLAMG.getPC()
        kspLAMG.setType('preonly')
        pc.setType('lu')
        # pc.setFactorSolverPackage("pastix")
        OptDB = PETSc.Options()
        OptDB['pc_factor_shift_amount'] = .1
        OptDB['pc_factor_mat_ordering_type'] = 'rcm'
        OptDB['pc_factor_mat_solver_package']  = 'mumps'
        # kspLAMG.max_it = 1
        kspLAMG.setFromOptions()
        self.kspLAMG = kspLAMG
        # print kspLAMG.view()

        kspNLAMG = PETSc.KSP()
        kspNLAMG.create(comm=PETSc.COMM_WORLD)
        pc = kspNLAMG.getPC()
        kspNLAMG.setType('preonly')
        pc.setType('lu')
        # pc.setFactorSolverPackage("pastix")
        # kspNLAMG.max_it = 1
        kspNLAMG.setFromOptions()
        self.kspNLAMG = kspNLAMG
        # print kspNLAMG.view()

        kspQCG = PETSc.KSP()
        kspQCG.create(comm=PETSc.COMM_WORLD)
        pc = kspQCG.getPC()
        kspQCG.setType('preonly')
        pc.setType('lu')

        # pc.setType('icc')
        # pc.setFactorSolverPackage("pastix")

        # kspQCG.max_it = 4
        kspQCG.setFromOptions()
        self.kspQCG = kspQCG

        kspCurlCurl = PETSc.KSP()
        kspCurlCurl.create(comm=PETSc.COMM_WORLD)
        pcCurlCurl = kspCurlCurl.getPC()
        kspCurlCurl.setType('preonly')
        pcCurlCurl.setType('lu')
        # pc.setFactorSolverPackage("umfpack")
#
        # kspCurlCurl.max_it = 1
        kspCurlCurl.setFromOptions()
        self.kspCurlCurl = kspCurlCurl
        # print kspCurlCurl.view()

        kspLapl = PETSc.KSP()
        kspLapl.create(comm=PETSc.COMM_WORLD)
        pcLapl = kspLapl.getPC()
        kspLapl.setType('preonly')
        pcLapl.setType('lu')
        # pc.setFactorSolverPackage("umfpack")
        # kspLapl.max_it = 1
        kspLapl.setFromOptions()
        self.kspLapl = kspLapl

    def setUp(self, pc):

        self.Bt = self.P.getSubMatrix(self.u_is,self.p_is)
        F = self.P.getSubMatrix(self.u_is,self.u_is)

        self.kspNLAMG.setOperators(F)
        self.kspLAMG.setOperators(self.L)
        self.kspQCG.setOperators(self.Q)



        CurlCurl = self.P.getSubMatrix(self.b_is,self.b_is)
        Laplace = self.P.getSubMatrix(self.r_is,self.r_is)

        self.kspCurlCurl.setOperators(CurlCurl)
        self.kspLapl.setOperators(Laplace)

        # print self.kspNS.view()

    def apply(self, pc, x, y):


        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()
        yOut = y2.duplicate()

        self.kspLAMG.solve(-x2, y2)
        yy2 = self.F*y2
        self.kspQCG.solve(yy2, yOut)
        x1 = x1 - self.Bt*yOut
        self.kspNLAMG.solve(x1, y1)

        x1 = x.getSubVector(self.b_is)
        yy1 = x1.duplicate()
        x2 = x.getSubVector(self.r_is)
        yy2 = x2.duplicate()

        self.kspCurlCurl.solve(x1, yy1)
        self.kspLapl.solve(x2, yy2)

        y.array = (np.concatenate([y1.array, yOut.array,yy1.array,yy2.array]))

























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


# class second(BaseMyPC):

#     def __init__(self, Fspace, P,Q,F,L):
#         self.Fspace = Fspace
#         self.P = P
#         self.Q = Q

#         self.F = F
#         self.L = L

#         self.u_is = PETSc.IS().createGeneral(range(Fspace[0].dim()))
#         self.p_is = PETSc.IS().createGeneral(range(Fspace[0].dim(),Fspace[0].dim()+Fspace[1].dim()))
#         self.b_is = PETSc.IS().createGeneral(range(Fspace[0].dim()+Fspace[1].dim(),Fspace[0].dim()+Fspace[1].dim()+Fspace[2].dim()))
#         self.r_is = PETSc.IS().createGeneral(range(Fspace[0].dim()+Fspace[1].dim()+Fspace[2].dim(),Fspace[0].dim()+Fspace[1].dim()+Fspace[2].dim()+Fspace[3].dim()))
#         self.NS_is = PETSc.IS().createGeneral(range(Fspace[0].dim()+Fspace[1].dim()))
#         self.M_is = PETSc.IS().createGeneral(range(Fspace[0].dim()+Fspace[1].dim(),Fspace[0].dim()+Fspace[1].dim()+Fspace[2].dim()+Fspace[3].dim()))

#     def create(self, pc):
#         self.diag = None

#         ksp = PETSc.KSP()
#         ksp.create(comm=PETSc.COMM_WORLD)
#         ksp.setType('gmres')
#         pc.setType('python')
#         pc.setPythonContext(Direct(self.Fspace, self.P,self.Q, self.F, self.L))
#         ksp.setTolerances(1e-3)
#         ksp.setFromOptions()
#         self.ksp = ksp

#     def setUp(self, pc):
#         PP = PETSc.Mat().createPython([A.size[0], A.size[0]])
#         PP.setType('python')
#         p = PrecondMulti.P(Fspace,P,Mass,L,F)

#         PP.setPythonContext(p)
#         ksp.setOperators(PP)

#         # print self.kspNS.view()

#     def apply(self, pc, x, y):
#         # self.kspCurlCurl.setOperators(self.B)
#         x1 = x.getSubVector(self.NS_is)
#         y1 = x1.duplicate()
#         x2 = x.getSubVector(self.M_is)
#         y2 = x2.duplicate()
#         reshist = {}
#         def monitor(ksp, its, fgnorm):
#             reshist[its] = fgnorm
#         self.kspNS.setMonitor(monitor)
#         self.kspNS.solve(x1, y1)

#         self.kspM.solve(x2, y2)
#         print reshist
#         for line in reshist.values():
#             print line
#         y.array = (np.concatenate([y1.array,y2.array]))

