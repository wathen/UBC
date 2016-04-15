import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np


class LSCnew(object):

    def __init__(self, W,A,L,Bd,dBt):
        self.W = W
        self.A = A
        self.L = L
        self.Bd = Bd
        self.dBt = dBt

        self.u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
        self.p_is = PETSc.IS().createGeneral(range(W.sub(0).dim(),W.sub(0).dim()+W.sub(1).dim()))
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
        # kspLAMG.setFromOptions()
        # kspLAMG.max_it = 1
        kspLAMG.setFromOptions()
        self.kspLAMG = kspLAMG
        # print kspLAMG.view()
        nsp = PETSc.NullSpace().create(constant=True)
        kspLAMG.setNullSpace(nsp)
        kspNLAMG = PETSc.KSP()
        kspNLAMG.create(comm=PETSc.COMM_WORLD)
        pc = kspNLAMG.getPC()
        kspNLAMG.setType('preonly')
        pc.setType('lu')
        # pc.setFactorSolverPackage("pastix")
        # kspNLAMG.max_it = 1
        kspNLAMG.setFromOptions()
        kspLAMG.setFromOptions()
        self.kspNLAMG = kspNLAMG
        # print kspNLAMG.view()


    def setUp(self, pc):
        # self.P = P
        F = self.A.getSubMatrix(self.u_is,self.u_is)
        self.Bt = self.A.getSubMatrix(self.u_is,self.p_is)
        self.kspNLAMG.setOperators(F)

        self.P = self.Bd*F*self.dBt
        self.kspLAMG.setOperators(self.L)


    def apply(self, pc, x, y):
        # print 1000
        # self.kspLAMG.setOperators(self.B)
        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()
        yOut = y2.duplicate()

        # self.kspNLAMG.solve(x1, y1)
        self.kspLAMG.solve(-x2, y2)
        yy2 = self.P*y2
        self.kspLAMG.solve(yy2, yOut)

        x1 = x1 - self.Bt*yOut
        self.kspNLAMG.solve(x1, y1)

        y.array = (np.concatenate([y1.array, yOut.array]))


class LSC(object):

    def __init__(self, W,A,P,L):
        self.W = W
        self.A = A
        self.P = P
        self.L = L
        self.u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
        self.p_is = PETSc.IS().createGeneral(range(W.sub(0).dim(),W.sub(0).dim()+W.sub(1).dim()))
    def create(self, pc):
        self.diag = None
        kspLAMG = PETSc.KSP()
        kspLAMG.create(comm=PETSc.COMM_WORLD)
        pc = kspLAMG.getPC()
        kspLAMG.setType('preonly')
        pc.setType('lu')
        # pc.setFactorSolverPackage("pastix")

        kspLAMG.max_it = 1
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


    def setUp(self, pc):
        # self.P = P
        F = self.A.getSubMatrix(self.u_is,self.u_is)
        self.Bt = self.A.getSubMatrix(self.u_is,self.p_is)
        B = self.A.getSubMatrix(self.p_is,self.u_is)
        Q = self.P.getSubMatrix(self.u_is,self.u_is)

        self.kspNLAMG.setOperators(F)
        Pdiag = Q.getVecLeft()
        Q.getDiagonal(Pdiag)
        ones,invDiag = Q.getVecs()
        ones.set(1)
        invDiag.pointwiseDivide(ones,Pdiag)
        invDiag = Pdiag
        print F.view()
        F.diagonalScale(invDiag)
        self.Bt.diagonalScale(invDiag)

        # self.PP =PETSc.Mat().create()
        # self.PP.setSizes([self.W.sub(0).dim(),self.W.sub(0).dim()])

        # FBt =PETSc.Mat().create()
        # FBt.setSizes([self.W.sub(1).dim(),self.W.sub(0).dim()])

        # self.P1 =PETSc.Mat().create()
        # self.P.setSizes([self.W.sub(0).dim(),self.W.sub(0).dim()])

        FBt = F.matMult(self.Bt)
        self.P1 = B.matMult(self.Bt)
        self.PP = B.matMult(self.Bt)

        self.P1 = B*F*self.Bt
        self.PP = B*self.Bt

        self.kspLAMG.setOperators(self.PP)


    def apply(self, pc, x, y):
        # self.kspLAMG.setOperators(self.B)
        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()
        yOut = y2.duplicate()

        self.kspNLAMG.solve(x1, y1)


        self.kspLAMG.solve(x2, y2)
        yy2 = self.P1*y2
        self.kspLAMG.solve(yy2, yOut)
        # y1 = y1 - self.Bt*yOut
        y.array = (np.concatenate([y1.array, yOut.array]))



class PCD(object):

    def __init__(self,  W, Q,F,L):
        self.W = W
        self.Q = Q
        self.F = F
        self.L = L
        self.u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
        self.p_is = PETSc.IS().createGeneral(range(W.sub(0).dim(),W.sub(0).dim()+W.sub(1).dim()))
    def create(self, pc):
        self.diag = None
        kspLAMG = PETSc.KSP()
        kspLAMG.create(comm=PETSc.COMM_WORLD)
        pc = kspLAMG.getPC()
        kspLAMG.setType('richardson')
        pc.setType('hypre')
        # pc.setFactorSolverPackage("pastix")
        # OptDB = PETSc.Options()
        # OptDB['pc_factor_shift_amount'] = .1
        # OptDB['pc_factor_mat_ordering_type'] = 'rcm'
        # OptDB['pc_factor_mat_solver_package']  = 'umfpack'
        kspLAMG.max_it = 1
        kspLAMG.setFromOptions()
        self.kspLAMG = kspLAMG
        # print kspLAMG.view()

        kspNLAMG = PETSc.KSP()
        kspNLAMG.create(comm=PETSc.COMM_WORLD)
        pc = kspNLAMG.getPC()
        kspNLAMG.setType('richardson')
        pc.setType('hypre')
        # pc.setFactorSolverPackage("pastix")
        kspNLAMG.max_it = 1
        kspNLAMG.setFromOptions()
        self.kspNLAMG = kspNLAMG
        # print kspNLAMG.view()

        kspQCG = PETSc.KSP()
        kspQCG.create(comm=PETSc.COMM_WORLD)
        pc = kspQCG.getPC()
        kspQCG.setType('cg')
        pc.setType('jacobi')

        # pc.setType('icc')
        # pc.setFactorSolverPackage("pastix")

        # kspQCG.max_it = 4
        kspQCG.setFromOptions()
        self.kspQCG = kspQCG


    def setUp(self, pc):
        A, P, flag = pc.getOperators()
        # self.P = P
        self.Bt = P.getSubMatrix(self.u_is,self.p_is)
        F = P.getSubMatrix(self.u_is,self.u_is)
        del A, P


        self.kspNLAMG.setOperators(F)
        self.kspLAMG.setOperators(self.L)
        self.kspQCG.setOperators(self.Q)




    def apply(self, pc, x, y):
        # self.kspLAMG.setOperators(self.B)
        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()
        yOut = y2.duplicate()

        self.kspLAMG.solve(x2, y2)
        yy2 = self.F*y2
        self.kspQCG.solve(yy2, yOut)
        x1 = x1 - self.Bt*yOut
        self.kspNLAMG.solve(x1, y1)

        y.array = (np.concatenate([y1.array, yOut.array]))


class PCDdirect(object):

    def __init__(self, W, Q,F,L):
        self.W = W
        self.Q = Q
        self.F = F
        self.L = L
        self.u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
        self.p_is = PETSc.IS().createGeneral(range(W.sub(0).dim(),W.sub(0).dim()+W.sub(1).dim()))
    def create(self, pc):
        self.diag = None
        kspLAMG = PETSc.KSP()
        kspLAMG.create(comm=PETSc.COMM_WORLD)
        pc = kspLAMG.getPC()
        kspLAMG.setType('preonly')
        pc.setType('cholesky')
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


    def setUp(self, pc):
        A, P, flag = pc.getOperators()
        # self.P = P
        self.Bt = A.getSubMatrix(self.u_is,self.p_is)
        F = A.getSubMatrix(self.u_is,self.u_is)

        self.kspNLAMG.setOperators(F)
        self.kspLAMG.setOperators(self.L)
        self.kspQCG.setOperators(self.Q)




    def apply(self, pc, x, y):
        # self.kspLAMG.setOperators(self.B)
        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()
        yOut = y2.duplicate()

        self.kspLAMG.solve(x2, y2)
        yy2 = self.F*y2
        self.kspQCG.solve(yy2, yOut)
        x1 = x1 - self.Bt*yOut
        self.kspNLAMG.solve(x1, y1)
        y.array = (np.concatenate([y1.array, yOut.array]))
        # print y.array




