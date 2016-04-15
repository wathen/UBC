import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import MatrixOperations as MO
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

    def __init__(self, W, A):
        print 333
        self.W = W
        self.A = A
        IS = MO.IndexSet(W)
        self.u_is = IS[0]
        self.p_is = IS[1]

    def create(self, pc):
        self.diag = None
        kspL = PETSc.KSP()
        kspL.create(comm=PETSc.COMM_WORLD)
        pc = kspL.getPC()
        kspL.setType('preonly')
        pc.setType('lu')
        OptDB = PETSc.Options()
        # OptDB['pc_factor_shift_amount'] = 1
        OptDB['pc_factor_mat_ordering_type'] = 'rcm'
        OptDB['pc_factor_mat_solver_package']  = 'mumps'
        kspL.setFromOptions()
        self.kspL = kspL

        kspM = PETSc.KSP()
        kspM.create(comm=PETSc.COMM_WORLD)
        pc = kspM.getPC()
        kspM.setType('preonly')
        pc.setType('lu')
        kspM.setFromOptions()
        self.kspM = kspM
        # print kspM.view()


    def setUp(self, pc):
        A, P = pc.getOperators()
        L = A.getSubMatrix(self.u_is,self.u_is)
        self.kspM.setOperators(self.A,self.A)
        self.kspL.setOperators(L,L)


    def apply(self, pc, x, y):
        # print 1000
        # self.kspL.setOperators(self.B)
        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()
        # print 111

        self.kspM.solve(x2, y2)
        self.kspL.solve(x1, y1)

        y.array = (np.concatenate([y1.array, y2.array]))


class Approx(object):

    def __init__(self, W, A):
        self.W = W
        self.A = A
        IS = MO.IndexSet(W)
        self.u_is = IS[0]
        self.p_is = IS[1]
    def create(self, pc):
        kspL = PETSc.KSP()
        kspL.create(comm=PETSc.COMM_WORLD)
        pcL = kspL.getPC()
        kspL.setType('preonly')
        pcL.setType('gamg')
        # kspL.max_it = 1
        kspL.setFromOptions()
        self.kspL = kspL

        kspM = PETSc.KSP()
        kspM.create(comm=PETSc.COMM_WORLD)
        pcM = kspM.getPC()
        kspM.setType('preonly')
        pcM.setType('gamg')
        kspM.setFromOptions()
        self.kspM = kspM


    def setUp(self, pc):
        A, P = pc.getOperators()
        L = A.getSubMatrix(self.u_is,self.u_is)
        M = P.getSubMatrix(self.p_is,self.p_is)
        self.kspM.setOperators(M,M)
        self.kspL.setOperators(L,L)


    def apply(self, pc, x, y):
        # self.kspL.setOperators(self.B)
        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()

        self.kspL.solve(x1, y1)
        self.kspM.solve(x2, y2)

        y.array = (np.concatenate([y1.array, y2.array]))


class ApproxSplit(object):

    def __init__(self, W, A, M):
        self.W = W
        self.A = A
        self.M = M
        IS = MO.IndexSet(W)
        self.u_is = IS[0]
        self.p_is = IS[1]
    def create(self, pc):
        self.diag = None
        kspL = PETSc.KSP()
        kspL.create(comm=PETSc.COMM_WORLD)
        pcL = kspL.getPC()
        kspL.setType('preonly')
        pcL.setType('ml')
        # kspL.max_it = 1
        kspL.setFromOptions()
        self.kspL = kspL

        kspM = PETSc.KSP()
        kspM.create(comm=PETSc.COMM_WORLD)
        pcM = kspM.getPC()
        kspM.setType('cg')
        pcM.setType('jacobi')
        kspM.setFromOptions()
        self.kspM = kspM


    def setUp(self, pc):
        self.kspM.setOperators(self.M,self.M)
        self.kspL.setOperators(self.A,self.A)


    def apply(self, pc, x, y):
        # self.kspL.setOperators(self.B)
        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()

        self.kspL.solve(x1, y1)
        self.kspM.solve(x2, y2)

        y.array = (np.concatenate([y1.array, y2.array]))







class MHDApprox(object):

    def __init__(self, W, kspA, kspQ):
        self.W = W
        self.kspA = kspA
        self.kspQ = kspQ
        self.u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
        self.p_is = PETSc.IS().createGeneral(range(W.sub(0).dim(),W.sub(0).dim()+W.sub(1).dim()))


    def apply(self, pc, x, y):
        # self.kspL.setOperators(self.B)
        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()


        self.kspQ.solve(x2, y2)
        self.kspA.solve(x1, y1)

        y.array = (np.concatenate([y1.array, y2.array]))





def ApproxFunc(W, A, x, y):
    IS = MO.IndexSet(W)
    u_is = IS[0]
    p_is = IS[1]
    diag = None
    kspL = PETSc.KSP()
    kspL.create(comm=PETSc.COMM_WORLD)
    pcL = kspL.getPC()
    kspL.setType('preonly')
    pcL.setType('gamg')
    # kspL.max_it = 1
    kspL.setFromOptions()

    kspM = PETSc.KSP()
    kspM.create(comm=PETSc.COMM_WORLD)
    pcM = kspM.getPC()
    kspM.setType('cg')
    pcM.setType('jacobi')
    kspM.setFromOptions()


    L = A.getSubMatrix(u_is,u_is)
    M = A.getSubMatrix(p_is,p_is)
    kspM.setOperators(M,M)
    kspL.setOperators(L,L)


    # kspL.setOperators(self.B)
    x1 = x.getSubVector(u_is)
    y1 = x1.duplicate()
    x2 = x.getSubVector(p_is)
    y2 = x2.duplicate()

    kspL.solve(x1, y1)
    kspM.solve(x2, y2)

    y.array = (np.concatenate([y1.array, y2.array]))


def ApproxSplitFunc(W, A, M,x,y):
    W = W
    A = A
    M = M
    IS = MO.IndexSet(W)
    u_is = IS[0]
    p_is = IS[1]
    diag = None
    kspL = PETSc.KSP()
    kspL.create(comm=PETSc.COMM_WORLD)
    pcL = kspL.getPC()
    kspL.setType('preonly')
    pcL.setType('gamg')
    # kspL.max_it = 1
    kspL.setFromOptions()

    kspM = PETSc.KSP()
    kspM.create(comm=PETSc.COMM_WORLD)
    pcM = kspM.getPC()
    kspM.setType('cg')
    pcM.setType('jacobi')
    kspM.setFromOptions()


    kspM.setOperators(M,M)
    kspL.setOperators(A,A)


    x1 = x.getSubVector(u_is)
    y1 = x1.duplicate()
    x2 = x.getSubVector(p_is)
    y2 = x2.duplicate()

    kspL.solve(x1, y1)
    kspM.solve(x2, y2)

    y.array = (np.concatenate([y1.array, y2.array]))

