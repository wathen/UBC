import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
from dolfin import tic, toc
import HiptmairSetup
import PETScIO as IO
import scipy.sparse as sp
import MatrixOperations as MO
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



class BlockTriInv(BaseMyPC):

    def __init__(self, W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol):
        self.W = W
        self.kspF = kspF
        self.kspA = kspA
        self.kspQ = kspQ
        self.Fp = Fp
        self.kspScalar = kspScalar
        self.kspCGScalar = kspCGScalar
        self.kspVector = kspVector
        # self.Bt = Bt
        self.HiptmairIts = 0
        self.CGits = 0



        # print range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim())
        # ss
        self.P = P
        self.G = G
        self.AA = A
        self.tol = Hiptmairtol
        self.u_is = PETSc.IS().createGeneral(range(self.W[0].dim()))
        self.p_is = PETSc.IS().createGeneral(range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim()))
        self.b_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim(),
            self.W[0].dim()+self.W[1].dim()+self.W[2].dim()))
        self.r_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim()+self.W[2].dim(),
            self.W[0].dim()+self.W[1].dim()+self.W[2].dim()+self.W[3].dim()))



    def create(self, pc):
        print "Create"



    def setUp(self, pc):
        A, P = pc.getOperators()
        print A.size
        if A.type == 'python':
            self.Ct = A.getPythonContext().getMatrix("Ct")
            self.Bt = A.getPythonContext().getMatrix("Bt")
        else:
            self.Ct = A.getSubMatrix(self.b_is,self.u_is)
            self.Bt = A.getSubMatrix(self.p_is,self.u_is)
            self.Dt = A.getSubMatrix(self.r_is,self.b_is)


        OptDB = PETSc.Options()
        OptDB["pc_factor_mat_ordering_type"] = "rcm"
        OptDB["pc_factor_mat_solver_package"] = "umfpack"

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

        kspMX = PETSc.KSP()
        kspMX.create(comm=PETSc.COMM_WORLD)
        pcMX = kspMX.getPC()
        kspMX.setType('preonly')
        pcMX.setType('lu')
        OptDB = PETSc.Options()
        kspMX.setOperators(self.AA,self.AA)
        self.kspMX = kspMX
        # self.kspCGScalar.setType('preonly')
        # self.kspCGScalar.getPC().setType('lu')
        # self.kspCGScalar.setFromOptions()
        # self.kspCGScalar.setPCSide(0)

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
        xb = bb.duplicate()
        xxr = bb.duplicate()
        self.Dt.multTranspose(xr,xxr)
        self.kspMX.solve(bb,xb)

        bu1 = x.getSubVector(self.u_is)
        bu2 = bu1.duplicate()
        bu4 = bu1.duplicate()
        self.Bt.multTranspose(xp,bu2)
        self.Ct.multTranspose(xb,bu4)

        XX = bu1.duplicate()
        xu = XX.duplicate()
        self.kspF.solve(bu1-bu4+bu2,xu)
        #self.kspF.solve(bu1,xu)

        y.array = (np.concatenate([xu.array, -xp.array,xb.array,xr.array]))
    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime



class BlockTriApp(BaseMyPC):

    def __init__(self, W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol):
        self.W = W
        self.kspF = kspF
        self.kspA = kspA
        self.kspQ = kspQ
        self.Fp = Fp
        self.kspScalar = kspScalar
        self.kspCGScalar = kspCGScalar
        self.kspVector = kspVector
        # self.Bt = Bt
        self.HiptmairIts = 0
        self.CGits = 0



        # print range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim())
        # ss
        self.P = P
        self.G = G
        self.AA = A
        self.tol = Hiptmairtol
        self.u_is = PETSc.IS().createGeneral(range(self.W[0].dim()))
        self.p_is = PETSc.IS().createGeneral(range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim()))
        self.b_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim(),
            self.W[0].dim()+self.W[1].dim()+self.W[2].dim()))
        self.r_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim()+self.W[2].dim(),
            self.W[0].dim()+self.W[1].dim()+self.W[2].dim()+self.W[3].dim()))



    def create(self, pc):
        print "Create"



    def setUp(self, pc):
        A, P = pc.getOperators()
        print A.size
        if A.type == 'python':
            self.Ct = A.getPythonContext().getMatrix("Ct")
            self.Bt = A.getPythonContext().getMatrix("Bt")
        else:
            self.Ct = A.getSubMatrix(self.b_is,self.u_is)
            self.Bt = A.getSubMatrix(self.p_is,self.u_is)
            self.Dt = A.getSubMatrix(self.r_is,self.b_is)


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
        xb = bb.duplicate()
        #self.kspMX.solve(bb,xb)
        xxr = bb.duplicate()
        self.Dt.multTranspose(xr,xxr)
        xb, its, self.HiptmairTime = HiptmairSetup.HiptmairApply(self.AA, bb, self.kspScalar, self.kspVector, self.G, self.P, self.tol)

        bu1 = x.getSubVector(self.u_is)
        bu2 = bu1.duplicate()
        bu4 = bu1.duplicate()
        self.Bt.multTranspose(xp,bu2)
        self.Ct.multTranspose(xb,bu4)
        XX = bu1.duplicate()
        xu = XX.duplicate()
        self.kspF.solve(bu1-bu4+bu2,xu)
        #self.kspF.solve(bu1,xu)

        y.array = (np.concatenate([xu.array, -xp.array,xb.array,xr.array]))
    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime


def SchurComplement(kspF, B):
    n = min(B.size)
    A = sp.csr_matrix((n, n))
    row = []
    column = []
    data = np.zeros(0)
    for i in range(n):
        (y, u) = B.getVecs()
        kspF.solve(B.getColumnVector(i), u)
        B.multTranspose(u, y)
        if i == 0:
            data = y.array
            row = np.linspace(0, n-1, n)
            column = i*np.ones(n)
        else:
            row = np.concatenate([row, np.linspace(0,n-1,n)])
            column = np.concatenate([column, i*np.ones(n)])
            data = np.concatenate([data, y.array])

    A = sp.csr_matrix((data, (row, column)), shape=(n, n))
    return PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))



def FluidSchur(A, b):
    if len(A) == 1:
        print "exact Schur complement"
        x = b.duplicate()
        A[0].solve(b, x)
        return x
    else:
        x1 = b.duplicate()
        x2 = b.duplicate()
        x3 = b.duplicate()

        A[2].solve(b,x1)
        A[1].mult(x1,x2)
        A[0].solve(x2,x3)
        return x3


class ApproxInv(BaseMyPC):

    def __init__(self, W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol):
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

        self.P = P
        self.G = G
        self.AA = A
        self.tol = Hiptmairtol
        self.u_is = PETSc.IS().createGeneral(range(self.W[0].dim()))
        self.p_is = PETSc.IS().createGeneral(range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim()))
        self.b_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim(),
            self.W[0].dim()+self.W[1].dim()+self.W[2].dim()))
        self.r_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim()+self.W[2].dim(),
            self.W[0].dim()+self.W[1].dim()+self.W[2].dim()+self.W[3].dim()))
        self.FluidApprox = "Schur"


    def create(self, pc):
        print "Create"



    def setUp(self, pc):
        A, P = pc.getOperators()
        print A.size
        if A.type == 'python':
            self.Ct = A.getPythonContext().getMatrix("Ct")
            self.Bt = A.getPythonContext().getMatrix("Bt")
        else:
            self.C = A.getSubMatrix(self.b_is,self.u_is)
            self.B = A.getSubMatrix(self.p_is,self.u_is)
            self.D = A.getSubMatrix(self.r_is,self.b_is)

        OptDB = PETSc.Options()
        OptDB["pc_factor_mat_ordering_type"] = "rcm"
        OptDB["pc_factor_mat_solver_package"] = "umfpack"


        self.kspA.setType('cg')
        self.kspA.getPC().setType('hypre')
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

        kspMX = PETSc.KSP()
        kspMX.create(comm=PETSc.COMM_WORLD)
        pcMX = kspMX.getPC()
        kspMX.setType('preonly')
        pcMX.setType('lu')
        OptDB = PETSc.Options()
        kspMX.setOperators(self.AA,self.AA)
        self.kspMX = kspMX

        self.kspVector.setType('preonly')
        self.kspVector.getPC().setType('lu')
        self.kspVector.setFromOptions()
        self.kspVector.setPCSide(0)

        if self.FluidApprox == "Schur":
            Schur = SchurComplement(self.kspF, A.getSubMatrix(self.u_is, self.p_is))
            kspS = PETSc.KSP()
            kspS.create(comm=PETSc.COMM_WORLD)
            pcS = kspMX.getPC()
            kspS.setType('preonly')
            pcS.setType('lu')
            OptDB = PETSc.Options()
            kspS.setOperators(Schur, Schur)
            self.kspS = kspS

        print "setup"
    def apply(self, pc, x, y):

        bu = x.getSubVector(self.u_is)
        invF = bu.duplicate()

        bp = x.getSubVector(self.p_is)

        bb = x.getSubVector(self.b_is)
        invMX = bb.duplicate()

        br = x.getSubVector(self.r_is)
        invL = br.duplicate()

        self.kspF.solve(bu,invF)

        x1 = bp.duplicate()
        x2 = bp.duplicate()

        if self.FluidApprox == "Schur":
            invS = FluidSchur([self.kspA, self.Fp, self.kspQ], bp)
        else:
            invS = bp.duplicate()
            self.kspS.solve(bp, invS)
        self.kspMX.solve(bb,invMX)
        self.kspScalar.solve(br,invL)


        xp1 = invS.duplicate()
        self.B.mult(invF, xp1)
        if self.FluidApprox == "Schur":
            barF = FluidSchur([self.kspA, self.Fp, self.kspQ], xp1)
        else:
            barF = invS.duplicate()
            self.kspS.solve(xp1, barF)

        xu1 = invF.duplicate()
        barS = invF.duplicate()
        self.B.multTranspose(invS, xu1)
        self.kspF.solve(xu1, barS)

        # outR = (L(D*invMx));
        xr1 = invL.duplicate()
        outR = invL.duplicate()
        self.D.mult(invMX, xr1)
        self.kspScalar(xr1, outR)

        # outB = (Mx(C*barS) + invMx + Mx(D'*invL));
        xb1 = invMX.duplicate()
        xb2 = invMX.duplicate()
        xb3 = invMX.duplicate()
        xb4 = invMX.duplicate()
        self.D.multTranspose(invL, xb1)
        self.kspMX.solve(xb1, xb2)
        self.C.mult(barS, xb3)
        self.kspMX.solve(xb3, xb4)
        outB = xb4 + invMX + xb2

        # outP = barF - invS - Schur(B*F(C'*invMx));
        xp1 = invF.duplicate()
        xp2 = invF.duplicate()
        xp3 = invS.duplicate()
        self.C.multTranspose(invMX, xp1)
        self.kspF.solve(xp1, xp2)
        self.B.mult(xp2, xp3A)
        if self.FluidApprox == "Schur":
            xp4 = FluidSchur([self.kspA, self.Fp, self.kspQ], xp3)
        else:
            xp4 = invS.duplicate()
            self.kspS.solve(xp3, xp4)
        outP = barF - invS - xp4;

        xu1 = invF.duplicate()
        xu2 = invF.duplicate()
        self.B.multTranspose(barF, xu1)
        self.kspF.solve(xu1, xu2)
        outU = invF - xu2 + barS;

        y.array = (np.concatenate([outU.array, outP.array, outB.array, outR.array]))
    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime

