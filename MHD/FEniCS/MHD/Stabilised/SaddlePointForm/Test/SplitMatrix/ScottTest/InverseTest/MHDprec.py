import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
from dolfin import tic, toc
import HiptmairSetup
import PETScIO as IO
import scipy.sparse as sp
import matplotlib.pylab as plt
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

class Matrix(object):
    def __init__(self):
        pass
    def create(self, mat):
        pass
    def destroy(self, mat):
        pass







class InnerOuterMAGNETICinverse(BaseMyPC):

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
        # print self.Ct.view()
        #CFC = sp.csr_matrix( (data,(row,column)), shape=(self.W[1].dim(),self.W[1].dim()) )
        #print CFC.shape
        #CFC = PETSc.Mat().createAIJ(size=CFC.shape,csr=(CFC.indptr, CFC.indices, CFC.data))
        #print CFC.size, self.AA.size
        # MO.StoreMatrix(B,"A")
        # print FC.todense()

        OptDB = PETSc.Options()
        OptDB["pc_factor_mat_ordering_type"] = "rcm"
        OptDB["pc_factor_mat_solver_package"] = "mumps"

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




class InnerOuterMAGNETICapprox(BaseMyPC):

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
        # print self.Ct.view()
        #CFC = sp.csr_matrix( (data,(row,column)), shape=(self.W[1].dim(),self.W[1].dim()) )
        #print CFC.shape
        #CFC = PETSc.Mat().createAIJ(size=CFC.shape,csr=(CFC.indptr, CFC.indices, CFC.data))
        #print CFC.size, self.AA.size
        # MO.StoreMatrix(B,"A")
        # print FC.todense()
        #self.kspF.setType('preonly')
        #self.kspF.getPC().setType('lu')
        #self.kspF.setFromOptions()
        #self.kspF.setPCSide(0)




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




class InnerOuter(BaseMyPC):

    def __init__(self, AA,W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol,F):
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
        self.F = F
        self.A = AA


        # print range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim())
        # ss
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

        self.Dt = self.A.getSubMatrix(self.b_is,self.r_is)
        self.Bt = self.A.getSubMatrix(self.u_is,self.p_is)

    def setUp(self, pc):
        A, P = pc.getOperators()
        print A.size




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
        #self.kspMX.solve(bb,xb)
        xb, its, self.HiptmairTime = HiptmairSetup.HiptmairApply(self.AA, bb, self.kspScalar, self.kspVector, self.G, self.P, self.tol)

        bu1 = x.getSubVector(self.u_is)
        bu2 = self.Bt*xp
        XX = bu1.duplicate()
        xu = XX.duplicate()
        self.kspF.solve(bu1-bu2,xu)
        #self.kspF.solve(bu1,xu)

        y.array = (np.concatenate([xu.array, xb.array,xp.array,xr.array]))
    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime




class P(Matrix):
    def __init__(self, Fspace,P,Mass,L,F,M):
        self.Fspace = Fspace
        self.P = P
        self.Mass = Mass
        self.L = L
        self.kspFp = F
        self.M = M
        # self.N = (n, n, n)
        # self.F = zeros([n+2]*3, order='f')

    def create(self, A):

        self.IS = MO.IndexSet(self.Fspace)


        self.F = self.P.getSubMatrix(self.IS[0],self.IS[0])
        self.Bt = self.P.getSubMatrix(self.IS[0],self.IS[2])
        self.Ct = self.P.getSubMatrix(self.IS[0],self.IS[1])

        self.C = self.P.getSubMatrix(self.IS[1],self.IS[0])
        self.A = self.P.getSubMatrix(self.IS[3],self.IS[3])

#        ksp = PETSc.KSP()
#        ksp.create(comm=PETSc.COMM_WORLD)
#        pc = ksp.getPC()
#        ksp.setType('preonly')
#        pc.setType('hypre')
#        ksp.max_it = 1
#        ksp.setOperators(self.FF)
#        self.ksp = ksp
        print 13333


    def mult(self, A, x, y):
        print 'multi apply'
        print 333
        u = x.getSubVector(self.IS[0])
        p = x.getSubVector(self.IS[2])
        b = x.getSubVector(self.IS[1])
        r = x.getSubVector(self.IS[3])
        FQp = p.duplicate()

        uOut = self.F*u+self.Bt*p+self.Ct*b
        Qp =self.Mass*p
        self.kspFp.solve(Qp,FQp)
        pOut = -self.L*FQp
        bOut = self.C*u+self.M*b
        rOut = self.A*r

        y.array = (np.concatenate([uOut.array, bOut.array, pOut.array, rOut.array]))
        print "$$$$$$$/$$$$$$$$"
        # print x.array


    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.mult(x, y)

    # def getSubMatrix(self, isrow, iscol, submat=None):
    #     submat = self.P.get











class ApproxInverse(BaseMyPC):

    def __init__(self, W, kspFs, kspBFsB, kspMX,kspL, FF):
        self.W = W
        self.kspFs = kspFs
        self.kspBFsB = kspBFsB
        self.kspMX = kspMX
        self.kspL = kspL
        self.FF = FF
        self.IS = MO.IndexSet(W)



    def create(self, pc):
        print "Create"



    def setUp(self, pc):
        A, P = pc.getOperators()
        print A.size
        if A.type == 'python':
            self.Ct = A.getPythonContext().getMatrix("Ct")
            self.Bt = A.getPythonContext().getMatrix("Bt")
        else:
            self.Ct = A.getSubMatrix(self.IS[2],self.IS[0])
            self.Bt = A.getSubMatrix(self.IS[1],self.IS[0])
            self.Dt = A.getSubMatrix(self.IS[3],self.IS[2])

        OptDB = PETSc.Options()
        OptDB["pc_factor_mat_ordering_type"] = "rcm"
        OptDB["pc_factor_mat_solver_package"] = "mumps"

        self.kspFs.setType('preonly')
        self.kspFs.getPC().setType('lu')
        self.kspFs.setFromOptions()
        self.kspFs.setPCSide(0)

        self.kspBFsB.setType('preonly')
        self.kspBFsB.getPC().setType('lu')
        self.kspBFsB.setFromOptions()
        self.kspBFsB.setPCSide(0)

        self.kspL.setType('preonly')
        self.kspL.getPC().setType('lu')
        self.kspL.setFromOptions()
        self.kspL.setPCSide(0)

        self.kspMX.setType('preonly')
        self.kspMX.getPC().setType('lu')
        self.kspMX.setFromOptions()
        self.kspMX.setPCSide(0)


        print "setup"
    def apply(self, pc, x, y):
        bu = x.getSubVector(self.IS[0])
        xu = bu.duplicate()
        self.kspFs.solve(bu,xu)

        bp = x.getSubVector(self.IS[1])
        xp = bp.duplicate()
        self.kspBFsB.solve(bp,xp)
        xp1 = self.FF*xp
        xp2 = bp.duplicate()
        self.kspL.solve(xp1,xp2)



        bb = x.getSubVector(self.IS[2])
        xb1 = bb.duplicate()
        self.kspMX.solve(bb,xb1)

        br = x.getSubVector(self.IS[1])
        xr = br.duplicate()
        self.kspL.solve(br,xr)

        xb2 = bb.duplicate()
        xb3 = bb.duplicate()
        self.Dt.multTranspose(xr, xb2)
        self.kspMX.solve(xb1,xb2)

        xr1 = br.duplicate()
        xr2 = br.duplicate()
        self.Dt.mult(xb1,xr1)
        self.kspL.solve(xr1,xr2)
        # print np.concatenate([xu.array, xp.array, xb1.array+xb3.array, xr2.array])
        # sss


        y.array = (np.concatenate([xu.array, -xp2.array, xb1.array+xb3.array, xr2.array]))
    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime

