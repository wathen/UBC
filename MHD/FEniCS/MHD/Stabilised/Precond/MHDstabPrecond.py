import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
from dolfin import tic, toc
import HiptmairSetup
import scipy.sparse as sp
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
        y111 = x2.duplicate()
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







class InnerOuterWITHOUT(BaseMyPC):

    def __init__(self, W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol,PP):
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
        self.PP = PP



        # print range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim())
        # ss
        self.P = P
        self.G = G
        self.AA = A
        self.tol = Hiptmairtol
        #self.u_is = PETSc.IS().createGeneral(range(self.W[0].dim()))
        #self.b_is = PETSc.IS().createGeneral(range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim()))
        #self.p_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim(),
        #    self.W[0].dim()+self.W[1].dim()+self.W[2].dim()))
        #self.r_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim()+self.W[2].dim(),
        #    self.W[0].dim()+self.W[1].dim()+self.W[2].dim()+self.W[3].dim()))
        self.u_is = PETSc.IS().createGeneral(range(self.W[0].dim()))
        self.p_is = PETSc.IS().createGeneral(range(self.W[0].dim(),
            self.W[0].dim()+self.W[1].dim()))
        self.b_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim(),
            self.W[0].dim()+self.W[1].dim()+self.W[2].dim()))
        self.r_is = PETSc.IS().createGeneral(range(self.W[0].dim()
            +self.W[1].dim()+self.W[2].dim(),self.W[0].dim()+self.W[1].dim()
            +self.W[2].dim()+self.W[3].dim()))

        

    def create(self, pc):
        print "Create"



    def setUp(self, pc):
        P,A, flag = pc.getOperators()
        print A.size
        print A
        A = self.PP
        self.Ct = A.getSubMatrix(self.u_is,self.b_is)
        self.C = A.getSubMatrix(self.b_is,self.u_is)
        self.D = A.getSubMatrix(self.r_is,self.b_is)
        self.Bt = A.getSubMatrix(self.u_is,self.p_is)
        self.B = A.getSubMatrix(self.p_is,self.u_is)
        self.Dt = A.getSubMatrix(self.b_is,self.r_is)
        # print self.Ct.view()
        kspF = PETSc.KSP()
        kspF.create(comm=PETSc.COMM_WORLD)
        pcF = kspF.getPC()
        kspF.setType('preonly')
        pcF.setType('lu')
        OptDB = PETSc.Options()
        OptDB["pc_factor_mat_ordering_type"] = "rcm"
        OptDB["pc_factor_mat_solver_package"] = "mumps"
        # asas
        # print B.shape
        # ss
        kspF.setFromOptions()
        kspF.setOperators(A.getSubMatrix(self.u_is,self.u_is),A.getSubMatrix(self.u_is,self.u_is))
        FC = sp.csr_matrix((self.W[0].dim(),self.W[1].dim()))
        row = np.array([])
        column = np.array([])
        data = np.array([])
        for i in range(0,self.W[1].dim()):
            x = A.getSubMatrix(self.u_is,self.u_is).getVecRight()
            rhs = self.Ct.getColumnVector(i)
            kspF.solve(rhs,x)
            xx = self.AA.getVecRight()
            self.C.mult(x,xx)
            xx = xx.array
            if xx.nonzero()[0].shape != 0:
                row = np.concatenate((row,xx.nonzero()[0]))
                data = np.concatenate((data,xx[xx.nonzero()[0]]))
                column = np.concatenate((column,i*np.ones(xx.nonzero()[0].shape)))
         
        CFC = sp.csr_matrix( (data,(row,column)), shape=(self.W[2].dim(),self.W[2].dim()) )
        print CFC.shape
        CFC = PETSc.Mat().createAIJ(size=CFC.shape,csr=(CFC.indptr, CFC.indices, CFC.data))
        print CFC.size, self.AA.size
        MX = self.AA - CFC
        # B = IO.matToSparse(MX).tocsr()
        # MO.StoreMatrix(B,"A")
        # print FC.todense()
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

        kspMX = PETSc.KSP()
        kspMX.create(comm=PETSc.COMM_WORLD)
        pcMX = kspMX.getPC()
        kspMX.setType('preonly')
        pcMX.setType('lu')
        OptDB = PETSc.Options()
        kspMX.setOperators(MX,MX)
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
        bu1 = x.getSubVector(self.u_is)
        bu1 = bu1 - self.Bt*xp
        xx = bu1.duplicate()
        self.kspF.solve(bu1,xx)

        bb = x.getSubVector(self.b_is)
        bb = bb - self.Dt*xr
        xb = bb.duplicate()
        #self.kspMX.solve(bb,xb)
        self.kspMX.solve(bb-self.C*xx,xb)
        bu1 = x.getSubVector(self.u_is)
        bu1 = bu1-self.Bt*xp
        bu2 = self.Bt*xp
        bu4 = self.Ct*xb
        XX = bu1.duplicate()
        xu = XX.duplicate() 
        # self.kspF.solve(bu1-bu4-bu2,xu) 
        self.kspF.solve(self.Ct*xb,XX)
        #self.kspF.solve(bu1-self.Ct*xb,xu)
        self.kspF.solve(bu1-XX,xu)

        y.array = (np.concatenate([xu.array, xb.array,xp.array,xr.array]))

    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime





class SaddleFromTest(BaseMyPC):

    def __init__(self, W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol,PP):
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
        self.PP = PP



        # print range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim())
        # ss
        self.P = P
        self.G = G
        self.AA = A
        self.tol = Hiptmairtol
        #self.u_is = PETSc.IS().createGeneral(range(self.W[0].dim()))
        #self.b_is = PETSc.IS().createGeneral(range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim()))
        #self.p_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim(),
        #    self.W[0].dim()+self.W[1].dim()+self.W[2].dim()))
        #self.r_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim()+self.W[2].dim(),
        #    self.W[0].dim()+self.W[1].dim()+self.W[2].dim()+self.W[3].dim()))
        self.u_is = PETSc.IS().createGeneral(range(self.W[0].dim()))
        self.p_is = PETSc.IS().createGeneral(range(self.W[0].dim(),
            self.W[0].dim()+self.W[1].dim()))
        self.b_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim(),
            self.W[0].dim()+self.W[1].dim()+self.W[2].dim()))
        self.r_is = PETSc.IS().createGeneral(range(self.W[0].dim()
            +self.W[1].dim()+self.W[2].dim(),self.W[0].dim()+self.W[1].dim()
            +self.W[2].dim()+self.W[3].dim()))

        

    def create(self, pc):
        print "Create"



    def setUp(self, pc):
        A, P, flag = pc.getOperators()
        print A.size
        print A
        self.Ct = A.getSubMatrix(self.u_is,self.b_is)
        self.C = A.getSubMatrix(self.b_is,self.u_is)
        self.D = A.getSubMatrix(self.r_is,self.b_is)
        self.Bt = A.getSubMatrix(self.u_is,self.p_is)
        self.B = A.getSubMatrix(self.p_is,self.u_is)
        self.Dt = A.getSubMatrix(self.b_is,self.r_is)
        # print self.Ct.view()
        MX = self.PP
        # B = IO.matToSparse(MX).tocsr()
        # MO.StoreMatrix(B,"A")
        # print FC.todense()
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

        kspMX = PETSc.KSP()
        kspMX.create(comm=PETSc.COMM_WORLD)
        pcMX = kspMX.getPC()
        kspMX.setType('preonly')
        pcMX.setType('lu')
        OptDB = PETSc.Options()
        kspMX.setOperators(MX,MX)
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
        bu1 = x.getSubVector(self.u_is)
        bu1 = bu1
        #- self.Bt*xp
        xx = bu1.duplicate()
        self.kspF.solve(bu1,xx)

        bb = x.getSubVector(self.b_is)
        bb = bb
        #- self.Dt*xr
        xb = bb.duplicate()
        #self.kspMX.solve(bb,xb)
        self.kspMX.solve(bb,xb)
        #-self.C*xx,xb)
        bu1 = x.getSubVector(self.u_is)
        bu1 = bu1
        #-self.Bt*xp
        bu2 = self.Bt*xp
        bu4 = self.Ct*xb
        XX = bu1.duplicate()
        xu = XX.duplicate() 
        # self.kspF.solve(bu1-bu4-bu2,xu) 
        #self.kspF.solve(self.Ct*xb,XX)
        #self.kspF.solve(bu1-self.Ct*xb,xu)
        self.kspF.solve(bu1-bu4,xu)
        #-XX,xu)

        y.array = (np.concatenate([xu.array, xb.array,xp.array,xr.array]))

    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime

