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

class Matrix(object):
    def __init__(self):
        pass
    def create(self, mat):
        pass
    def destroy(self, mat):
        pass







class InnerOuterMAGNETICinverse(BaseMyPC):

    def __init__(self, W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol,Options):
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
        self.kspMX.solve(bb-xxr,xb)

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

    def __init__(self, W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol,Options):
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
        xb, its, self.HiptmairTime = HiptmairSetup.HiptmairApply(self.AA, bb-xxr, self.kspScalar, self.kspVector, self.G, self.P, self.tol)

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









class ApproxInv(BaseMyPC):

    def __init__(self, W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol,Options):
        self.W = W
        self.kspF = kspF
        self.kspA = kspA
        self.kspQ = kspQ
        self.Fp = Fp
        self.kspScalar = kspScalar
        self.kspCGScalar = kspCGScalar
        self.kspVector = kspVector
        self.Options = Options
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
        if self.Options == 'BT':
            b = x.getSubVector(self.b_is)
            Mxb = b.duplicate()
            self.kspMX.solve(b,Mxb)

            r = x.getSubVector(self.r_is)
            Lr = r.duplicate()
            self.kspScalar.solve(r, Lr)

            DL = b.duplicate()
            self.Dt.multTranspose(Lr,DL)
            K = b.duplicate()
            self.kspMX.solve(DL,K)

            DM = r.duplicate()
            self.Dt.mult(Mxb,DM)
            E = r.duplicate()
            self.kspScalar.solve(DM,E)

            p = x.getSubVector(self.p_is)
            Sp2 = p.duplicate()
            Sp3 = p.duplicate()
            Sp = p.duplicate()
            self.kspA.solve(p,Sp2)
            self.Fp.mult(Sp2,Sp3)
            self.kspQ.solve(Sp3,Sp)



            u = x.getSubVector(self.u_is)
            Fu = u.duplicate()
            Cb = u.duplicate()
            Bp = u.duplicate()
            self.Ct.multTranspose(Mxb,Cb)
            self.Bt.multTranspose(Sp,Bp)
            self.kspF.solve(u-Cb+Bp,Fu)



            y.array = (np.concatenate([Fu.array, -Sp.array, Mxb.array+K.array,E.array]))
        else:
            u = x.getSubVector(self.u_is)
            Fu = u.duplicate()
            self.kspF.solve(u,Fu)

            p = x.getSubVector(self.p_is)
            Sp2 = p.duplicate()
            Sp3 = p.duplicate()
            Sp = p.duplicate()
            self.kspA.solve(p,Sp2)
            self.Fp.mult(Sp2,Sp3)
            self.kspQ.solve(Sp3,Sp)

            b = x.getSubVector(self.b_is)
            Mxb = b.duplicate()
            self.kspMX.solve(b,Mxb)

            r = x.getSubVector(self.r_is)
            Lr = r.duplicate()
            self.kspScalar.solve(r, Lr)
            if self.Options == 'p4':
                Q = u.duplicate()
            else:
                Q1 = u.duplicate()
                self.Bt.multTranspose(Sp,Q1)
                Q = u.duplicate()
                self.kspF(Q1,Q)

            Y1 = u.duplicate()
            self.Ct.multTranspose(Mxb,Y1)
            Y = u.duplicate()
            self.kspF(Y1,Y)

            BF = p.duplicate()
            self.Bt.mult(Fu,BF)

            if self.Options == 'p3':
                H = p.duplicate()
            else:
                H1 = p.duplicate()
                H2 = p.duplicate()
                H = p.duplicate()
                self.kspA.solve(BF,H1)
                self.Fp.mult(H1,H2)
                self.kspQ.solve(H2,H)


            if self.Options == 'p3':
                J = p.duplicate()
            else:
                BY = p.duplicate()
                self.Bt.mult(Fu,BY)
                J1 = p.duplicate()
                J2 = p.duplicate()
                J = p.duplicate()
                self.kspA.solve(BY,J1)
                self.Fp.mult(J1,J2)
                self.kspQ.solve(J2,J)

            CF = b.duplicate()
            self.Ct.mult(Fu,CF)
            T = b.duplicate()
            self.kspMX.solve(CF,T)

            if self.Options == 'p4':
                V = b.duplicate()
            else:
                CQ = b.duplicate()
                self.Ct.mult(Q,CQ)
                V = b.duplicate()
                self.kspMX.solve(CQ,V)


            DL = b.duplicate()
            self.Dt.multTranspose(Lr,DL)
            K = b.duplicate()
            self.kspMX.solve(DL,K)

            DM = r.duplicate()
            self.Dt.mult(Mxb,DM)
            E = r.duplicate()
            self.kspScalar.solve(DM,E)


            y.array = (np.concatenate([Fu.array+Q.array-Y.array, H.array-Sp.array-J.array, T.array+V.array+Mxb.array+K.array,E.array]))

    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime















class ApproxInvApprox(BaseMyPC):

    def __init__(self, W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol,Options):
        self.W = W
        self.kspF = kspF
        self.kspA = kspA
        self.kspQ = kspQ
        self.Fp = Fp
        self.kspScalar = kspScalar
        self.kspCGScalar = kspCGScalar
        self.kspVector = kspVector
        self.Options = Options
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
        if self.Options == 'BT':
            b = x.getSubVector(self.b_is)
            Mxb = b.duplicate()
            # self.kspMX.solve(b,Mxb)
            Mxb, its, self.HiptmairTime = HiptmairSetup.HiptmairApply(self.AA, b, self.kspScalar, self.kspVector, self.G, self.P, self.tol)

            r = x.getSubVector(self.r_is)
            Lr = r.duplicate()
            self.kspScalar.solve(r, Lr)

            DL = b.duplicate()
            self.Dt.multTranspose(Lr,DL)
            K = b.duplicate()
            K, its, self.HiptmairTime = HiptmairSetup.HiptmairApply(self.AA, DL, self.kspScalar, self.kspVector, self.G, self.P, self.tol)

            DM = r.duplicate()
            self.Dt.mult(Mxb,DM)
            E = r.duplicate()
            self.kspScalar.solve(DM,E)

            p = x.getSubVector(self.p_is)
            Sp2 = p.duplicate()
            Sp3 = p.duplicate()
            Sp = p.duplicate()
            self.kspA.solve(p,Sp2)
            self.Fp.mult(Sp2,Sp3)
            self.kspQ.solve(Sp3,Sp)



            u = x.getSubVector(self.u_is)
            Fu = u.duplicate()
            Cb = u.duplicate()
            Bp = u.duplicate()
            self.Ct.multTranspose(Mxb,Cb)
            self.Bt.multTranspose(Sp,Bp)
            self.kspF.solve(u-Cb+Bp,Fu)



            y.array = (np.concatenate([Fu.array, -Sp.array, Mxb.array+K.array,E.array]))
        else:
            u = x.getSubVector(self.u_is)
            Fu = u.duplicate()
            self.kspF.solve(u,Fu)

            p = x.getSubVector(self.p_is)
            Sp2 = p.duplicate()
            Sp3 = p.duplicate()
            Sp = p.duplicate()
            self.kspA.solve(p,Sp2)
            self.Fp.mult(Sp2,Sp3)
            self.kspQ.solve(Sp3,Sp)

            b = x.getSubVector(self.b_is)
            Mxb = b.duplicate()
            Mxb, its, self.HiptmairTime = HiptmairSetup.HiptmairApply(self.AA, b, self.kspScalar, self.kspVector, self.G, self.P, self.tol)

            r = x.getSubVector(self.r_is)
            Lr = r.duplicate()
            self.kspScalar.solve(r, Lr)
            if self.Options == 'p4':
                Q = u.duplicate()
            else:
                Q1 = u.duplicate()
                self.Bt.multTranspose(Sp,Q1)
                Q = u.duplicate()
                self.kspF(Q1,Q)

            Y1 = u.duplicate()
            self.Ct.multTranspose(Mxb,Y1)
            Y = u.duplicate()
            self.kspF(Y1,Y)

            BF = p.duplicate()
            self.Bt.mult(Fu,BF)

            if self.Options == 'p3':
                H = p.duplicate()
            else:
                H1 = p.duplicate()
                H2 = p.duplicate()
                H = p.duplicate()
                self.kspA.solve(BF,H1)
                self.Fp.mult(H1,H2)
                self.kspQ.solve(H2,H)


            BY = p.duplicate()
            self.Bt.mult(Fu,BY)
            if self.Options == 'p3':
                J = p.duplicate()
            else:
                J1 = p.duplicate()
                J2 = p.duplicate()
                J = p.duplicate()
                self.kspA.solve(BY,J1)
                self.Fp.mult(J1,J2)
                self.kspQ.solve(J2,J)

            CF = b.duplicate()
            self.Ct.mult(Fu,CF)
            T, its, self.HiptmairTime = HiptmairSetup.HiptmairApply(self.AA, CF, self.kspScalar, self.kspVector, self.G, self.P, self.tol)

            if self.Options == 'p4':
                V = b.duplicate()
            else:
                CQ = b.duplicate()
                self.Ct.mult(Q,CQ)
                V, its, self.HiptmairTime = HiptmairSetup.HiptmairApply(self.AA, CQ, self.kspScalar, self.kspVector, self.G, self.P, self.tol)


            DL = b.duplicate()
            self.Dt.multTranspose(Lr,DL)
            K = b.duplicate()
            K, its, self.HiptmairTime = HiptmairSetup.HiptmairApply(self.AA, DL, self.kspScalar, self.kspVector, self.G, self.P, self.tol)

            DM = r.duplicate()
            self.Dt.mult(Mxb,DM)
            E = r.duplicate()
            self.kspScalar.solve(DM,E)


            y.array = (np.concatenate([Fu.array+Q.array-Y.array, H.array-Sp.array-J.array, T.array+V.array+Mxb.array+K.array,E.array]))

    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime




# class ApproxBT(BaseMyPC):

#     def __init__(self, W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol,Options):
#         self.W = W
#         self.kspF = kspF
#         self.kspA = kspA
#         self.kspQ = kspQ
#         self.Fp = Fp
#         self.kspScalar = kspScalar
#         self.kspCGScalar = kspCGScalar
#         self.kspVector = kspVector
#         self.Options = Options
#         # self.Bt = Bt
#         self.HiptmairIts = 0
#         self.CGits = 0



#         # print range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim())
#         # ss
#         self.P = P
#         self.G = G
#         self.AA = A
#         self.tol = Hiptmairtol
#         self.u_is = PETSc.IS().createGeneral(range(self.W[0].dim()))
#         self.p_is = PETSc.IS().createGeneral(range(self.W[0].dim(),self.W[0].dim()+self.W[1].dim()))
#         self.b_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim(),
#             self.W[0].dim()+self.W[1].dim()+self.W[2].dim()))
#         self.r_is = PETSc.IS().createGeneral(range(self.W[0].dim()+self.W[1].dim()+self.W[2].dim(),
#             self.W[0].dim()+self.W[1].dim()+self.W[2].dim()+self.W[3].dim()))



#     def create(self, pc):
#         print "Create"



#     def setUp(self, pc):
#         A, P = pc.getOperators()
#         print A.size
#         if A.type == 'python':
#             self.Ct = A.getPythonContext().getMatrix("Ct")
#             self.Bt = A.getPythonContext().getMatrix("Bt")
#         else:
#             self.Ct = A.getSubMatrix(self.b_is,self.u_is)
#             self.Bt = A.getSubMatrix(self.p_is,self.u_is)
#             self.Dt = A.getSubMatrix(self.r_is,self.b_is)
#         # print self.Ct.view()
#         #CFC = sp.csr_matrix( (data,(row,column)), shape=(self.W[1].dim(),self.W[1].dim()) )
#         #print CFC.shape
#         #CFC = PETSc.Mat().createAIJ(size=CFC.shape,csr=(CFC.indptr, CFC.indices, CFC.data))
#         #print CFC.size, self.AA.size
#         # MO.StoreMatrix(B,"A")
#         # print FC.todense()

#         OptDB = PETSc.Options()
#         OptDB["pc_factor_mat_ordering_type"] = "rcm"
#         OptDB["pc_factor_mat_solver_package"] = "mumps"

#         self.kspA.setType('preonly')
#         self.kspA.getPC().setType('lu')
#         self.kspA.setFromOptions()
#         self.kspA.setPCSide(0)

#         self.kspQ.setType('preonly')
#         self.kspQ.getPC().setType('lu')
#         self.kspQ.setFromOptions()
#         self.kspQ.setPCSide(0)

#         self.kspScalar.setType('preonly')
#         self.kspScalar.getPC().setType('lu')
#         self.kspScalar.setFromOptions()
#         self.kspScalar.setPCSide(0)

#         kspMX = PETSc.KSP()
#         kspMX.create(comm=PETSc.COMM_WORLD)
#         pcMX = kspMX.getPC()
#         kspMX.setType('preonly')
#         pcMX.setType('lu')
#         OptDB = PETSc.Options()
#         kspMX.setOperators(self.AA,self.AA)
#         self.kspMX = kspMX
#         # self.kspCGScalar.setType('preonly')
#         # self.kspCGScalar.getPC().setType('lu')
#         # self.kspCGScalar.setFromOptions()
#         # self.kspCGScalar.setPCSide(0)

#         self.kspVector.setType('preonly')
#         self.kspVector.getPC().setType('lu')
#         self.kspVector.setFromOptions()
#         self.kspVector.setPCSide(0)



#         print "setup"
#     def apply(self, pc, x, y):


#     def ITS(self):
#         return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime


def FluidSchur(A, b):
    if len(A) == 1:
        print "exact Schur complement"
        x = b.duplicate()
        A[0].solve(b, x)
        return x
    else:
        print "PCD Schur complement"
        x1 = b.duplicate()
        x2 = b.duplicate()
        x3 = b.duplicate()
        A[0].solve(b,x1)
        A[1].mult(x1,x2)
        A[2].solve(x2,x3)
        return x3


class ApproxInv(BaseMyPC):

    def __init__(self, W, kspF, kspA, kspQ,Fp,kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol,Options):
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
            self.C = A.getSubMatrix(self.u_is,self.b_is)
            self.B = A.getSubMatrix(self.u_is,self.p_is)
            self.D = A.getSubMatrix(self.b_is,self.r_is)
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

        bu = x.getSubVector(self.u_is)
        xu = bu.duplicate()

        bp = x.getSubVector(self.p_is)
        xp = bp.duplicate()

        bb = x.getSubVector(self.b_is)
        xb = bb.duplicate()

        br = x.getSubVector(self.r_is)
        xr = br.duplicate()

        self.kspF.solve(bu,xu)
        xp = FluidSchur([kspA, Fp, KspQ], bp)
        self.kspMX.solve(bb,xb)
        self.kspScalar.solve(br,xr)

        xp1 = xp.duplicate()
        self.B.mult(xu, xp1)
        barF = FluidSchur([kspA, Fp, KspQ], xp1)

        xu1 = xu.duplicate()
        barS = xu.duplicate()
        self.B.multTranspose(xp, xu1)
        self.kspF.solve(xu1, barS)

        xr1 = xr.duplicate()
        outR = xr.duplicate()
        self.D.mult(xb, xr1)
        self.kspScalar(xr1, outR)

        xb1 = xb.duplicate()
        xb2 = xb.duplicate()
        xb3 = xb.duplicate()
        xb4 = xb.duplicate()

        self.D.multTranspose(xr, xb1)
        self.kspMX.solve(xb1, xb2)
        self.X.mult(xp, xb3)
        self.kspMX.solve(xb3, xb4)
        outB = xb4 + xb + xb2

        xp1 = xu.duplicate()
        xp2 = xu.duplicate()
        xp3 = xp.duplicate()
        self.C.multTranspose(xb, xp1)
        self.kspF.solve(xp1, xp2)
        self.B.mult(xp2, xp3)
        xp4 = FluidSchur([kspA, Fp, KspQ], xp3)
        outP = barF - xp - xp4;

        xu1 = xu.duplicate()
        xu2 = xu.duplicate()
        self.B.multTranspose(barF, xu1)
        self.kspF.solve(xu1, xu2)
        outU = xu - xu2 + barS;


        y.array = (np.concatenate([outU.array, outP.array, outB.array, outR.array]))
    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime


