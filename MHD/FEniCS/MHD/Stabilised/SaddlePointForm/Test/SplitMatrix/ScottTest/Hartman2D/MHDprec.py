import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
from dolfin import tic, toc
import HiptmairSetup
import PETScIO as IO
import scipy.sparse as sp
# import matplotlib.pylab as plt
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





class InnerOuterMAGNETICinverse1(BaseMyPC):

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



class BlockSchur(BaseMyPC):

    def __init__(self, W):
        self.NS_is = PETSc.IS().createGeneral(range(W.sub(0).dim()+W.sub(1).dim()))
        self.M_is = PETSc.IS().createGeneral(range(W.sub(0).dim()+W.sub(1).dim(), W.dim()))

    def create(self, A):
        kspNS = PETSc.KSP()
        kspNS.create(comm=PETSc.COMM_WORLD)
        pcNS = kspNS.getPC()
        kspNS.setType('preonly')
        pcNS.setType('lu')
        OptDB = PETSc.Options()
        OptDB['pc_factor_mat_solver_package']  = "umfpack"
        OptDB['pc_factor_mat_ordering_type']  = "rcm"
        kspNS.setFromOptions()
        self.kspNS = kspNS

        kspM = PETSc.KSP()
        kspM.create(comm=PETSc.COMM_WORLD)
        pcM = kspM.getPC()
        kspM.setType('preonly')
        pcM.setType('lu')
        OptDB = PETSc.Options()
        OptDB['pc_factor_mat_solver_package']  = "umfpack"
        OptDB['pc_factor_mat_ordering_type']  = "rcm"
        kspM.setFromOptions()
        self.kspM = kspM

    def setUp(self, pc):
        A, P = pc.getOperators()
        self.kspNS.setOperators(P.getSubMatrix(self.NS_is, self.NS_is))
        self.kspM.setOperators(P.getSubMatrix(self.M_is, self.M_is))
        self.Bt = P.getSubMatrix(self.NS_is, self.M_is)

    def apply(self, pc, x, y):

        b = x.getSubVector(self.M_is)
        g = b.duplicate()
        u = x.getSubVector(self.NS_is)
        f = u.duplicate()

        self.kspM.solve(b, g)
        self.kspNS.solve(u-self.Bt*g, f)

        y.array = (np.concatenate([f.array, g.array]))



class BlockSchurComponetwise(BaseMyPC):

    def __init__(self, W, Fp, Mp, Ap):
        self.Fp = Fp
        self.Mp = Mp
        self.Ap = Ap
        self.NS_is = PETSc.IS().createGeneral(range(W.sub(0).dim()+W.sub(1).dim()))
        self.M_is = PETSc.IS().createGeneral(range(W.sub(0).dim()+W.sub(1).dim(), W.dim()))
        self.u_is = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs())
        self.b_is = PETSc.IS().createGeneral(W.sub(1).dofmap().dofs())
        self.p_is = PETSc.IS().createGeneral(W.sub(2).dofmap().dofs())
        self.r_is = PETSc.IS().createGeneral(W.sub(3).dofmap().dofs())

    def create(self, A):
        kspNS = PETSc.KSP()
        kspNS.create(comm=PETSc.COMM_WORLD)
        pcNS = kspNS.getPC()
        kspNS.setType('preonly')
        pcNS.setType('lu')
        OptDB = PETSc.Options()
        OptDB['pc_factor_mat_solver_package']  = "umfpack"
        OptDB['pc_factor_mat_ordering_type']  = "rcm"
        kspNS.setFromOptions()
        self.kspNS = kspNS

        kspAp = PETSc.KSP()
        kspAp.create(comm=PETSc.COMM_WORLD)
        pcAp = kspAp.getPC()
        kspAp.setType('preonly')
        pcAp.setType('lu')
        OptDB = PETSc.Options()
        OptDB['pc_factor_mat_solver_package']  = "umfpack"
        OptDB['pc_factor_mat_ordering_type']  = "rcm"
        kspAp.setFromOptions()
        self.kspAp = kspAp

        kspMp = PETSc.KSP()
        kspMp.create(comm=PETSc.COMM_WORLD)
        pcMp = kspMp.getPC()
        kspMp.setType('preonly')
        pcMp.setType('lu')
        OptDB = PETSc.Options()
        OptDB['pc_factor_mat_solver_package']  = "umfpack"
        OptDB['pc_factor_mat_ordering_type']  = "rcm"
        kspMp.setFromOptions()
        self.kspMp = kspMp

        kspL = PETSc.KSP()
        kspL.create(comm=PETSc.COMM_WORLD)
        pcL = kspL.getPC()
        kspL.setType('preonly')
        pcL.setType('lu')
        OptDB = PETSc.Options()
        OptDB['pc_factor_mat_solver_package']  = "umfpack"
        OptDB['pc_factor_mat_ordering_type']  = "rcm"
        kspL.setFromOptions()
        self.kspL = kspL

    def setUp(self, pc):

        A, P = pc.getOperators()
        self.kspNS.setOperators(A.getSubMatrix(self.NS_is, self.NS_is))
        self.kspMp.setOperators(self.Mp, self.Mp)
        self.kspAp.setOperators(self.Ap, self.Ap)
        self.kspL.setOperators(P.getSubMatrix(self.r_is, self.r_is))
        self.Bt = P.getSubMatrix(self.u_is, self.p_is)

    def apply(self, pc, x, y):

        p = x.getSubVector(self.p_is)
        r = x.getSubVector(self.r_is)

        rOut = r.duplicate()
        self.kspL.solve(r, rOut)

        p1 = p.duplicate()
        self.kspAp.solve(p, p1)
        p2 = p.duplicate()
        self.Fp.mult(p1, p2)
        p3 = p.duplicate()
        self.kspMp.solve(p2, p3)

        ub = x.getSubVector(self.NS_is)
        u = x.getSubVector(self.u_is) + self.Bt*p3
        b = x.getSubVector(self.b_is)

        ub.array = np.concatenate([u.array, b.array])
        f = ub.duplicate()
        self.kspNS.solve(ub, f)

        y.array = (np.concatenate([f.array, -p3.array, rOut.array]))






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
            self.C = A.getSubMatrix(self.b_is,self.u_is)
            self.B = A.getSubMatrix(self.p_is,self.u_is)
            self.D = A.getSubMatrix(self.r_is,self.b_is)
        # print self.Ct.view()
        #CFC = sp.csr_matrix( (data,(row,column)), shape=(self.W[1].dim(),self.W[1].dim()) )
        #print CFC.shape
        #CFC = PETSc.Mat().createAIJ(size=CFC.shape,csr=(CFC.indptr, CFC.indices, CFC.data))
        #print CFC.size, self.AA.size
        # MO.StoreMatrix(B,"A")
        # print FC.todense()

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

        bu = x.getSubVector(self.u_is)
        invF = bu.duplicate()

        bp = x.getSubVector(self.p_is)

        bb = x.getSubVector(self.b_is)
        invMX = bb.duplicate()

        br = x.getSubVector(self.r_is)
        invL = br.duplicate()

        self.kspF.solve(bu,invF)
        invS = FluidSchur([self.kspA, self.Fp, self.kspQ], bp)
        self.kspMX.solve(bb,invMX)
        self.kspScalar.solve(br,invL)


        # outU = invF - F(B'*barF) + barS;

        xp1 = invS.duplicate()
        self.B.mult(invF, xp1)
        barF = FluidSchur([self.kspA, self.Fp, self.kspQ], xp1)

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
        self.B.mult(xp2, xp3)
        xp4 = FluidSchur([self.kspA, self.Fp, self.kspQ], xp3)
        outP = barF - invS - xp4;

        # outU = invF - F(B'*barF) + barS;
        xu1 = invF.duplicate()
        xu2 = invF.duplicate()
        self.B.multTranspose(barF, xu1)
        self.kspF.solve(xu1, xu2)
        outU = invF - xu2 + barS;

        y.array = (np.concatenate([outU.array, outP.array, outB.array, outR.array]))
    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime

