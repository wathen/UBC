from dolfin import assemble, MixedFunctionSpace,tic,toc
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import CheckPetsc4py as CP
import numpy as np
import MatrixOperations as MO

class P:
    def __init__(self, Fspace,P,Mass,L,F):
        self.Fspace = Fspace
        self.P = P
        self.Mass = Mass
        self.L = L
        self.FF = F

        # self.N = (n, n, n)
        # self.F = zeros([n+2]*3, order='f')

    def create(self, A):
        
        self.IS = MO.IndexSet(self.Fspace)


        self.F = self.P.getSubMatrix(self.IS[0],self.IS[0])
        self.Bt = self.P.getSubMatrix(self.IS[0],self.IS[2])
        self.Ct = self.P.getSubMatrix(self.IS[0],self.IS[1])

        self.C = self.P.getSubMatrix(self.IS[1],self.IS[0])
        self.M = self.P.getSubMatrix(self.IS[1],self.IS[1])
        self.A = self.P.getSubMatrix(self.IS[3],self.IS[3])

        ksp = PETSc.KSP()
        ksp.create(comm=PETSc.COMM_WORLD)
        pc = ksp.getPC()
        ksp.setType('preonly')
        pc.setType('hypre')
        ksp.max_it = 1
        ksp.setOperators(self.FF)
        self.ksp = ksp
        print 1


    def mult(self, A, x, y):


        u =x.getSubVector(self.IS[0])
        p =x.getSubVector(self.IS[2])
        b =x.getSubVector(self.IS[1])
        r =x.getSubVector(self.IS[3])
        FQp = p.duplicate()

        uOut = self.F*u+self.Bt*p+self.Ct*b
        Qp =self.Mass*p
        self.ksp.solve(Qp,FQp)
        pOut = -self.L*FQp
        bOut = self.C*u+self.M*b
        rOut = self.A*r

        y.array = (np.concatenate([uOut.array, pOut.array, bOut.array, rOut.array]))
        # print "$$$$$$$/$$$$$$$$"
        # print x.array


    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.mult(x, y)

    # def getSubMatrix(self, isrow, iscol, submat=None):
    #     submat = self.P.get


class MultiApply:
    def __init__(self, Fspace,A,M,Mass,L,kspFp,kspL):
        self.Fspace = Fspace
        self.M = M
        self.Mass = Mass
        self.L = L
        self.kspFp = kspFp
        self.kspL = kspL
        self.A = A
        # self.N = (n, n, n)
        # self.F = zeros([n+2]*3, order='f')

    def create(self, A):

        self.IS[0] = PETSc.IS().createGeneral(range(self.Fspace[0].dim()))
        self.IS[2] = PETSc.IS().createGeneral(range(self.Fspace[0].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()))
        self.IS[1] = PETSc.IS().createGeneral(range(self.Fspace[0].dim()+self.Fspace[1].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim()))
        self.IS[3] = PETSc.IS().createGeneral(range(self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim()+self.Fspace[3].dim()))

        self.F = self.A.getSubMatrix(self.IS[0],self.u_is)
        self.Bt = self.A.getSubMatrix(self.IS[0],self.IS[2])
        self.Ct = self.A.getSubMatrix(self.IS[0],self.IS[1])

        self.C = self.A.getSubMatrix(self.IS[1],self.IS[0])
        self.LL = self.kspL.getOperators()[0]
        print self.LL
    #     ksp = PETSc.KSP()
    #     ksp.create(comm=PETSc.COMM_WORLD)
    #     pc = ksp.getPC()
    #     ksp.setType('richardson')
    #     pc.setType('hypre')
    #     ksp.max_it = 1
    #     ksp.setOperators(self.FF)
    #     self.ksp = ksp
    #     print 1


    def mult(self, A, x, y):


        u =x.getSubVector(self.IS[0])
        p =x.getSubVector(self.IS[2])
        b =x.getSubVector(self.IS[1])
        r =x.getSubVector(self.IS[3])
        FQp = p.duplicate()

        uOut = self.F*u+self.Bt*p+self.Ct*b
        Qp =self.Mass*p
        self.kspFp.solve(Qp,FQp)
        pOut = -self.L*FQp
        bOut = self.C*u+self.M*b
        rOut = self.LL*r

        y.array = (np.concatenate([uOut.array, pOut.array, bOut.array, rOut.array]))
        # print "$$$$$$$/$$$$$$$$"
        # print x.array


    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.mult(x, y)

    # def getSubMatrix(self, isrow, iscol, submat=None):
    #     submat = self.P.get

