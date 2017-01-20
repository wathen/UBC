from dolfin import *
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import CheckPetsc4py as CP
import numpy as np

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

        self.u_is = PETSc.IS().createGeneral(range(self.Fspace[0].dim()))
        self.p_is = PETSc.IS().createGeneral(range(self.Fspace[0].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()))
        self.b_is = PETSc.IS().createGeneral(range(self.Fspace[0].dim()+self.Fspace[1].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim()))
        self.r_is = PETSc.IS().createGeneral(range(self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim()+self.Fspace[3].dim()))

        self.F = self.P.getSubMatrix(self.u_is,self.u_is)
        self.Bt = self.P.getSubMatrix(self.u_is,self.p_is)
        self.Ct = self.P.getSubMatrix(self.u_is,self.b_is)

        self.C = self.P.getSubMatrix(self.b_is,self.u_is)
        self.M = self.P.getSubMatrix(self.b_is,self.b_is)
        self.A = self.P.getSubMatrix(self.r_is,self.r_is)

        ksp = PETSc.KSP()
        ksp.create(comm=PETSc.COMM_WORLD)
        pc = ksp.getPC()
        ksp.setType('richardson')
        pc.setType('hypre')
        ksp.max_it = 1
        ksp.setOperators(self.FF)
        self.ksp = ksp
        print 1


    def mult(self, A, x, y):


        u =x.getSubVector(self.u_is)
        p =x.getSubVector(self.p_is)
        b =x.getSubVector(self.b_is)
        r =x.getSubVector(self.r_is)
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

        self.u_is = PETSc.IS().createGeneral(range(self.Fspace[0].dim()))
        self.p_is = PETSc.IS().createGeneral(range(self.Fspace[0].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()))
        self.b_is = PETSc.IS().createGeneral(range(self.Fspace[0].dim()+self.Fspace[1].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim()))
        self.r_is = PETSc.IS().createGeneral(range(self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim()+self.Fspace[3].dim()))

        self.F = self.A.getSubMatrix(self.u_is,self.u_is)
        self.Bt = self.A.getSubMatrix(self.u_is,self.p_is)
        self.Ct = self.A.getSubMatrix(self.u_is,self.b_is)

        self.C = self.A.getSubMatrix(self.b_is,self.u_is)
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


        u =x.getSubVector(self.u_is)
        p =x.getSubVector(self.p_is)
        b =x.getSubVector(self.b_is)
        r =x.getSubVector(self.r_is)
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

