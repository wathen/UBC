
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import MatrixOperations as MO
import numpy as np

class Matrix(object):
    def __init__(self):
        pass
    def create(self, mat):
        pass
    def destroy(self, mat):
        pass

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
        self.IS0 = PETSc.IS().createGeneral(range(self.Fspace[0].dim()))
        self.IS1 = PETSc.IS().createGeneral(range(self.Fspace[0].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()))
        self.IS2 = PETSc.IS().createGeneral(range(self.Fspace[0].dim()+self.Fspace[1].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim()))
        self.IS3 = PETSc.IS().createGeneral(range(self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim(),self.Fspace[0].dim()+self.Fspace[1].dim()+self.Fspace[2].dim()+self.Fspace[3].dim()))



    def create(self, A):
        
        self.IS = MO.IndexSet(self.Fspace)
       
        self.F = self.P.getSubMatrix(self.IS0,self.IS0)
        self.Bt = self.P.getSubMatrix(self.IS0,self.IS2)
        self.Ct = self.P.getSubMatrix(self.IS0,self.IS1)

        self.C = self.P.getSubMatrix(self.IS1,self.IS0)
        self.A = self.P.getSubMatrix(self.IS3,self.IS3)

        print 13333


    def mult(self, A, x, y):
        print 'multi apply'
        u = x.getSubVector(self.IS0)
        p = x.getSubVector(self.IS2)
        b = x.getSubVector(self.IS1)
        r = x.getSubVector(self.IS3)
        FQp = p.duplicate()

        uOut = self.F*u+self.Bt*p+self.Ct*b
        Qp =self.Mass*p
        self.kspFp.solve(Qp,FQp)
        pOut = -self.L*FQp
        bOut = self.C*u+self.M*b
        rOut = self.A*r

        y.array = (np.concatenate([uOut.array, bOut.array, pOut.array, rOut.array]))
        # print x.array

    def matMult(self, A, x, y):
        print 'multi apply'
        u = x.getSubVector(self.IS0)
        p = x.getSubVector(self.IS2)
        b = x.getSubVector(self.IS1)
        r = x.getSubVector(self.IS3)
        FQp = p.duplicate()

        uOut = self.F*u+self.Bt*p+self.Ct*b
        Qp =self.Mass*p
        self.kspFp.solve(Qp,FQp)
        pOut = -self.L*FQp
        bOut = self.C*u+self.M*b
        rOut = self.A*r

        y.array = (np.concatenate([uOut.array, bOut.array, pOut.array, rOut.array]))
        # print x.array



    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.mult(x, y)
