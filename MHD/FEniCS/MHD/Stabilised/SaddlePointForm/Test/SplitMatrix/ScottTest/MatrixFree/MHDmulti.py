from dolfin import assemble, Function

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

    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.mult(x, y)











class MHDmat(Matrix):
    def __init__(self, Fspace,A):
        self.Fspace = Fspace
        self.A = A
        self.IS = MO.IndexSet(Fspace)



    def mult(self, A, x, y):
        u = x.getSubVector(self.IS[0])
        p = x.getSubVector(self.IS[1])
        b = x.getSubVector(self.IS[2])
        r = x.getSubVector(self.IS[3])

        yu = MO.PETScMultiDuplications(u,3)
        uOut = u.duplicate()

        self.A[0].mult(u,yu[0])
        self.A[2].multTranspose(p,yu[1])
        if self.A[1] != None:
            self.A[1].multTranspose(b,yu[2])
        for i in range(3):
            uOut.axpy(1,yu[i])


        yp = MO.PETScMultiDuplications(p,2)
        pOut = p.duplicate()

        self.A[2].mult(u,yp[0])
        self.A[5].mult(p,yp[1])
        for i in range(2):
            pOut.axpy(1,yp[i])


        yb = MO.PETScMultiDuplications(b,3)
        bOut = b.duplicate()

        self.A[1].mult(u,yb[0])
        self.A[3].mult(b,yb[1])
        self.A[4].mult(r,yb[2])
        yb[0].scale(-1)

        for i in range(3):
            bOut.axpy(1,yb[i])


        yr = MO.PETScMultiDuplications(r,2)
        rOut = r.duplicate()

        self.A[4].multTranspose(b,yr[0])
        self.A[6].mult(r,yr[1])
        for i in range(2):
            rOut.axpy(1,yr[i])




        y.array = (np.concatenate([uOut.array, pOut.array, bOut.array, rOut.array]))

    def multTranspose(self, A, x, y):

        u = x.getSubVector(self.IS[0])
        p = x.getSubVector(self.IS[1])
        b = x.getSubVector(self.IS[2])
        r = x.getSubVector(self.IS[3])

        yu = MO.PETScMultiDuplications(u,3)
        uOut = u.duplicate()

        self.A[0].multTranspose(u,yu[0])
        self.A[2].multTranspose(p,yu[1])
        if self.A[1] != None:
            self.A[1].multTranspose(b,yu[2])
        yu[2].scale(-1)
        for i in range(3):
            uOut.axpy(1,yu[i])


        yp = MO.PETScMultiDuplications(p,2)
        pOut = p.duplicate()

        self.A[2].mult(u,yp[0])
        self.A[5].mult(p,yp[1])
        for i in range(2):
            pOut.axpy(1,yp[i])


        yb = MO.PETScMultiDuplications(b,3)
        bOut = b.duplicate()

        self.A[1].mult(u,yb[0])
        self.A[3].mult(b,yb[1])
        self.A[4].mult(r,yb[2])
        for i in range(3):
            bOut.axpy(1,yb[i])


        yr = MO.PETScMultiDuplications(r,2)
        rOut = r.duplicate()

        self.A[4].multTranspose(b,yr[0])
        self.A[6].mult(r,yr[1])
        for i in range(2):
            rOut.axpy(1,yr[i])




        y.array = (np.concatenate([uOut.array, pOut.array, bOut.array, rOut.array]))

    def getMatrix(self,matrix):

        if matrix == 'Ct':
            return self.A[1]
        elif matrix == 'Bt':
            return self.A[2]
        elif matrix == 'A':
            return self.A[0]

class MatFluid(Matrix):
    def __init__(self, Fspace,A):
        self.Fspace = Fspace
        self.A = A
        self.IS = MO.IndexSet(Fspace)



    def mult(self, A, x, y):
        u = x.getSubVector(self.IS[0])
        p = x.getSubVector(self.IS[1])

        yu = MO.PETScMultiDuplications(u,2)
        uOut = u.duplicate()

        self.A[0].mult(u,yu[0])
        self.A[2].multTranspose(p,yu[1])
        for i in range(2):
            uOut.axpy(1,yu[i])


        yp = MO.PETScMultiDuplications(p,2)
        pOut = p.duplicate()

        self.A[2].mult(u,yp[0])
        self.A[5].mult(p,yp[1])
        for i in range(2):
            pOut.axpy(1,yp[i])


        y.array = (np.concatenate([uOut.array, pOut.array]))

    def getMatrix(self,matrix):


        if matrix == 'Bt':
            return self.A[2]

class MatMag(Matrix):
    def __init__(self, Fspace,A):
        self.Fspace = Fspace
        self.A = A
        self.IS = MO.IndexSet(Fspace)



    def mult(self, A, x, y):
        b = x.getSubVector(self.IS[0])
        r = x.getSubVector(self.IS[1])


        yb = MO.PETScMultiDuplications(b,2)
        bOut = b.duplicate()


        self.A[3].mult(b,yb[0])
        self.A[4].mult(r,yb[1])
        for i in range(2):
            bOut.axpy(1,yb[i])


        yr = MO.PETScMultiDuplications(r,2)
        rOut = r.duplicate()

        self.A[4].multTranspose(b,yr[0])
        self.A[6].mult(r,yr[1])
        for i in range(2):
            rOut.axpy(1,yr[i])




        y.array = (np.concatenate([ bOut.array, rOut.array]))


class MatVec(Matrix):
    def __init__(self, W, A, bc,PrecondTmult):
        self.A = A
        self.IS = MO.IndexSet(W)
        self.bc = bc
        self.PrecondTmult = PrecondTmult


    def mult(self, A, x, y):

        u = assemble(self.A)
        for bc in self.bc:
            bc.apply(u)

        y.array = u.array()

    def getMatrix(self,matrix):

        if matrix == 'Ct':
            return self.PrecondTmult['Ct']
        elif matrix == 'Bt':
            return self.PrecondTmult['Bt']
        elif matrix == 'BC':
            return self.PrecondTmult['BC']





    # def multTranspose(self, A, x, y):
    #     "y <- A' * x"
    #     self.mult(x, y)
class PetscMatVec:
    def __init__(self, W, A, bc, PrecondTmult):
        self.A = A
        self.IS = MO.IndexSet(W)
        self.bc = bc
        self.W = W
        self.PrecondTmult = PrecondTmult

    def create(self, A):

        self.u_k = Function(self.W['velocity'])
        self.p_k = Function(self.W['pressure'])
        self.b_k = Function(self.W['magnetic'])
        self.r_k = Function(self.W['multiplier'])


    def mult(self, A, x, y):

        self.u_k.vector()[:] = x.getSubVector(self.IS['velocity']).array
        self.p_k.vector()[:] = x.getSubVector(self.IS['pressure']).array
        self.b_k.vector()[:] = x.getSubVector(self.IS['magnetic']).array
        self.r_k.vector()[:] = x.getSubVector(self.IS['multiplier']).array
        aVec, L_M, L_NS, Bt, CoupleT = forms.MHDmatvec(mesh, W, Laplacian, Laplacian,u_k,b_k,self.u_k,self.b_k,self.p_k,self.r_k, params,"Full","CG", SaddlePoint = "No")
        u = assemble(aVec)
        for bc in self.bc:
            bc.apply(u)

        y.array = u.array()

    def getMatrix(self,matrix):

        if matrix == 'Ct':
            return self.PrecondTmult['Ct']
        elif matrix == 'Bt':
            return self.PrecondTmult['Bt']
        elif matrix == 'BC':
            return self.PrecondTmult['BC']
    # def multTranspose(self, A, x, y):
    #     "y <- A' * x"
    #     self.mult(x, y)


class SplitMatVec:
    def __init__(self, W, A, bc):
        self.A = A
        self.IS = MO.IndexSet(W)
        self.bc = bc


    def mult(self, A, x, y):
        u = 0
        for i in range(len(self.A['velocity'])):
            u += assemble(self.A['velocity'][i])
        self.bc['velocity'].apply(u)
        p = 0
        for i in range(len(self.A['pressure'])):
            p += assemble(self.A['pressure'][i])
        b = 0
        for i in range(len(self.A['magnetic'])):
            b = assemble(self.A['magnetic'][i])
        self.bc['magnetic'].apply(b)
        r = 0
        for i in range(len(self.A['multiplier'])):
            r = assemble(self.A['multiplier'][i])
        self.bc['multiplier'].apply(r)
        y.array = np.concatenate((u.array(),p.array(),b.array(),r.array()), axis=0)

    # def multTranspose(self, A, x, y):
    #     "y <- A' * x"
    #     self.mult(x, y)




