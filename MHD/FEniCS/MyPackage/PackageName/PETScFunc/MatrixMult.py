import petsc4py
import sys
petsc4py.init(sys.argv)
import petsc4py.PETSc as PETSc

import PETScMatOps

class SaddleMult(Matrix):
    def __init__(self, FS,A):
        self.FS = FS
        self.A = A
        self.IS = MO.IndexSet(FS)

    def mult(self, A, x, y):
        u = x.getSubVector(self.IS[0])
        p = x.getSubVector(self.IS[1])

        yu = PETScMatOps.PETScMultiDuplications(u,2)
        uOut = u.duplicate()

        self.A['A'].mult(u,yu[0])
        self.A['B'].multTranspose(p,yu[1])
        for i in range(2):
            uOut.axpy(1,yu[i])

        if self.A.has_key('Stab') == True:
            yp = PETScMatOps.PETScMultiDuplications(p,2)
            pOut = p.duplicate()
            self.A['B'].mult(u,yu[0])
            self.A['C'].mult(u,yu[1])
            for i in range(2):
                uOut.axpy(1,yu[i])
        else:
            pOut = p.duplicate()
            self.A['B'].mult(u,pOut)

        y.array = (np.concatenate([uOut.array, pOut.array]))

    def getMatrix(self,matrix):

        if matrix == 'Bt':
            return self.A[1]
        if matrix == 'A':
            return self.A[0]



class MHDmult(Matrix):
    def __init__(self, Fspace,A):
        self.Fspace = Fspace
        self.A = A
        self.IS = MO.IndexSet(Fspace)

    def mult(self, A, x, y):
        u = x.getSubVector(self.IS[0])
        p = x.getSubVector(self.IS[1])

        yu = PETScMatOps.PETScMultiDuplications(u,2)
        uOut = u.duplicate()

        self.A['A'].mult(u,yu[0])
        self.A['B'].multTranspose(p,yu[1])
        for i in range(2):
            uOut.axpy(1,yu[i])

        if self.A.has_key('Stab') == True:
            yp = PETScMatOps.PETScMultiDuplications(p,2)
            pOut = p.duplicate()
            self.A['B'].mult(u,yu[0])
            self.A['C'].mult(u,yu[1])
            for i in range(2):
                uOut.axpy(1,yu[i])
        else:
            pOut = p.duplicate()
            self.A['B'].mult(u,pOut)

        y.array = (np.concatenate([uOut.array, pOut.array]))

    def getMatrix(self,matrix):

        if matrix == 'Bt':
            return self.A[1]
        if matrix == 'A':
            return self.A[0]


