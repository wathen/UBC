from dolfin import assemble
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np
import MatrixOperations as MO
class MatVec:
    def __init__(self, W, A, bc):
        self.A = A
        self.IS = MO.IndexSet(W)
        self.bc = bc


    def mult(self, A, x, y):

        u = assemble(self.A)
        for bc in self.bc:
            bc.apply(u)

        y.array = u.array()


    # def multTranspose(self, A, x, y):
    #     "y <- A' * x"
    #     self.mult(x, y)
class PetscMatVec:
    def __init__(self, W, A, bc):
        self.A = A
        self.IS = MO.IndexSet(W)
        self.bc = bc

    def create(self, A):

        u_k = Function(W['velocity'])
        p_k = Function(W['pressure'])
        b_k = Function(W['magnetic'])
        r_k = Function(W['multiplier'])


    def mult(self, A, x, y):
        u_k.vector()[:] = x.getSubVector(IS['velocity']).array
        p_k.vector()[:] = x.getSubVector(IS['pressure']).array
        b_k.vector()[:] = x.getSubVector(IS['magnetic']).array
        r_k.vector()[:] = x.getSubVector(IS['multiplier']).array

        u = assemble(self.A)
        for bc in self.bc:
            bc.apply(u)

        y.array = u.array()


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



