#!/usr/bin/python
from dolfin import *
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
Print = PETSc.Sys.Print

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import PETScIO as IO
import scipy
import scipy.io as io
import CheckPetsc4py as CP
import MaxwellPrecond as MP
import StokesPrecond as SP
import time
import MatrixOperations as MO


class Mat2x2multi(Matrix):
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
        self.A[1].multTranspose(p,yu[1])
        for i in range(2):
            uOut.axpy(1,yu[i])


        pOut = p.duplicate()
        self.A[1].mult(u,pOut)
        y.array = (np.concatenate([uOut.array, pOut.array]))

    def getMatrix(self,matrix):

        if matrix == 'Bt':
            return self.A[1]
        if matrix == 'A':
            return self.A[0]