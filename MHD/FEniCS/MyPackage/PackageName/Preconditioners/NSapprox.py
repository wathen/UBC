import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from PackageName.PETScFunc import PETScMatOps

def PCD(A, b):

    u = PETScMatOps.PETScMultiDuplications(b,3)

    A['kspL'].solve(b,u[0])
    A['Fp'].mult(u[0],u[1])
    A['kspM'].solve(u[1],u[2])


    return u[2]


def LSC(A, b):


    u = b.duplicate()
    A['kspL'].solve(b,u)

    y = A['scaledBt'].getVecLeft()
    A['scaledBt'].mult(u,y)

    x = A['F'].getVecLeft()
    A['F'].mult(y,x)

    u.set(0)
    A['scaledBt'].multTranspose(x,u)

    y.destroy()
    y = u.duplicate()
    A['kspL'].solve(u,y)


    return y