import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
from dolfin import *








def create(A, VectorLaplacian, ScalarLaplacian):
    OptDB = PETSc.Options()
    # OptDB['pc_factor_shift_amount'] = 1
    # OptDB['pc_hypre_type'] = 'boomeramg'
    # OptDB['pc_hypre_boomeramg_strong_threshold']  = 0.5
    # OptDB['pc_hypre_boomeramg_grid_sweeps_all']  = 3
    OptDB['pc_factor_mat_ordering_type'] = 'amd'
    OptDB['pc_factor_mat_solver_package'] = 'mumps'

    kspVector = PETSc.KSP()
    kspVector.create(comm=PETSc.COMM_WORLD)
    pcVector = kspVector.getPC()
    kspVector.setType('preonly')
    pcVector.setType('lu')
    # kspVector.setFromOptions()
    # kspVector.max_it = 1
    OptDB = PETSc.Options()
    kspVector.setFromOptions()

    kspScalar = PETSc.KSP()
    kspScalar.create(comm=PETSc.COMM_WORLD)
    pcScalar = kspScalar.getPC()
    kspScalar.setType('preonly')
    pcScalar.setType('lu')
    # kspScalar.setFromOptions()
    # kspScalar.max_it = 1
    kspScalar.setFromOptions()
    kspVector.setOperators(VectorLaplacian,VectorLaplacian)
    kspScalar.setOperators(ScalarLaplacian,ScalarLaplacian)
    diag = A.getDiagonal()
    diag.scale(3.0/2)
    diag.reciprocal()
    return kspVector, kspScalar, diag




def Happly(x, kspVector, kspScalar, diag, P, G, Magnetic, bc, A):

    xhat = P[0].getVecRight()
    P[0].multTranspose(x,xhat)
    yp1 =P[0].getVecLeft()
    yhat =P[0].getVecRight()
    kspVector.solve(xhat, yhat)
    P[0].mult(yhat,yp1)
    xhat.destroy()
    yhat.destroy()
    # y.axpy(1.0,yp1)


    xhat = P[1].getVecRight()
    P[1].multTranspose(x,xhat)
    yp2 =P[1].getVecLeft()
    yhat =P[1].getVecRight()
    kspVector.solve(xhat, yhat)
    P[1].mult(yhat,yp2)
    xhat.destroy()
    yhat.destroy()
    # y.axpy(1.0,yp2)

    if len(P) == 3:
        xhat = P[2].getVecRight()
        P[2].multTranspose(x,xhat)
        yp3 =P[2].getVecLeft()
        yhat =P[2].getVecRight()
        kspVector.solve(xhat, yhat)
        P[2].mult(yhat,yp3)
        # y.axpy(1.0,yp3)


    xhat = G.getVecRight()
    G.multTranspose(x,xhat)
    yg =G.getVecLeft()
    yhat =G.getVecRight()
    kspScalar.solve(xhat, yhat)
    G.mult(yhat,yg)
    xhat.destroy()
    yhat.destroy()


    xx = x.duplicate()
    xsave = x.duplicate()
    for i in range(1,2):
        xx.pointwiseMult(diag, x-A*(xx))
        xsave += xx
    # y = x.duplicate()
    # x = Function(Magnetic)
    # x.vector()[:] = (yp1+yp2+yg).array
    # print x.vector().array()
    # bc.apply(x.vector())
    # print x.vector().array()

    # y.array = xx.array+x.vector().array()
    # return y
    if len(P) == 2:
        return (xsave+yp1+yp2+yg)
    else:
        return (xx+yp1+yp2+yp3+yg)




def cg(A, b, x, VectorLaplacian ,ScalarLaplacian , P, G,Magnetic, bc, imax=50, eps=1e-6):
    r = b.duplicate()
    p = b.duplicate()
    q = b.duplicate()
    kspVector, kspScalar, diag = create(A, VectorLaplacian, ScalarLaplacian)
    i = 0
    A.mult(x, r)
    r.aypx(-1, b)
    z = Happly(r, kspVector, kspScalar, diag, P, G, Magnetic, bc, A )
    z.copy(p)
    delta_0 = z.dot(r)
    delta = delta_0
    while i < imax and delta > delta_0 * eps**2:
        A.mult(p, q)
        alpha = delta / p.dot(q)
        x.axpy(+alpha, p)
        r.axpy(-alpha, q)
        z = Happly(r, kspVector, kspScalar, diag, P, G, Magnetic, bc, A)
        delta_old = delta
        delta = z.dot(r)
        beta = delta / delta_old
        p = z+beta*p
        i = i + 1
        print np.linalg.norm(r)
    return x, i#, delta**0.5