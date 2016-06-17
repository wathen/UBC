import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import CheckPetsc4py as CP
import os
import scipy
def StoreMatrix(A,name):
    test ="".join([name,".mat"])
    scipy.io.savemat( test, {name: A},oned_as='row')

def SaveMatrices(W, level, A, Fluid, Magnetic):
    u_is = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs())
    p_is = PETSc.IS().createGeneral(W.sub(1).dofmap().dofs())
    b_is = PETSc.IS().createGeneral(W.sub(2).dofmap().dofs())
    r_is = PETSc.IS().createGeneral(W.sub(3).dofmap().dofs())

    F = CP.PETSc2Scipy(A.getSubMatrix(u_is, u_is))
    B = CP.PETSc2Scipy(A.getSubMatrix(u_is, p_is))
    C = -CP.PETSc2Scipy(A.getSubMatrix(u_is, b_is))
    M = CP.PETSc2Scipy(A.getSubMatrix(b_is, b_is))
    D = CP.PETSc2Scipy(A.getSubMatrix(b_is, r_is))
    Stab = CP.PETSc2Scipy(A.getSubMatrix(r_is, r_is))

    Fp = CP.PETSc2Scipy(Fluid['Fp'])
    Ap = CP.PETSc2Scipy(Fluid['Ap'])
    Qp = CP.PETSc2Scipy(Fluid['Qp'])
    Fs = CP.PETSc2Scipy(Fluid['Fs'])

    Lp = CP.PETSc2Scipy(Magnetic['Lp'])
    MX = CP.PETSc2Scipy(Magnetic['MX'])

    os.chdir("/Users/michaelwathen/Desktop/PhD/MHD/FEniCS/MHD/Stabilised/SaddlePointForm/Test/SplitMatrix/ScottTest/Hartman2D/matrix1")

    StoreMatrix(F, "F_"+str(level))
    StoreMatrix(B, "B_"+str(level))
    StoreMatrix(C, "C_"+str(level))
    StoreMatrix(M, "M_"+str(level))
    StoreMatrix(D, "D_"+str(level))
    StoreMatrix(Stab, "Stab_"+str(level))

    StoreMatrix(Fp, "Fp_"+str(level))
    StoreMatrix(Ap, "Ap_"+str(level))
    StoreMatrix(Qp, "Qp_"+str(level))
    StoreMatrix(Fs, "Fs_"+str(level))

    StoreMatrix(Lp, "Lp_"+str(level))
    StoreMatrix(MX, "MX_"+str(level))

    os.chdir("/Users/michaelwathen/Desktop/PhD/MHD/FEniCS/MHD/Stabilised/SaddlePointForm/Test/SplitMatrix/ScottTest/Hartman2D")
