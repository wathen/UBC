import scipy.sparse as sp

import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import CheckPetsc4py as CP
import MatrixOperations as MO
import matplotlib.pylab as plt
from scipy.linalg import eigvals
def IndexSet(W):
    if str(W.__class__).find('list') == -1:
        n = W.num_sub_spaces()
        IS = [0]*n
        Start = 0
        End = W.sub(0).dim()
        for i in range(n):
            if i>0:
                Start += W.sub(i-1).dim()
                End += W.sub(i).dim()
            IS[i] = PETSc.IS().createGeneral(range(Start,End))
    else:
        n = len(W)
        IS = [0]*n
        Start = 0
        End = W[0].dim()
        for i in range(n):
            if i>0:
                Start += W[i-1].dim()
                End += W[i].dim()
            IS[i] = PETSc.IS().createGeneral(range(Start,End))
    return IS

def eigens(A, W, PCD, L):

    IS = IndexSet(W)
    F = A.getSubMatrix(IS[0],IS[0])
    Ct = A.getSubMatrix(IS[0],IS[1])
    Dt = A.getSubMatrix(IS[1],IS[3])
    M = A.getSubMatrix(IS[1],IS[1])
    Bt = A.getSubMatrix(IS[0],IS[2])
    C = A.getSubMatrix(IS[2],IS[2])
    F = CP.PETSc2Scipy(F)
    Ct = CP.PETSc2Scipy(Ct)
    Dt = CP.PETSc2Scipy(Dt)
    Bt = CP.PETSc2Scipy(Bt)
    L = CP.PETSc2Scipy(L)
    M = CP.PETSc2Scipy(M)
    Ap = CP.PETSc2Scipy(PCD[0])
    Mp = CP.PETSc2Scipy(PCD[1])
    Fp = CP.PETSc2Scipy(PCD[2])
    A = CP.PETSc2Scipy(A)
    C = CP.PETSc2Scipy(C)
    MO.StoreMatrix(Dt,"D")
    ssss
    Fpinv = sp.linalg.inv(Fp)
    Linv = sp.linalg.inv(L)
    MX = M+Dt*Linv*Dt.transpose()
    MXinv = sp.linalg.inv(MX)
    FX = F+Ct*MXinv*Ct.transpose()
    PCD = Mp*Fpinv*Ap
    Finv = sp.linalg.inv(F)
    PCD = -C+Bt.transpose()*Finv*Bt
    
    # print P.toarray()
    # print P.shape
    # P = P.tocsc()

    P = sp.bmat([[FX,Ct,Bt,None],[None,MX,None,None],[None,None,-PCD,None],[None,None,None,L]])
    return A, P

def eigensORIG(A,AA, W, PCD, L):

    IS = IndexSet(W)
    F = A.getSubMatrix(IS[0],IS[0])
    Ct = A.getSubMatrix(IS[0],IS[1])
    Dt = A.getSubMatrix(IS[1],IS[3])
    M = A.getSubMatrix(IS[1],IS[1])
    Bt = A.getSubMatrix(IS[0],IS[2])
    C = A.getSubMatrix(IS[2],IS[2])
    F = CP.PETSc2Scipy(F)
    Ct = CP.PETSc2Scipy(Ct)
    Dt = CP.PETSc2Scipy(Dt)
    Bt = CP.PETSc2Scipy(Bt)
    L = CP.PETSc2Scipy(L)
    M = CP.PETSc2Scipy(M)
    Ap = CP.PETSc2Scipy(PCD[0])
    Mp = CP.PETSc2Scipy(PCD[1])
    Fp = CP.PETSc2Scipy(PCD[2])
    A = CP.PETSc2Scipy(A)
    C = CP.PETSc2Scipy(C)
    Fpinv = sp.linalg.inv(Fp)
    Linv = sp.linalg.inv(L)
    MO.StoreMatrix(L,"L")
    MX = M+Dt*Linv*Dt.transpose()
    MXinv = sp.linalg.inv(MX)
    FX = F+Ct*MXinv*Ct.transpose()
    PCD = Mp*Fpinv*Ap
    #Finv = sp.linalg.inv(F)
    #PCD = -C + Bt.transpose()*Finv*Bt
    MO.StoreMatrix(Dt,"Dt")
    MO.StoreMatrix(L,"L")
    MO.StoreMatrix(Ct,"Ct")
    MO.StoreMatrix(MX,"MX")
    
    A = sp.bmat([[F,Bt,Ct,None],[Bt.transpose(),C,None,None],[-Ct.transpose(),None,M,Dt],[None,None,Dt.transpose(),None]])
    P = sp.bmat([[F,Bt,Ct,None],[None,-PCD,None,None],[-Ct.transpose(),None,MX,None],[None,None,None,L]])
    Pinner = sp.bmat([[F,Bt,None,None],[None,-PCD,None,None],[None,None,MX,None],[None,None,None,L]])
    Papprox = sp.bmat([[FX,Bt,Ct,None],[None,-PCD,None,None],[None,None,MX,None],[None,None,None,L]])
    return A, P, Pinner, Papprox


def eigensApprox(A, W, PCD, KSPL, MX, FX):
    IS = IndexSet(W)
    Ct = A.getSubMatrix(IS[0],IS[1])
    Dt = A.getSubMatrix(IS[1],IS[3])
    Bt = A.getSubMatrix(IS[0],IS[2])
    L = KSPL.getOperators()
    MX = CP.PETSc2Scipy(MX)
    FX = CP.PETSc2Scipy(FX)
    Ct = CP.PETSc2Scipy(Ct)
    Dt = CP.PETSc2Scipy(Dt)
    Bt = CP.PETSc2Scipy(Bt)
    L = CP.PETSc2Scipy(L[0])
    Ap = CP.PETSc2Scipy(PCD[0])
    Mp = CP.PETSc2Scipy(PCD[1])
    Fp = CP.PETSc2Scipy(PCD[2])
    A = CP.PETSc2Scipy(A)

    Fpinv = sp.linalg.inv(Fp)
    PCD = Mp*Fpinv*Ap
    # print P.toarray()
    # print P.shape
    # P = P.tocsc()

    P = sp.bmat([[FX,Ct,Bt,None],[None,MX,None,None],[None,None,-PCD,None],[None,None,None,L]])
    return A, P

def ShiftedCDtest(F):


    x, b = F.getVecs()

    x.set(1.0)
    b = F*x
    x.set(0.0)
    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    pc = ksp.getPC()
    ksp.setType('gmres')
    pc.setType('hypre')
    ksp.setOperators(F,F)
    ksp.solve(b,x)

    MO.PrintStr("Shifted CD operator iterations: "+str(ksp.its),80,'=','\n','\n')
    return ksp.its

