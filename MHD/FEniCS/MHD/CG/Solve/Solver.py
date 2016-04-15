from dolfin import assemble, MixedFunctionSpace, tic,toc
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import CheckPetsc4py as CP
import StokesPrecond
import NSprecond
import MaxwellPrecond as MP
import PETScIO as IO
import numpy as np
import P as PrecondMulti
import MHDprecond
import scipy.sparse as sp
from scipy.linalg import svd
import matplotlib.pylab as plt
from scipy.sparse.linalg.dsolve import spsolve


def solve(A,b,u,P,IS,Fspace,IterType,OuterTol,InnerTol,Mass=0,L=0,F=0):

    # u = b.duplicate()
    if IterType == "Full":

        kspOuter = PETSc.KSP().create()
        kspOuter.setTolerances(OuterTol)
        kspOuter.setType('fgmres')

        reshist = {}
        def monitor(ksp, its, fgnorm):
            reshist[its] = fgnorm
            print "OUTER:", fgnorm
        kspOuter.setMonitor(monitor)
        pcOutter = kspOuter.getPC()
        pcOutter.setType(PETSc.PC.Type.KSP)

        kspOuter.setOperators(A)

        kspOuter.max_it = 500

        kspInner = pcOutter.getKSP()
        kspInner.max_it = 100

        reshist1 = {}
        def monitor(ksp, its, fgnorm):
            reshist1[its] = fgnorm
            print "INNER:", fgnorm
        # kspInner.setMonitor(monitor)

        kspInner.setType('gmres')
        kspInner.setTolerances(InnerTol)

        pcInner = kspInner.getPC()
        pcInner.setType(PETSc.PC.Type.PYTHON)
        pcInner.setPythonContext(MHDprecond.D(Fspace,P, Mass, F, L))

        PP = PETSc.Mat().createPython([A.size[0], A.size[0]])
        PP.setType('python')
        p = PrecondMulti.P(Fspace,P,Mass,L,F)

        PP.setPythonContext(p)
        kspInner.setOperators(PP)

        tic()
        scale = b.norm()
        b = b/scale
        print b.norm()
        kspOuter.solve(b, u)
        u = u*scale
        print toc()

        # print s.getvalue()
        NSits = kspOuter.its
        del kspOuter
        Mits = kspInner.its
        del kspInner
        # print u.array
        return u,NSits,Mits

    NS_is = IS[0]
    M_is = IS[1]
    kspNS = PETSc.KSP().create()
    kspM = PETSc.KSP().create()
    kspNS.setTolerances(OuterTol)

    kspNS.setOperators(A.getSubMatrix(NS_is,NS_is),P.getSubMatrix(NS_is,NS_is))
    kspM.setOperators(A.getSubMatrix(M_is,M_is),P.getSubMatrix(M_is,M_is))
    # print P.symmetric
    A.destroy()
    P.destroy()
    if IterType == "MD":
        kspNS.setType('gmres')
        kspNS.max_it = 500

        pcNS = kspNS.getPC()
        pcNS.setType(PETSc.PC.Type.PYTHON)
        pcNS.setPythonContext(NSprecond.PCDdirect(MixedFunctionSpace([Fspace[0],Fspace[1]]), Mass, F, L))
    elif IterType == "CD":
        kspNS.setType('minres')
        pcNS = kspNS.getPC()
        pcNS.setType(PETSc.PC.Type.PYTHON)
        pcNS.setPythonContext(StokesPrecond.Approx(MixedFunctionSpace([Fspace[0],Fspace[1]])))
    reshist = {}
    def monitor(ksp, its, fgnorm):
        reshist[its] = fgnorm
        print fgnorm
    # kspNS.setMonitor(monitor)

    uNS = u.getSubVector(NS_is)
    bNS = b.getSubVector(NS_is)
    # print kspNS.view()
    scale = bNS.norm()
    bNS = bNS/scale
    print bNS.norm()
    kspNS.solve(bNS, uNS)
    uNS = uNS*scale
    NSits = kspNS.its
    kspNS.destroy()
    # for line in reshist.values():
    #     print line
    kspM.setFromOptions()
    kspM.setType(kspM.Type.MINRES)
    kspM.setTolerances(InnerTol)
    pcM = kspM.getPC()
    pcM.setType(PETSc.PC.Type.PYTHON)
    pcM.setPythonContext(MP.Direct(MixedFunctionSpace([Fspace[2],Fspace[3]])))

    uM = u.getSubVector(M_is)
    bM = b.getSubVector(M_is)
    scale = bM.norm()
    bM = bM/scale
    print bM.norm()
    kspM.solve(bM, uM)
    uM = uM*scale
    Mits = kspM.its
    kspM.destroy()
    u = IO.arrayToVec(np.concatenate([uNS.array, uM.array]))
    return u,NSits,Mits







def SchurPCD(Mass,L,F, backend):
    Mass = Mass.sparray()
    F = F.sparray()
    F = F + 1e-10*sp.identity(Mass.shape[0])
    F = PETSc.Mat().createAIJ(size=F.shape,csr=(F.indptr, F.indices, F.data))
    Mass.tocsc()
    Schur = sp.rand(Mass.shape[0], Mass.shape[0], density=0.00, format='csr')
    ksp = PETSc.KSP().create()
    pc = ksp.getPC()
    ksp.setOperators(F,F)
    ksp.setType('preonly')
    pc.setType('lu')
    OptDB = PETSc.Options()
    OptDB['pc_factor_shift_amount'] = "0.1"
    # OptDB['pc_factor_shift_type'] = 'POSITIVE_DEFINITE'
    OptDB['pc_factor_mat_ordering_type'] = 'amd'
    # OptDB['rtol']  = 1e-8
    # ksp.max_it = 5
    ksp.setFromOptions()
    for i in range(0,Mass.shape[0]):
        Col = Mass.getcol(i)
        Col = Col.toarray()
        Col = IO.arrayToVec(Col)
        u = Col.duplicate()
        ksp.solve(Col,u)
        C = u.duplicate()
        L.mult(u,C)
        # print C.array
        Schur[i,:] = C.array

    if backend == "PETSc":
        return PETSc.Mat().createAIJ(size=Schur.transpose().shape,csr=(Schur.transpose().indptr, Schur.transpose().indices, Schur.transpose().data))
    else:
        return Schur.transpose()

def SchurPCD2(Mass,L,F, backend):
    Mass = Mass.sparray()
    F = F.sparray()
    L = L.sparray()
    X = np.empty_like(Mass.todense())
    print X.shape
    Mass = Mass+ 1e-4*sp.identity(Mass.shape[0])
    Mass = Mass.toarray()
    print np.array(Mass[:,0]).T
    for i in range(X.shape[1]):
        print Mass[i,:].T
        X[i,:] = spsolve(F,Mass[:,i])    # print Schur.todense()

    Schur = L*X
    print Schur.diagonal()
    if backend == "PETSc":
        return PETSc.Mat().createAIJ(size=Schur.transpose().shape,csr=(Schur.transpose().indptr, Schur.transpose().indices, Schur.transpose().data))
    else:
        return Schur.transpose()


def ExactPrecond(P,Mass,L,F,Fspace,Backend="PETSc"):
    Schur = SchurPCD(Mass,L,F, "uBLAS")
    P = P.sparray()
    P.eliminate_zeros()

    P = P.tocsr()

            # P = IO.matToSparse(P)
    plt.spy(Schur)
    plt.savefig("plt2")
    P[Fspace[0].dim():Fspace[0].dim()+Fspace[1].dim(),Fspace[0].dim():Fspace[0].dim()+Fspace[1].dim()] = -Schur #-1e-10*sp.identity(Schur.shape[0])
    u, s, v = svd(P.todense())
    print "#####################",np.sum(s > 1e-10)
    P.eliminate_zeros()
    # P[-2,-2] += 1
    u, s, v = svd(P.todense())
    print "#####################",np.sum(s > 1e-10)

    if Backend == "PETSc":
        return PETSc.Mat().createAIJ(size=P.shape,csr=(P.indptr, P.indices, P.data))
    else:
        return P #.transpose()


