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
import time
import NSprecondSetup
import NSpreconditioner
import MHDstabPrecond

def solve(A,b,u,IS,Fspace,IterType,OuterTol,InnerTol,HiptmairMatrices,KSPlinearfluids,kspF,Fp,MatrixLinearFluids,kspFp):

    if IterType == "Full":
        kspOuter = PETSc.KSP().create()
        kspOuter.setTolerances(OuterTol)
        kspOuter.setType('fgmres')

        u_is = PETSc.IS().createGeneral(range(Fspace[0].dim()))
        p_is = PETSc.IS().createGeneral(range(Fspace[0].dim(),Fspace[0].dim()+Fspace[1].dim()))
        Bt = A.getSubMatrix(u_is,p_is)

        reshist = {}
        def monitor(ksp, its, fgnorm):
            reshist[its] = fgnorm
            print "OUTER:", fgnorm
        kspOuter.setMonitor(monitor)
        pcOutter = kspOuter.getPC()
        pcOutter.setType(PETSc.PC.Type.KSP)

        kspOuter.setOperators(A,A)
        kspOuter.max_it = 500

        kspInner = pcOutter.getKSP()
        kspInner.max_it = 100

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
        pcInner.setPythonContext(MHDstabPrecond.InnerOuterWITHOUT(Fspace,kspF, KSPlinearfluids[0], KSPlinearfluids[1],Fp, HiptmairMatrices[3], HiptmairMatrices[4], HiptmairMatrices[2], HiptmairMatrices[0], HiptmairMatrices[1], HiptmairMatrices[6],1e-6,A))

        PP = PETSc.Mat().createPython([A.size[0], A.size[0]])
        PP.setType('python')
        p = PrecondMulti.MultiApply(Fspace,A,HiptmairMatrices[6],MatrixLinearFluids[1],MatrixLinearFluids[0],kspFp, HiptmairMatrices[3])

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

    kspNS.setOperators(A.getSubMatrix(NS_is,NS_is))
    kspM.setOperators(A.getSubMatrix(M_is,M_is))
    del A

    uNS = u.getSubVector(NS_is)
    bNS = b.getSubVector(NS_is)
    kspNS.setType('gmres')
    pcNS = kspNS.getPC()
    kspNS.setTolerances(OuterTol)
    pcNS.setType(PETSc.PC.Type.PYTHON)
    if IterType == "MD":
        pcNS.setPythonContext(NSpreconditioner.NSPCD(MixedFunctionSpace([Fspace[0],Fspace[1]]), kspF, KSPlinearfluids[0], KSPlinearfluids[1],Fp))
    else:
        pcNS.setPythonContext(StokesPrecond.MHDApprox(MixedFunctionSpace([Fspace[0],Fspace[1]]), kspF, KSPlinearfluids[1]))

    scale = bNS.norm()
    bNS = bNS/scale
    start_time = time.time()
    kspNS.solve(bNS, uNS)
    print ("{:25}").format("NS, time: "), " ==>  ",("{:4f}").format(time.time() - start_time),("{:9}").format("   Its: "), ("{:4}").format(kspNS.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    uNS = scale*uNS

    NSits = kspNS.its
    # kspNS.destroy()
    # for line in reshist.values():
    #     print line

    kspM.setFromOptions()
    kspM.setType(kspM.Type.MINRES)
    kspM.setTolerances(InnerTol)
    pcM = kspM.getPC()
    pcM.setType(PETSc.PC.Type.PYTHON)
    pcM.setPythonContext(MP.Hiptmair(MixedFunctionSpace([Fspace[2],Fspace[3]]), HiptmairMatrices[3], HiptmairMatrices[4], HiptmairMatrices[2], HiptmairMatrices[0], HiptmairMatrices[1], HiptmairMatrices[6],1e-6))


    # x = x*scale
    uM = u.getSubVector(M_is)
    bM = b.getSubVector(M_is)

    scale = bM.norm()
    bM = bM/scale
    start_time = time.time()
    kspM.solve(bM, uM)
    print ("{:25}").format("Maxwell solve, time: "), " ==>  ",("{:4f}").format(time.time() - start_time),("{:9}").format("   Its: "), ("{:4}").format(kspM.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    uM = uM*scale

    Mits = kspM.its
    kspM.destroy()
    u = IO.arrayToVec(np.concatenate([uNS.array, uM.array]))

    return u,NSits,Mits








