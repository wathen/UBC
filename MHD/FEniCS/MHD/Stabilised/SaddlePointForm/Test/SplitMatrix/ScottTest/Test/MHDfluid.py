import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
from dolfin import *
import common
import numpy as np
import ExactSol
import MatrixOperations as MO
import MHDmatrixPrecondSetup as PrecondSetup
import BiLinear as forms
import time
import CheckPetsc4py as CP
import NSprecondSetup
import Solver as S


LevelN = 1
Level = np.zeros(LevelN)
Wdim = np.zeros(LevelN)
Vdim = np.zeros(LevelN)
Pdim = np.zeros(LevelN)
Mdim = np.zeros(LevelN)
Ldim = np.zeros(LevelN)

for i in range(0, LevelN):
    Level[i] = i
    n = int(2**(Level[i]))

    mesh = UnitSquareMesh(n, n)
    parameters['reorder_dofs_serial'] = False
    Velocity = VectorFunctionSpace(mesh, "CG", 2)
    Pressure = FunctionSpace(mesh, "CG", 1)
    Magnetic = FunctionSpace(mesh, "N1curl", 1)
    Lagrange = FunctionSpace(mesh, "CG", 1)

    W = MixedFunctionSpace([Velocity, Pressure, Magnetic, Lagrange])

    Vdim[i] = Velocity.dim()
    Pdim[i] = Pressure.dim()
    Mdim[i] = Magnetic.dim()
    Ldim[i] = Lagrange.dim()
    Wdim[i] = W.dim()
    print "\n\nW:  ",Wdim[i],"Velocity:  ",Vdim[i],"Pressure:  ",Pdim[i],"Magnetic:  ",Mdim[i],"Lagrange:  ",Ldim[i],"\n\n"
    dim = [Velocity.dim(), Pressure.dim(), Magnetic.dim(), Lagrange.dim()]
    b_t = TrialFunction(Velocity)
    c_t = TestFunction(Velocity)
    IterType = "Full"

    def boundary(x, on_boundary):
        return on_boundary

    u0, p0,b0, r0, Laplacian, Advection, gradPres,CurlCurl, gradR, NS_Couple, M_Couple = ExactSol.MHD2D(4,1)

    kappa = 1.0
    Mu_m =1e1
    MU = 1.0/1

    F_NS = -MU*Laplacian+Advection+gradPres-kappa*NS_Couple
    if kappa == 0:
        F_M = Mu_m*CurlCurl+gradR -kappa*M_Couple
    else:
        F_M = Mu_m*kappa*CurlCurl+gradR -kappa*M_Couple
    params = [kappa,Mu_m,MU]


    MO.PrintStr("Preconditioning MHD setup",5,"+","\n\n","\n\n")
    HiptmairMatrices = PrecondSetup.MagneticSetup(Magnetic, Lagrange, b0, r0, 1e-4, params)


    MO.PrintStr("Setting up MHD initial guess",5,"+","\n\n","\n\n")
    u_k,p_k,b_k,r_k = common.InitialGuess([Velocity, Pressure, Magnetic, Lagrange],[u0,p0,b0,r0],[F_NS,F_M],params,HiptmairMatrices,1e-6,options="New")

    KSPlinearfluids, MatrixLinearFluids = PrecondSetup.FluidLinearSetup(Pressure, MU)
    kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k)

    ns,maxwell,CoupleTerm,Lmaxwell,Lns = forms.MHD2D(mesh, W,F_M,F_NS, u_k,b_k,params,IterType)
    RHSform = forms.PicardRHS(mesh, W, u_k, p_k, b_k, r_k, params)


    bcu = DirichletBC(W.sub(0),Expression(("0.0","0.0")), boundary)
    bcb = DirichletBC(W.sub(2),Expression(("0.0","0.0")), boundary)
    bcr = DirichletBC(W.sub(3),Expression(("0.0")), boundary)
    bcs = [bcu,bcb,bcr]

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-4     # tolerance
    iter = 0            # iteration counter
    maxiter = 10       # max no of iterations allowed
    SolutionTime = 0
    outer = 0

    TotalStart =time.time()
    SolutionTime = 0
    while eps > tol  and iter < maxiter:
        iter += 1
        MO.PrintStr("Iter "+str(iter),7,"=","\n\n","\n\n")
        AssembleTime = time.time()
        AA, bb = assemble_system(maxwell+ns+CoupleTerm, (Lmaxwell + Lns) - RHSform,  bcs)
        A,b = CP.Assemble(AA,bb)
        MO.StrTimePrint("MHD total assemble, time: ", time.time()-AssembleTime)

        u = b.duplicate()
        kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k)
        print "Inititial guess norm: ",  u.norm(PETSc.NormType.NORM_INFINITY)
        #A,Q

        n = FacetNormal(mesh)
        mat =  as_matrix([[b_k[1]*b_k[1],-b_k[1]*b_k[0]],[-b_k[1]*b_k[0],b_k[0]*b_k[0]]])
        a = params[2]*inner(grad(b_t), grad(c_t))*dx(W.mesh()) + inner((grad(b_t)*u_k),c_t)*dx(W.mesh()) +(1./2)*div(u_k)*inner(c_t,b_t)*dx(W.mesh()) - (1./2)*inner(u_k,n)*inner(c_t,b_t)*ds(W.mesh())+kappa/Mu_m*inner(mat*b_t,c_t)*dx(W.mesh())
        ShiftedMass = assemble(a)
        bcu.apply(ShiftedMass)
        ShiftedMass = CP.Assemble(ShiftedMass)
        kspF = NSprecondSetup.LSCKSPnonlinear(ShiftedMass)



        Options = 'p4'



        stime = time.time()
        u, mits,nsits = S.solve(A,b,u,params,W,'Directass',IterType,OuterTol,InnerTol,HiptmairMatrices,Hiptmairtol,KSPlinearfluids, Fp,kspF,Options)
        Soltime = time.time()- stime
        MO.StrTimePrint("MHD solve, time: ", Soltime)
        Mits += mits
        NSits += nsits
        SolutionTime += Soltime

        u1, p1, b1, r1, eps= Iter.PicardToleranceDecouple(u,x,FSpaces,dim,"2",iter)
        p1.vector()[:] += - assemble(p1*dx)/assemble(ones*dx)
        u_k.assign(u1)
        p_k.assign(p1)
        b_k.assign(b1)
        r_k.assign(r1)
        uOld= np.concatenate((u_k.vector().array(),p_k.vector().array(),b_k.vector().array(),r_k.vector().array()), axis=0)
        x = IO.arrayToVec(uOld)



    XX= np.concatenate((u_k.vector().array(),p_k.vector().array(),b_k.vector().array(),r_k.vector().array()), axis=0)
    SolTime[xx-1] = SolutionTime/iter
    NSave[xx-1] = (float(NSits)/iter)
    Mave[xx-1] = (float(Mits)/iter)
    iterations[xx-1] = iter
    TotalTime[xx-1] = time.time() - TotalStart
