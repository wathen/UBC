#!/usr/bin/python

# interpolate scalar gradient onto nedelec space

import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
from dolfin import *
Print = PETSc.Sys.Print
# from MatrixOperations import *
import numpy as np
import PETScIO as IO
import common
import scipy
import scipy.io
import time

import BiLinear as forms
import IterOperations as Iter
import MatrixOperations as MO
import CheckPetsc4py as CP
import ExactSol
import Solver as S
import MHDmatrixPrecondSetup as PrecondSetup
import NSprecondSetup
import MHDprec as MHDpreconditioner
import memory_profiler
import gc
import MHDmulti
import MHDmatrixSetup as MHDsetup
#@profile
def foo():
    m = 3


    errL2u =np.zeros((m-1,1))
    errH1u =np.zeros((m-1,1))
    errL2p =np.zeros((m-1,1))
    errL2b =np.zeros((m-1,1))
    errCurlb =np.zeros((m-1,1))
    errL2r =np.zeros((m-1,1))
    errH1r =np.zeros((m-1,1))



    l2uorder =  np.zeros((m-1,1))
    H1uorder =np.zeros((m-1,1))
    l2porder =  np.zeros((m-1,1))
    l2border =  np.zeros((m-1,1))
    Curlborder =np.zeros((m-1,1))
    l2rorder =  np.zeros((m-1,1))
    H1rorder = np.zeros((m-1,1))

    NN = np.zeros((m-1,1))
    DoF = np.zeros((m-1,1))
    Velocitydim = np.zeros((m-1,1))
    Magneticdim = np.zeros((m-1,1))
    Pressuredim = np.zeros((m-1,1))
    Lagrangedim = np.zeros((m-1,1))
    Wdim = np.zeros((m-1,1))
    iterations = np.zeros((m-1,1))
    SolTime = np.zeros((m-1,1))
    udiv = np.zeros((m-1,1))
    MU = np.zeros((m-1,1))
    level = np.zeros((m-1,1))
    NSave = np.zeros((m-1,1))
    Mave = np.zeros((m-1,1))
    TotalTime = np.zeros((m-1,1))

    nn = 2

    dim = 2
    ShowResultPlots = 'yes'
    split = 'Linear'

    MU[0]= 1e0
    for xx in xrange(1,m):
        print xx
        level[xx-1] = xx+ 8
        nn = 2**(level[xx-1])



        # Create mesh and define function space
        nn = int(nn)
        NN[xx-1] = nn/2
        # parameters["form_compiler"]["quadrature_degree"] = 6
        # parameters = CP.ParameterSetup()
        mesh = UnitSquareMesh(nn,nn)

        order = 2
        parameters['reorder_dofs_serial'] = False
        Velocity = VectorFunctionSpace(mesh, "CG", order)
        Pressure = FunctionSpace(mesh, "CG", order-1)
        Magnetic = FunctionSpace(mesh, "N1curl", order)
        Lagrange = FunctionSpace(mesh, "CG", order)
        W = MixedFunctionSpace([Velocity, Pressure, Magnetic,Lagrange])
        # W = Velocity*Pressure*Magnetic*Lagrange
        Velocitydim[xx-1] = Velocity.dim()
        Pressuredim[xx-1] = Pressure.dim()
        Magneticdim[xx-1] = Magnetic.dim()
        Lagrangedim[xx-1] = Lagrange.dim()
        Wdim[xx-1] = W.dim()
        print "\n\nW:  ",Wdim[xx-1],"Velocity:  ",Velocitydim[xx-1],"Pressure:  ",Pressuredim[xx-1],"Magnetic:  ",Magneticdim[xx-1],"Lagrange:  ",Lagrangedim[xx-1],"\n\n"
        dim = [Velocity.dim(), Pressure.dim(), Magnetic.dim(), Lagrange.dim()]


        def boundary(x, on_boundary):
            return on_boundary

        u0, p0,b0, r0, Laplacian, Advection, gradPres,CurlCurl, gradR, NS_Couple, M_Couple = ExactSol.MHD2D(4,1)


        bcu = DirichletBC(Velocity,u0, boundary)
        bcb = DirichletBC(Magnetic,b0, boundary)
        bcr = DirichletBC(Lagrange,r0, boundary)

        # bc = [u0,p0,b0,r0]
        bcs = [bcu,bcb,bcr]
        FSpaces = [Velocity,Pressure,Magnetic,Lagrange]


        (u, b, p, r) = TrialFunctions(W)
        (v, c, q, s) = TestFunctions(W)
        kappa = 10.0
        Mu_m =10.0
        MU = 1.0/1
        IterType = 'Full'
        Split = "No"
        Saddle = "No"
        Stokes = "No"
        SetupType = 'python-class'
        F_NS = -MU*Laplacian+Advection+gradPres-kappa*NS_Couple
        if kappa == 0:
            F_M = Mu_m*CurlCurl+gradR -kappa*M_Couple
        else:
            F_M = Mu_m*kappa*CurlCurl+gradR -kappa*M_Couple
        params = [kappa,Mu_m,MU]

        MO.PrintStr("Seting up initial guess matricies",2,"=","\n\n","\n")
        BCtime = time.time()
        BC = MHDsetup.BoundaryIndices(mesh)
        MO.StrTimePrint("BC index function, time: ", time.time()-BCtime)
        Hiptmairtol = 1e-5
        HiptmairMatrices = PrecondSetup.MagneticSetup(Magnetic, Lagrange, b0, r0, Hiptmairtol, params)


        MO.PrintStr("Setting up MHD initial guess",5,"+","\n\n","\n\n")
        u_k,p_k,b_k,r_k = common.InitialGuess(FSpaces,[u0,p0,b0,r0],[F_NS,F_M],params,HiptmairMatrices,1e-10,Neumann=Expression(("0","0")),options ="New")
        b_t = TrialFunction(Velocity)
        c_t = TestFunction(Velocity)

        ones = Function(Pressure)
        ones.vector()[:]=(0*ones.vector().array()+1)
        # pConst = - assemble(p_k*dx)/assemble(ones*dx)
        p_k.vector()[:] += - assemble(p_k*dx)/assemble(ones*dx)
        x = Iter.u_prev(u_k,p_k,b_k,r_k)

        KSPlinearfluids, MatrixLinearFluids = PrecondSetup.FluidLinearSetup(Pressure, MU)
        kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k)
        #plot(b_k)

        ns,maxwell,CoupleTerm,Lmaxwell,Lns = forms.MHD2D(mesh, W,F_M,F_NS, u_k,b_k,params,IterType,"CG",Saddle,Stokes)
        RHSform = forms.PicardRHS(mesh, W, u_k, p_k, b_k, r_k, params,"CG",Saddle,Stokes)

        bcu = DirichletBC(W.sub(0),Expression(("0.0","0.0")), boundary)
        bcb = DirichletBC(W.sub(2),Expression(("0.0","0.0")), boundary)
        bcr = DirichletBC(W.sub(3),Expression(("0.0")), boundary)
        bcs = [bcu,bcb,bcr]

        parameters['linear_algebra_backend'] = 'uBLAS'

        eps = 1.0           # error measure ||u-u_k||
        tol = 1.0E-4     # tolerance
        iter = 0            # iteration counter
        maxiter = 10       # max no of iterations allowed
        SolutionTime = 0
        outer = 0
        # parameters['linear_algebra_backend'] = 'uBLAS'

        # FSpaces = [Velocity,Magnetic,Pressure,Lagrange]

        if IterType == "CD":
            MO.PrintStr("Setting up PETSc "+SetupType,2,"=","\n","\n")
            Alin = MHDsetup.Assemble(W,ns,maxwell,CoupleTerm,Lns,Lmaxwell,RHSform,bcs+BC, "Linear",IterType)
            Fnlin,b = MHDsetup.Assemble(W,ns,maxwell,CoupleTerm,Lns,Lmaxwell,RHSform,bcs+BC, "NonLinear",IterType)
            A = Fnlin+Alin
            A,b = MHDsetup.SystemAssemble(FSpaces,A,b,SetupType,IterType)
            u = b.duplicate()


        u_is = PETSc.IS().createGeneral(range(Velocity.dim()))
        NS_is = PETSc.IS().createGeneral(range(Velocity.dim()+Pressure.dim()))
        M_is = PETSc.IS().createGeneral(range(Velocity.dim()+Pressure.dim(),W.dim()))
        OuterTol = 1e-5
        InnerTol = 1e-5
        NSits =0
        Mits =0
        TotalStart =time.time()
        SolutionTime = 0
        while eps > tol  and iter < maxiter:
            iter += 1
            MO.PrintStr("Iter "+str(iter),7,"=","\n\n","\n\n")
            AssembleTime = time.time()
            if IterType == "CD":
                MO.StrTimePrint("MHD CD RHS assemble, time: ", time.time()-AssembleTime)
                b = MHDsetup.Assemble(W,ns,maxwell,CoupleTerm,Lns,Lmaxwell,RHSform,bcs+BC, "CD",IterType)
            else:
                MO.PrintStr("Setting up PETSc "+SetupType,2,"=","\n","\n")
                if  Split == "Yes":
                    if iter == 1:
                        Alin = MHDsetup.Assemble(W,ns,maxwell,CoupleTerm,Lns,Lmaxwell,RHSform,bcs+BC, "Linear",IterType)
                        Fnlin,b = MHDsetup.Assemble(W,ns,maxwell,CoupleTerm,Lns,Lmaxwell,RHSform,bcs+BC, "NonLinear",IterType)
                        A = Fnlin+Alin
                        A,b = MHDsetup.SystemAssemble(FSpaces,A,b,SetupType,IterType)
                        u = b.duplicate()
                    else:
                        Fnline,b = MHDsetup.Assemble(W,ns,maxwell,CoupleTerm,Lns,Lmaxwell,RHSform,bcs+BC, "NonLinear",IterType)
                        A = Fnlin+Alin
                        A,b = MHDsetup.SystemAssemble(FSpaces,A,b,SetupType,IterType)
                else:
                    AA, bb = assemble_system(maxwell+ns+CoupleTerm, (Lmaxwell + Lns) - RHSform,  bcs)
                    A,b = CP.Assemble(AA,bb)
            # if iter == 1:
            MO.StrTimePrint("MHD total assemble, time: ", time.time()-AssembleTime)

            u = b.duplicate()
            kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k)
            print "Inititial guess norm: ",  u.norm(PETSc.NormType.NORM_INFINITY)
            #A,Q
            if IterType == 'Full':

                n = FacetNormal(mesh)
                mat =  as_matrix([[b_k[1]*b_k[1],-b_k[1]*b_k[0]],[-b_k[1]*b_k[0],b_k[0]*b_k[0]]])
                a = params[2]*inner(grad(b_t), grad(c_t))*dx(W.mesh()) + inner((grad(b_t)*u_k),c_t)*dx(W.mesh()) +(1./2)*div(u_k)*inner(c_t,b_t)*dx(W.mesh()) - (1./2)*inner(u_k,n)*inner(c_t,b_t)*ds(W.mesh())+kappa/Mu_m*inner(mat*b_t,c_t)*dx(W.mesh())
                ShiftedMass = assemble(a)
                bcu.apply(ShiftedMass)
                ShiftedMass = CP.Assemble(ShiftedMass)
                kspF = NSprecondSetup.LSCKSPnonlinear(ShiftedMass)
            else:
                F = A.getSubMatrix(u_is,u_is)
                kspF = NSprecondSetup.LSCKSPnonlinear(F)
            stime = time.time()
            u, mits,nsits = S.solve(A,b,u,params,W,'Directclase',IterType,OuterTol,InnerTol,HiptmairMatrices,Hiptmairtol,KSPlinearfluids, Fp,kspF)
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
#        dim = [Velocity.dim(), Pressure.dim(), Magnetic.dim(),Lagrange.dim()]
#
#        ExactSolution = [u0,p0,b0,r0]
#        errL2u[xx-1], errH1u[xx-1], errL2p[xx-1], errL2b[xx-1], errCurlb[xx-1], errL2r[xx-1], errH1r[xx-1] = Iter.Errors(XX,mesh,FSpaces,ExactSolution,order,dim, "DG")
#
#        if xx > 1:
#            l2uorder[xx-1] =  np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1]))
#            H1uorder[xx-1] =  np.abs(np.log2(errH1u[xx-2]/errH1u[xx-1]))
#
#            l2porder[xx-1] =  np.abs(np.log2(errL2p[xx-2]/errL2p[xx-1]))
#
#            l2border[xx-1] =  np.abs(np.log2(errL2b[xx-2]/errL2b[xx-1]))
#            Curlborder[xx-1] =  np.abs(np.log2(errCurlb[xx-2]/errCurlb[xx-1]))
#
#            l2rorder[xx-1] =  np.abs(np.log2(errL2r[xx-2]/errL2r[xx-1]))
#            H1rorder[xx-1] =  np.abs(np.log2(errH1r[xx-2]/errH1r[xx-1]))
#
#
#
#
#    import pandas as pd
#
#
#
#    LatexTitles = ["l","DoFu","Dofp","V-L2","L2-order","V-H1","H1-order","P-L2","PL2-order"]
#    LatexValues = np.concatenate((level,Velocitydim,Pressuredim,errL2u,l2uorder,errH1u,H1uorder,errL2p,l2porder), axis=1)
#    LatexTable = pd.DataFrame(LatexValues, columns = LatexTitles)
#    pd.set_option('precision',3)
#    LatexTable = MO.PandasFormat(LatexTable,"V-L2","%2.4e")
#    LatexTable = MO.PandasFormat(LatexTable,'V-H1',"%2.4e")
#    LatexTable = MO.PandasFormat(LatexTable,"H1-order","%1.2f")
#    LatexTable = MO.PandasFormat(LatexTable,'L2-order',"%1.2f")
#    LatexTable = MO.PandasFormat(LatexTable,"P-L2","%2.4e")
#    LatexTable = MO.PandasFormat(LatexTable,'PL2-order',"%1.2f")
#    print LatexTable
#
#
#    print "\n\n   Magnetic convergence"
#    MagneticTitles = ["l","B DoF","R DoF","B-L2","L2-order","B-Curl","HCurl-order"]
#    MagneticValues = np.concatenate((level,Magneticdim,Lagrangedim,errL2b,l2border,errCurlb,Curlborder),axis=1)
#    MagneticTable= pd.DataFrame(MagneticValues, columns = MagneticTitles)
#    pd.set_option('precision',3)
#    MagneticTable = MO.PandasFormat(MagneticTable,"B-Curl","%2.4e")
#    MagneticTable = MO.PandasFormat(MagneticTable,'B-L2',"%2.4e")
#    MagneticTable = MO.PandasFormat(MagneticTable,"L2-order","%1.2f")
#    MagneticTable = MO.PandasFormat(MagneticTable,'HCurl-order',"%1.2f")
#    print MagneticTable




    import pandas as pd




    print "\n\n   Iteration table"
    if IterType == "Full":
        IterTitles = ["l","DoF","AV solve Time","Total picard time","picard iterations","Av Outer its","Av Inner its",]
    else:
        IterTitles = ["l","DoF","AV solve Time","Total picard time","picard iterations","Av NS iters","Av M iters"]
    IterValues = np.concatenate((level,Wdim,SolTime,TotalTime,iterations,Mave,NSave),axis=1)
    IterTable= pd.DataFrame(IterValues, columns = IterTitles)
    if IterType == "Full":
        IterTable = MO.PandasFormat(IterTable,'Av Outer its',"%2.1f")
        IterTable = MO.PandasFormat(IterTable,'Av Inner its',"%2.1f")
    else:
        IterTable = MO.PandasFormat(IterTable,'Av NS iters',"%2.1f")
        IterTable = MO.PandasFormat(IterTable,'Av M iters',"%2.1f")
    print IterTable
    print " \n  Outer Tol:  ",OuterTol, "Inner Tol:   ", InnerTol

    tableName = "2d_nu="+str(MU)+"_nu_m="+str(Mu_m)+"_kappa="+str(kappa)+".tex"
    IterTable.to_latex(tableName)

    # # # if (ShowResultPlots == 'yes'):

#    plot(u_k)
#    plot(interpolate(u0,Velocity))
#
#    plot(p_k)
#
#    plot(interpolate(p0,Pressure))
#
#    plot(b_k)
#    plot(interpolate(b0,Magnetic))
#
#    plot(r_k)
#    plot(interpolate(r0,Lagrange))
#
#    interactive()

    interactive()
foo()
