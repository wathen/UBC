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
#import matplotlib.pylab as plt
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
import MHDallatonce as MHDpreconditioner
import memory_profiler
import gc
import MHDmulti

#@profile
def foo():
    m = 2


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
        level[xx-1] = xx+ 10
        nn = 2**(level[xx-1])



        # Create mesh and define function space
        nn = int(nn)
        NN[xx-1] = nn/2
        # parameters["form_compiler"]["quadrature_degree"] = 6
        # parameters = CP.ParameterSetup()
        mesh = UnitSquareMesh(nn,nn)

        order = 1
        parameters['reorder_dofs_serial'] = False
        Velocity = VectorFunctionSpace(mesh, "CG", order)
        Pressure = FunctionSpace(mesh, "DG", order-1)
        Magnetic = FunctionSpace(mesh, "N1curl", order)
        Lagrange = FunctionSpace(mesh, "CG", order)
        W = MixedFunctionSpace([Velocity,Magnetic, Pressure, Lagrange])
        # W = Velocity*Pressure*Magnetic*Lagrange
        Velocitydim[xx-1] = Velocity.dim()
        Pressuredim[xx-1] = Pressure.dim()
        Magneticdim[xx-1] = Magnetic.dim()
        Lagrangedim[xx-1] = Lagrange.dim()
        Wdim[xx-1] = W.dim()
        print "\n\nW:  ",Wdim[xx-1],"Velocity:  ",Velocitydim[xx-1],"Pressure:  ",Pressuredim[xx-1],"Magnetic:  ",Magneticdim[xx-1],"Lagrange:  ",Lagrangedim[xx-1],"\n\n"
        dim = [Velocity.dim(), Magnetic.dim(), Pressure.dim(), Lagrange.dim()]


        def boundary(x, on_boundary):
            return on_boundary

        u0, p0,b0, r0, Laplacian, Advection, gradPres,CurlCurl, gradR, NS_Couple, M_Couple = ExactSol.MHD2D(4,1)


        bcu = DirichletBC(W.sub(0),u0, boundary)
        bcb = DirichletBC(W.sub(1),b0, boundary)
        bcr = DirichletBC(W.sub(3),r0, boundary)

        # bc = [u0,p0,b0,r0]
        bcs = [bcu,bcb,bcr]
        FSpaces = [Velocity,Pressure,Magnetic,Lagrange]


        (u, b, p, r) = TrialFunctions(W)
        (v, c, q, s) = TestFunctions(W)
        kappa = 1.0
        Mu_m =1e1
        MU = 1.0/1
        IterType = 'Full'

        F_NS = -MU*Laplacian+Advection+gradPres-kappa*NS_Couple
        if kappa == 0:
            F_M = Mu_m*CurlCurl+gradR -kappa*M_Couple
        else:
            F_M = Mu_m*kappa*CurlCurl+gradR -kappa*M_Couple
        params = [kappa,Mu_m,MU]


        # MO.PrintStr("Preconditioning MHD setup",5,"+","\n\n","\n\n")
        HiptmairMatrices = PrecondSetup.MagneticSetup(Magnetic, Lagrange, b0, r0, 1e-5, params)


        MO.PrintStr("Setting up MHD initial guess",5,"+","\n\n","\n\n")
        u_k,p_k,b_k,r_k = common.InitialGuess(FSpaces,[u0,p0,b0,r0],[F_NS,F_M],params,HiptmairMatrices,1e-6,Neumann=Expression(("0","0")),options ="New", FS = "DG")
        #plot(p_k, interactive = True) 
        b_t = TrialFunction(Velocity)
        c_t = TestFunction(Velocity)
        #print assemble(inner(b,c)*dx).array().shape
        #print mat
        #ShiftedMass = assemble(inner(mat*b,c)*dx)
        #as_vector([inner(b,c)[0]*b_k[0],inner(b,c)[1]*(-b_k[1])])

        ones = Function(Pressure)
        ones.vector()[:]=(0*ones.vector().array()+1)
        # pConst = - assemble(p_k*dx)/assemble(ones*dx)
        p_k.vector()[:] += - assemble(p_k*dx)/assemble(ones*dx)
        x = Iter.u_prev(u_k,b_k,p_k,r_k)

        KSPlinearfluids, MatrixLinearFluids = PrecondSetup.FluidLinearSetup(Pressure, MU)
        kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k)
        #plot(b_k)

        ns,maxwell,CoupleTerm,Lmaxwell,Lns = forms.MHD2D(mesh, W,F_M,F_NS, u_k,b_k,params,IterType,"DG", SaddlePoint = "Yes")
        RHSform = forms.PicardRHS(mesh, W, u_k, p_k, b_k, r_k, params,"DG",SaddlePoint = "Yes")


        bcu = DirichletBC(W.sub(0),Expression(("0.0","0.0")), boundary)
        bcb = DirichletBC(W.sub(1),Expression(("0.0","0.0")), boundary)
        bcr = DirichletBC(W.sub(3),Expression(("0.0")), boundary)
        bcs = [bcu,bcb,bcr]

        eps = 1.0           # error measure ||u-u_k||
        tol = 1.0E-4     # tolerance
        iter = 0            # iteration counter
        maxiter = 40       # max no of iterations allowed
        SolutionTime = 0
        outer = 0
        # parameters['linear_algebra_backend'] = 'uBLAS'

        # FSpaces = [Velocity,Magnetic,Pressure,Lagrange]

        if IterType == "CD":
            AA, bb = assemble_system(maxwell+ns, (Lmaxwell + Lns) - RHSform,  bcs)
            A,b = CP.Assemble(AA,bb)
            # u = b.duplicate()
            # P = CP.Assemble(PP)


        u_is = PETSc.IS().createGeneral(range(Velocity.dim()))
        NS_is = PETSc.IS().createGeneral(range(Velocity.dim()+Pressure.dim()))
        M_is = PETSc.IS().createGeneral(range(Velocity.dim()+Pressure.dim(),W.dim()))
        OuterTol = 1e-5
        InnerTol = 1e-3
        NSits =0
        Mits =0
        TotalStart =time.time()
        SolutionTime = 0
        while eps > tol  and iter < maxiter:
            iter += 1
            MO.PrintStr("Iter "+str(iter),7,"=","\n\n","\n\n")
            tic()
            if IterType == "CD":
                bb = assemble((Lmaxwell + Lns) - RHSform)
                for bc in bcs:
                    bc.apply(bb)
                FF = AA.sparray()[0:dim[0],0:dim[0]]
                A,b = CP.Assemble(AA,bb)
                # if iter == 1
                if iter == 1:
                    u = b.duplicate()
                    F = A.getSubMatrix(u_is,u_is)
                    kspF = NSprecondSetup.LSCKSPnonlinear(F)
            else:
                AA, bb = assemble_system(maxwell+ns+CoupleTerm, (Lmaxwell + Lns) - RHSform,  bcs)
                A,b = CP.Assemble(AA,bb)
                del AA, bb
                n = FacetNormal(mesh)
                mat =  as_matrix([[b_k[1]*b_k[1],-b_k[1]*b_k[0]],[-b_k[1]*b_k[0],b_k[0]*b_k[0]]])
                F = A.getSubMatrix(u_is,u_is)
                a = params[2]*inner(grad(b_t), grad(c_t))*dx(W.mesh()) + inner((grad(b_t)*u_k),c_t)*dx(W.mesh()) +(1/2)*div(u_k)*inner(c_t,b_t)*dx(W.mesh()) - (1/2)*inner(u_k,n)*inner(c_t,b_t)*ds(W.mesh())+kappa/Mu_m*inner(mat*b_t,c_t)*dx(W.mesh())
                ShiftedMass = assemble(a)
                bcu.apply(ShiftedMass)

                #MO.StoreMatrix(AA.sparray()[0:dim[0],0:dim[0]]+ShiftedMass.sparray(),"A")
                kspF = NSprecondSetup.LSCKSPnonlinear(F)
            # if iter == 1:
                if iter == 1:
                    u = b.duplicate()
            print ("{:40}").format("MHD assemble, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

            kspFp, Fp = PrecondSetup.FluidNonLinearSetup(Pressure, MU, u_k)
            print "Inititial guess norm: ", u.norm()
            solver = 'Schur'
            if solver == 'Schur':
                FF = CP.Assemble(ShiftedMass)
                kspF = NSprecondSetup.LSCKSPnonlinear(FF)
                ksp = PETSc.KSP()
                ksp.create(comm=PETSc.COMM_WORLD)
                pc = ksp.getPC()
                ksp.setType('fgmres')
                pc.setType('python')
                pc.setType(PETSc.PC.Type.PYTHON)
                # FSpace = [Velocity,Magnetic,Pressure,Lagrange]
                reshist = {}
                def monitor(ksp, its, fgnorm):
                    reshist[its] = fgnorm
                    print its,"    OUTER:", fgnorm
                # ksp.setMonitor(monitor)
                ksp.max_it = 1000
                FFSS = [Velocity,Magnetic,Pressure,Lagrange]
                pc.setPythonContext(MHDpreconditioner.InnerOuterMAGNETICapprox(FFSS,kspF, KSPlinearfluids[0], KSPlinearfluids[1],Fp, HiptmairMatrices[3], HiptmairMatrices[4], HiptmairMatrices[2], HiptmairMatrices[0], HiptmairMatrices[1], HiptmairMatrices[6],1e-5,FF))
                #OptDB = PETSc.Options()

                # OptDB['pc_factor_mat_solver_package']  = "mumps"
                # OptDB['pc_factor_mat_ordering_type']  = "rcm"
                # ksp.setFromOptions()
                scale = b.norm()
                b = b/scale
                ksp.setOperators(A,A)
                del A
                stime = time.time()
                ksp.solve(b,u)
                Soltime = time.time()- stime
                NSits += ksp.its
                Mits += ksp.its
                # Mits +=dodim
                u = u*scale
                SolutionTime = SolutionTime +Soltime
                MO.PrintStr("Number iterations ="+str(ksp.its),60,"+","\n\n","\n\n")
                MO.PrintStr("Time: "+str(Soltime),60,"+","\n\n","\n\n")
            else:
                kspOuter = PETSc.KSP()
                kspOuter.create(comm=PETSc.COMM_WORLD)
                FFSS = [Velocity,Magnetic,Pressure,Lagrange]
                kspOuter.setType('fgmres')
                kspOuter.setOperators(A,A) 
                pcOuter = kspOuter.getPC()
                pcOuter.setType(PETSc.PC.Type.KSP)              
                
                kspInner = pcOuter.getKSP()
                kspInner.setType('gmres')
                pcInner = kspInner.getPC()

                # FSpace = [Velocity,Magnetic,Pressure,Lagrange]
                reshist = {}
                def monitor(ksp, its, fgnorm):
                    reshist[its] = fgnorm
                    print its,"    OUTER:", fgnorm
                
                kspOuter.setMonitor(monitor)
                kspOuter.max_it = 500
                kspInner.max_it = 100
                kspOuter.setTolerances(OuterTol)
                kspInner.setTolerances(InnerTol)
                
                pcInner.setType('python')
                pcInner.setPythonContext(MHDpreconditioner.InnerOuter(A,FFSS,kspF, KSPlinearfluids[0], KSPlinearfluids[1],Fp, HiptmairMatrices[3], HiptmairMatrices[4], HiptmairMatrices[2], HiptmairMatrices[0], HiptmairMatrices[1], HiptmairMatrices[6],1e-4,F))
                
                # OptDB = PETSc.Options()

                PP = PETSc.Mat().create()
                PP.setSizes([A.size[0], A.size[0]])

                #PP = PETSc.Mat().createPython([A.size[0], A.size[0]])
                PP.setType('python')
                pp = MHDmulti.P(FFSS,A,MatrixLinearFluids[1],MatrixLinearFluids[0],kspFp,HiptmairMatrices[6])
                
                PP.setPythonContext(pp)
                PP.setUp()
                kspInner.setOperators(PP,PP)
                kspInner.setFromOptions()
                scale = b.norm()
                b = b/scale
                del A
                stime = time.time()
                kspOuter.solve(b,u)
                Soltime = time.time()- stime
                NSits += kspOuter.its
                Mits += kspInner.its
                u = u*scale
                SolutionTime = SolutionTime +Soltime
                MO.PrintStr("Number of outer iterations ="+str(kspOuter.its),60,"+","\n\n","\n\n")
                MO.PrintStr("Number of inner iterations ="+str(kspInner.its),60,"+","\n\n","\n\n")
            u1, p1, b1, r1, eps= Iter.PicardToleranceDecouple(u,x,FSpaces,dim,"2",iter, SaddlePoint = "Yes")
            p1.vector()[:] += - assemble(p1*dx)/assemble(ones*dx)
            u_k.assign(u1)
            p_k.assign(p1)
            b_k.assign(b1)
            r_k.assign(r1)
            uOld= np.concatenate((u_k.vector().array(),b_k.vector().array(),p_k.vector().array(),r_k.vector().array()), axis=0)
            x = IO.arrayToVec(uOld)



        XX= np.concatenate((u_k.vector().array(),p_k.vector().array(),b_k.vector().array(),r_k.vector().array()), axis=0)
        SolTime[xx-1] = SolutionTime/iter
        NSave[xx-1] = (float(NSits)/iter)
        Mave[xx-1] = (float(Mits)/iter)
        iterations[xx-1] = iter
        TotalTime[xx-1] = time.time() - TotalStart
        dim = [Velocity.dim(), Pressure.dim(), Magnetic.dim(),Lagrange.dim()]
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




    import pandas as pd



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
#
#
#
#
#    import pandas as pd
#



    print "\n\n   Iteration table"
    if IterType == "Full":
        IterTitles = ["l","DoF","AV solve Time","Total picard time","picard iterations","Av Outer its","Av Inner its",]
    else:
        IterTitles = ["l","DoF","AV solve Time","Total picard time","picard iterations","Av NS iters","Av M iters"]
    IterValues = np.concatenate((level,Wdim,SolTime,TotalTime,iterations,NSave,Mave),axis=1)
    IterTable= pd.DataFrame(IterValues, columns = IterTitles)
    if IterType == "Full":
        IterTable = MO.PandasFormat(IterTable,'Av Outer its',"%2.1f")
        IterTable = MO.PandasFormat(IterTable,'Av Inner its',"%2.1f")
    else:
        IterTable = MO.PandasFormat(IterTable,'Av NS iters',"%2.1f")
        IterTable = MO.PandasFormat(IterTable,'Av M iters',"%2.1f")
    print IterTable.to_latex()
    print " \n  Outer Tol:  ",OuterTol, "Inner Tol:   ", InnerTol




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
