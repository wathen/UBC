#!/usr/bin/python

# interpolate scalar gradient onto nedelec space
from dolfin import *

import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
Print = PETSc.Sys.Print
# from MatrixOperations import *
import numpy as np
#import matplotlib.pylab as plt
import PETScIO as IO
import common
import scipy
import scipy.io
import time as t

import BiLinear as forms
import IterOperations as Iter
import MatrixOperations as MO
import CheckPetsc4py as CP
import Solver as S
import ExactSol
import P as Precond
import cProfile, pstats, StringIO

m = 4
# m = 2

IterType = "Full"


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
TotalWork = []
InnerTolerances = []

nn = 2

dim = 2
ShowResultPlots = 'yes'
split = 'Linear'

MU[0]= 1e0
test = [[1e-6,1e-6],[1e-6,1e-5],[1e-6,1e-4],[1e-6,1e-3],[1e-6,1e-6,'Reduce1',2],[1e-6,1e-6,'Reduce1',5],[1e-6,1e-6,'Reduce1',10]]
# test = [[1e-6,1e-6],[1e-5,1e-5],[1e-4,1e-4],[1e-3,1e-3]]
test = [[1e-6,1e-6],[1e-6,1e-4],[1e-6,1e-2],[1e-6,1e-1]]
# test =[[1e-6,1e-1]]
MinTol = 1e-3
aa = 0
for kk in xrange(1,len(test)+1):
    aa += 1
    for xx in xrange(1,m):
        print xx
        level[xx-1] = xx+3
        nn = 2**(level[xx-1])


        # Create mesh and define function space
        nn = int(nn)
        NN[xx-1] = nn/2

        mesh = RectangleMesh(0, 0, 1, 1, nn, nn,'left')
        parameters["form_compiler"]["precision"] = 15
        parameters["form_compiler"]["quadrature_degree"] = -1
        order = 2
        parameters['reorder_dofs_serial'] = False
        Velocity = VectorFunctionSpace(mesh, "CG", order)
        Pressure = FunctionSpace(mesh, "CG", order-1)
        Magnetic = FunctionSpace(mesh, "N1curl", order)
        Lagrange = FunctionSpace(mesh, "CG", order)
        W = MixedFunctionSpace([Velocity,Pressure,Magnetic,Lagrange])
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

        bcu = DirichletBC(W.sub(0),u0, boundary)
        bcp = DirichletBC(W.sub(1),p0, boundary)
        bcb = DirichletBC(W.sub(2),b0, boundary)
        bcr = DirichletBC(W.sub(3),r0, boundary)

        # bc = [u0,p0,b0,r0]
        bcs = [bcu,bcb,bcr]
        FSpaces = [Velocity,Pressure,Magnetic,Lagrange]


        (u, p, b, r) = TrialFunctions(W)
        (v, q, c,s ) = TestFunctions(W)
        kappa = 100.0

        parameters['linear_algebra_backend'] = 'uBLAS'

        Mu_m =1e1
        MU = 1.0


        F_NS = -MU*Laplacian+Advection+gradPres-kappa*NS_Couple
        if kappa == 0:
            F_M = Mu_m*CurlCurl+gradR -kappa*M_Couple
        else:
            F_M = Mu_m*kappa*CurlCurl+gradR -kappa*M_Couple
        params = [kappa,Mu_m,MU]

        F_NS = -MU*Laplacian+Advection+gradPres-kappa*NS_Couple

        F_M = Mu_m*kappa*CurlCurl+gradR -kappa*M_Couple

        params = [kappa,Mu_m,MU]
        u_k,p_k,b_k,r_k = common.InitialGuess(FSpaces,[u0,p0,b0,r0],[F_NS,F_M],params,Neumann=Expression(("0","0")),options ="New")
        p_k.vector()[:]= p_k.vector().array()+np.abs(np.min(p_k.vector().array()))
        # bcu.apply(u_k)
        # bcb.apply(b_k)
        # bcr.apply(r_k)

        x = Iter.u_prev(u_k,p_k,b_k,r_k)

        ns,maxwell,CoupleTerm,Lmaxwell,Lns = forms.MHD2D(mesh, W,F_M,F_NS, u_k,b_k,params,IterType)
        print CoupleTerm



        RHSform = forms.PicardRHS(mesh, W, u_k, p_k, b_k, r_k, params)

        bcu = DirichletBC(W.sub(0),Expression(("0","0")), boundary)
        bcb = DirichletBC(W.sub(2),Expression(("0","0")), boundary)
        bcr = DirichletBC(W.sub(3),Expression(("0")), boundary)
        bcs = [bcu,bcb,bcr]


        p = forms.Preconditioner(mesh,W,u_k,b_k,params,IterType)

        PP,Pb = assemble_system(p, Lns,bcs)
        NS_is = PETSc.IS().createGeneral(range(Velocity.dim()+Pressure.dim()))
        M_is = PETSc.IS().createGeneral(range(Velocity.dim()+Pressure.dim(),W.dim()))
        if IterType == "Full" or IterType == "MD":
            (pQ) = TrialFunction(Pressure)
            (qQ) = TestFunction(Pressure)
            print MU
            Q = assemble(inner(pQ,qQ)*dx)
            L = assemble(inner(grad(pQ),grad(qQ))*dx)
            n = FacetNormal(mesh)
            fp = MU*inner(grad(qQ), grad(pQ))*dx+inner((u_k[0]*grad(pQ)[0]+u_k[1]*grad(pQ)[1]),qQ)*dx + (1/2)*div(u_k)*inner(pQ,qQ)*dx - (1/2)*(u_k[0]*n[0]+u_k[1]*n[1])*inner(pQ,qQ)*ds
            L = CP.Assemble(L)

        if IterType == "CD":
            AA, bb = assemble_system(maxwell+ns+CoupleTerm, (Lmaxwell + Lns) - RHSform,  bcs)
            A,b = CP.Assemble(AA,bb)
            P = CP.Assemble(PP)
            u = b.duplicate()

        Mits = 0
        NSits = 0
        time
        InnerTol = []
        OuterTol = []

        # MPI.barrier()
        eps = 1.0           # error measure ||u-u_k||
        tol = 1.0E-4      # tolerance
        iter = 0            # iteration counter
        maxiter = 50       # max no of iterations allowed
        SolutionTime = 0
        outer = 0
        parameters['linear_algebra_backend'] = 'uBLAS'

        TotalStart = t.clock()
        while eps > tol  and iter < maxiter:
            iter += 1
            if IterType == "CD":
                bb = assemble((Lmaxwell + Lns) - RHSform)
                for bc in bcs:
                    bc.apply(bb)

                A,b = CP.Assemble(AA,bb)

                P = CP.Assemble(PP)
                print b
                Mass = 0
                L = 0
                F = 0

            else:
                # tic()
                AA, bb = assemble_system(maxwell+ns+CoupleTerm, (Lmaxwell + Lns) - RHSform,  bcs)
                A,b = CP.Assemble(AA,bb)
                del AA
                F = assemble(fp)
                F = CP.Assemble(F)
                P = CP.Assemble(PP)
                # P = S.ExactPrecond(PP,Q,L,F,FSpaces)
                Mass = CP.Assemble(Q)
                # print "Assemble time >>>>>>",toc()


            if iter == 1:
                uu = b.duplicate()
            else:
                uu = uu
            pr = cProfile.Profile()
            start = t.clock()
            pr.enable()
            # if len(test[kk-1]) ==2:
            OuterTol = test[kk-1][0]
            InnerTol = test[kk-1][1]
            print InnerTol
            print OuterTol
            u,it1,it2 = S.solve(A,b,uu,P,[NS_is,M_is],FSpaces,IterType,OuterTol,InnerTol,Mass,L,F)
            # else:
            #     if test[kk-1] == "Reduce":
            #         OuterTol = test[kk-1][0]
            #         InnerTol.append(test[kk-1][1]*((iter)*test[kk-1][3]))
            #         print InnerTol
            #         print OuterTol
            #         u,it1,it2 = S.solve(A,b,uu,P,[NS_is,M_is],FSpaces,IterType,OuterTol,InnerTol[iter-1],Mass,L,F)
            #     else:
            #         OuterTol = test[kk-1][0]
            #         if iter  == 1:
            #             InnerTol.append(test[kk-1][1])
            #         else:
            #             print InnerTol[iter-2]
            #             InnerTol.append(min(MinTol,InnerTol[iter-2]*test[kk-1][3]))

            #         print InnerTol
            #         print OuterTol
            #         u,it1,it2 = S.solve(A,b,uu,P,[NS_is,M_is],FSpaces,IterType,OuterTol,InnerTol[iter-1],Mass,L,F)

            del A,P
            # print InnerTol[iter-1]
            pr.disable()
            # time = toc()
            time = (t.clock() - start)
            s = StringIO.StringIO()
            print "Solve time >>>>>>", time
            print it1,it2
            NSits += it1
            Mits +=it2
            SolutionTime = SolutionTime +time
            # tic()
            u, p, b, r, eps= Iter.PicardToleranceDecouple(u,x,FSpaces,dim,"inf",iter)
            u_k.assign(u)
            p_k.assign(p)
            b_k.assign(b)
            r_k.assign(r)
            # print "Correction time >>>>>>", toc()
            # p_k.vector()[:]= p_k.vector().array()+np.abs(np.min(p_k.vector().array()))
            x = Iter.u_prev(u_k,p_k,b_k,r_k)

        print toc()
            # u_k,b_k,epsu,epsb=Direct.PicardTolerance(x,u_k,b_k,FSpaces,dim,"inf",iter)
        TotalTime[xx-1] = t.clock()-TotalStart
        NSave[xx-1] = (float(NSits)/iter)
        Mave[xx-1] = (float(Mits)/iter)
        iterations[xx-1] = iter
        SolTime[xx-1] = SolutionTime/iter

        ue = u0
        pe = p0
        be = b0
        re = r0




        ExactSolution = [ue,pe,be,re]
        #errL2u[xx-1], errH1u[xx-1], errL2p[xx-1], errL2b[xx-1], errCurlb[xx-1], errL2r[xx-1], errH1r[xx-1] = Iter.Errors(x,mesh,FSpaces,ExactSolution,order,dim)

        if xx == 1:
            l2uorder[xx-1] = 0
        else:
            l2uorder[xx-1] =  np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1]))
            H1uorder[xx-1] =  np.abs(np.log2(errH1u[xx-2]/errH1u[xx-1]))

            l2porder[xx-1] =  np.abs(np.log2(errL2p[xx-2]/errL2p[xx-1]))

            l2border[xx-1] =  np.abs(np.log2(errL2b[xx-2]/errL2b[xx-1]))
            Curlborder[xx-1] =  np.abs(np.log2(errCurlb[xx-2]/errCurlb[xx-1]))

            l2rorder[xx-1] =  np.abs(np.log2(errL2r[xx-2]/errL2r[xx-1]))
            H1rorder[xx-1] =  np.abs(np.log2(errH1r[xx-2]/errH1r[xx-1]))




    import pandas as pd


    # print "\n\n   Velocity convergence"
    # VelocityTitles = ["l","Total DoF","V DoF","Soln Time","V-L2","L2-order","V-H1","H1-order"]
    # VelocityValues = np.concatenate((level,Wdim,Velocitydim,SolTime,errL2u,l2uorder,errH1u,H1uorder),axis=1)
    # VelocityTable= pd.DataFrame(VelocityValues, columns = VelocityTitles)
    # pd.set_option('precision',3)
    # VelocityTable = MO.PandasFormat(VelocityTable,"V-L2","%2.4e")
    # VelocityTable = MO.PandasFormat(VelocityTable,'V-H1',"%2.4e")
    # VelocityTable = MO.PandasFormat(VelocityTable,"H1-order","%1.2f")
    # VelocityTable = MO.PandasFormat(VelocityTable,'L2-order',"%1.2f")
    # print VelocityTable.to_latex()

    # print "\n\n   Pressure convergence"
    # PressureTitles = ["l","Total DoF","P DoF","Soln Time","P-L2","L2-order"]
    # PressureValues = np.concatenate((level,Wdim,Pressuredim,SolTime,errL2p,l2porder),axis=1)
    # PressureTable= pd.DataFrame(PressureValues, columns = PressureTitles)
    # pd.set_option('precision',3)
    # PressureTable = MO.PandasFormat(PressureTable,"P-L2","%2.4e")
    # PressureTable = MO.PandasFormat(PressureTable,'L2-order',"%1.2f")
    # print PressureTable.to_latex()



    name = "OutputNew8/table"+str(aa)
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
    IterTable.to_csv(name)
    print " \n  Outer Tol:  ",OuterTol, "Inner Tol:   ", InnerTol

    file = open("OutputNew8/tol"+str(aa)+".txt", "w")
    file.write("Outer:\n"+str(OuterTol)+"\n\n")
    file.write("Inner:\n")
    if len(test[kk-1]) ==2:
        file.write(str(InnerTol))
    else:
        for s in InnerTol:
            file.write(str(s))
    file.write(str(kappa)+","+str(MU))
    print "-------------------------------------------------------\n\n\n\n\n\n\n"


# \begin{tabular}{lrrrrrll}
# \toprule
# {} &  l &    DoF &  AV solve Time &  Total picard time &  picard iterations & Av Outer its & Av Inner its \\
# \midrule
# 0 &  3 &   1620 &       7.568889 &              68.87 &                  9 &        245.2 &          3.4 \\
# 1 &  4 &   6180 &       1.267143 &              10.28 &                  7 &         96.9 &          2.0 \\
# 2 &  5 &  24132 &       5.147143 &              40.59 &                  7 &        107.6 &          2.7 \\
# 3 &  6 &  95364 &      19.560000 &             154.99 &                  7 &         94.4 &          2.1 \\
# \bottomrule
# \end{tabular}


#   Outer Tol:   1e-06 Inner Tol:    0.1


    # # # if (ShowResultPlots == 'yes'):

    # plot(u_k)
    # # plot(interpolate(ue,Velocity))

    # plot(p_k)
    # # pe = interpolate(pe,Pressure)
    # # pe.vector()[:] -= np.max(pe.vector().array() )/2
    # # plot(interpolate(pe,Pressure))

    # plot(b_k)
    # # plot(interpolate(be,Magnetic))

    # plot(r_k)
    # # plot(interpolate(re,Lagrange))

    # # # interactive()



print kappa, MU




    # interactive()
