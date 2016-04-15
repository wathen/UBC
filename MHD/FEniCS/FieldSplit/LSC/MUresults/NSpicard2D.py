
#!/opt/local/bin/python

from dolfin import *
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

# from MatrixOperations import *
import numpy as np
import matplotlib.pylab as plt
import os
import scipy.io
#from PyTrilinos import Epetra, EpetraExt, AztecOO, ML, Amesos
#from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
import PETScIO as IO
import time
import common
import CheckPetsc4py as CP
import NSprecond
from scipy.sparse import  spdiags
import MatrixOperations as MO
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
#MO.SwapBackend('epetra')
#os.system("echo $PATH")
m = 6
errL2u =np.zeros((m-1,1))
errH1u =np.zeros((m-1,1))
errL2p =np.zeros((m-1,1))

l2uorder =  np.zeros((m-1,1))
H1uorder =np.zeros((m-1,1))
l2porder =  np.zeros((m-1,1))
NN = np.zeros((m-1,1))
DoF = np.zeros((m-1,1))
Vdim = np.zeros((m-1,1))
Qdim = np.zeros((m-1,1))
Wdim = np.zeros((m-1,1))
l2uorder = np.zeros((m-1,1))
l2porder = np.zeros((m-1,1))
nonlinear = np.zeros((m-1,1))
SolTime = np.zeros((m-1,1))
AvIt = np.zeros((m-1,1))
level = np.zeros((m-1,1))
import ExactSol
nn = 2

dim = 2
Solver = 'PCD'
Saving = 'no'
case = 4
# parameters['linear_algebra_backend'] = 'uBLAS'
# parameters = CP.ParameterSetup()
def LOG(arg):
    if INFO:
        print(arg)

mm =4
MUsave = np.zeros((mm,1))
MUit = np.zeros((m-1,mm))

R = 100.0
# MU = Constant(0.01)
for yy in xrange(1,mm+1):

    MU =(R*10**(-yy))
    print MU
    MUsave[yy-1] = MU
    for xx in xrange(1,m):
        level[xx-1] = xx +4
        nn = 2**(level[xx-1])
        # Create mesh and define function space
        nn = int(nn)


    # mesh = RectangleMesh(0, 0, 1, 1, nn, nn,'right')
        mesh = UnitSquareMesh(nn,nn)
        # tic()
        parameters["form_compiler"]["quadrature_degree"] = -1

        parameters['reorder_dofs_serial'] = False
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        ones = Function(Q)
        ones.vector()[:]=(0*ones.vector().array()+1)
        # QQ = VectorFunctionSpace(mesh,"B",3)
        # V = V+QQ
        parameters['reorder_dofs_serial'] = False
        # print 'time to create function spaces', toc(),'\n\n'
        W = V*Q
        Vdim[xx-1] = V.dim()
        Qdim[xx-1] = Q.dim()
        Wdim[xx-1] = W.dim()
        print "\n\nV:  ",Vdim[xx-1],"Q:  ",Qdim[xx-1],"W:  ",Wdim[xx-1],"\n\n"

        def boundary(x, on_boundary):
            return on_boundary

        u0, p0, Laplacian, Advection, gradPres = ExactSol.NS2D(case)
        # plot(interpolate(p0,Q))
        # R = 10.0
        # MU = Constant(0.01)
        # MU = 1000.0
        # MU = 1.0
        bcc = DirichletBC(W.sub(0),u0, boundary)
        bcs = [bcc]

        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)

        # if case == 1:
        #     Laplacian = -MU*Expression(("0","0"))
        #     Advection = Expression(("pow(exp(x[0]),2)","0"))
        #     gradPres = Expression(("cos(x[1])*cos(x[0])","-sin(x[1])*sin(x[0])"))
        #     Laplacian = -MU*Expression(("6*x[1]","6*x[0]"))
        #     # Advection = Expression(("pow(exp(x[0]),2)","0"))
        #     Advection = Expression(("3*pow(x[0],3)*pow(x[1],2)","3*pow(x[1],3)*pow(x[0],2)"))
        #     # gradPres = Expression(("cos(x[1])*cos(x[0])","-sin(x[1])*sin(x[0])"))
        #     gradPres = Expression(("2*x[0]","0"))
        #     f = Laplacian+Advection+gradPres
        #     # f = Expression(("120*x[0]*x[1]*(1-mu)+ 400*x[0]*pow(x[1],6)+(5*pow(x[0],4)-5*pow(x[1],4))*60*x[0]*x[1]*x[1]","60*(pow(x[0],2)-pow(x[1],2))*(1-mu)+400*pow(x[0],4)*pow(x[1],3)-(5*pow(x[0],4)-5*pow(x[1],4))*20*x[1]*x[1]*x[1]"), mu = MU)
        # elif case == 2:
        #     Laplacian = -MU*Expression(("2*(-sin(x[1]) + cos(x[1]))*exp(x[0] + x[1]) + (sin(x[1]) + cos(x[1]))*exp(x[0] + x[1])","-exp(x[0] + x[1])*sin(x[1]) - 2*exp(x[0] + x[1])*cos(x[1])"))
        #     Advection = Expression((" pow((exp(x[0] + x[1])*sin(x[1]) + exp(x[0] + x[1])*cos(x[1])),2) - 2*exp(2*x[0] + 2*x[1])*sin(x[1])*cos(x[1])","-(-exp(x[0] + x[1])*sin(x[1]) - exp(x[0] + x[1])*cos(x[1]))*exp(x[0] + x[1])*sin(x[1]) - (exp(x[0] + x[1])*sin(x[1]) + exp(x[0] + x[1])*cos(x[1]))*exp(x[0] + x[1])*sin(x[1])"))
        #     gradPres = Expression(("3*pow(x[0],2)*sin(x[1]) + exp(x[0] + x[1])","pow(x[0],3)*cos(x[1]) + exp(x[0] + x[1])"))
        f = -MU*Laplacian+Advection+gradPres


        n = FacetNormal(mesh)
        h = CellSize(mesh)
        h_avg =avg(h)
        d = 0
        u_k,p_k = common.Stokes(V,Q,u0,f,[1,1,MU])
        # p_k.vector()[:] = p_k.vector().array()
        pConst = - assemble(p_k*dx)/assemble(ones*dx)
        p_k.vector()[:] += pConst
        # u_k = Function(V)
        # p_k = Function(Q)
        # plot(u_k)
        # plot(p_k)
        uOld = np.concatenate((u_k.vector().array(),p_k.vector().array()), axis=0)
        r = IO.arrayToVec(uOld)

        a11 = MU*inner(grad(v), grad(u))*dx + inner((grad(u)*u_k),v)*dx + (1/2)*div(u_k)*inner(u,v)*dx - (1/2)*inner(u_k,n)*inner(u,v)*ds
        a12 = div(v)*p*dx
        a21 = div(u)*q*dx
        L1  = inner(v, f)*dx
        a = a11-a12-a21


        r11 = MU*inner(grad(v), grad(u_k))*dx + inner((grad(u_k)*u_k),v)*dx + (1/2)*div(u_k)*inner(u_k,v)*dx - (1/2)*inner(u_k,n)*inner(u_k,v)*ds
        r12 = div(v)*p_k*dx
        r21 = div(u_k)*q*dx
        RHSform = r11-r12-r21
        # RHSform = 0

        p11 = inner(u,v)*dx
        # p12 = div(v)*p*dx
        # p21 = div(u)*q*dx
        p22 = inner(p,q)*dx
        prec = p11 +p22
        bc = DirichletBC(W.sub(0),Expression(("0.0","0.0")), boundary)
        bcs = [bc]

        eps = 1.0           # error measure ||u-u_k||
        tol = 1.0E-5       # tolerance
        iter = 0            # iteration counter
        maxiter = 100        # max no of iterations allowed
        # parameters = CP.ParameterSetup()



        outerit = 0

        if Solver == "LSC":
            parameters['linear_algebra_backend'] = 'uBLAS'
            BQB = assemble(inner(u,v)*dx- div(v)*p*dx-div(u)*q*dx)
            bc.apply(BQB)
            BQB = BQB.sparray()
            X = BQB[0:V.dim(),0:V.dim()]
            Xdiag = X.diagonal()
            # Xdiag = X.sum(1).A
            # print Xdiag
            B = BQB[V.dim():W.dim(),0:V.dim()]
            Bt = BQB[0:V.dim(),V.dim():W.dim()]
            d = spdiags(1.0/Xdiag, 0, len(Xdiag), len(Xdiag))
            L = B*d*Bt
            Bd = B*d
            dBt = d*Bt
            L = PETSc.Mat().createAIJ(size=L.shape,csr=(L.indptr, L.indices, L.data))
            Bd = PETSc.Mat().createAIJ(size=Bd.shape,csr=(Bd.indptr, Bd.indices, Bd.data))
            dBt = PETSc.Mat().createAIJ(size=dBt.shape,csr=(dBt.indptr, dBt.indices, dBt.data))
            parameters['linear_algebra_backend'] = 'PETSc'
        elif Solver == "PCD":
            (pQ) = TrialFunction(Q)
            (qQ) = TestFunction(Q)
            print MU
            Mass = assemble(inner(pQ,qQ)*dx)
            L = assemble(inner(grad(pQ),grad(qQ))*dx)

            fp = MU*inner(grad(qQ), grad(pQ))*dx+inner((u_k[0]*grad(pQ)[0]+u_k[1]*grad(pQ)[1]),qQ)*dx + (1/2)*div(u_k)*inner(pQ,qQ)*dx - (1/2)*(u_k[0]*n[0]+u_k[1]*n[1])*inner(pQ,qQ)*ds
            # print "hi"
            L = CP.Assemble(L)
            Mass = CP.Assemble(Mass)

        print L
        SolutionTime = 0
        while eps > tol and iter < maxiter:
            iter += 1
            x = Function(W)

            uu = Function(W)
            tic()
            AA, bb = assemble_system(a, L1-RHSform, bcs)
            A,b = CP.Assemble(AA,bb)
            print toc()
            print A
            # b = b.getSubVector(t_is)
            PP = assemble(prec)
            bcc.apply(PP)
            P = CP.Assemble(PP)

            b = bb.array()
            zeros = 0*b
            bb = IO.arrayToVec(b)
            scale = bb.norm()
            bb = bb/scale
            x = IO.arrayToVec(zeros)
            ksp = PETSc.KSP()
            ksp.create(comm=PETSc.COMM_WORLD)
            ksp.setTolerances(1e-5)
            ksp.setType('gmres')
            pc = ksp.getPC()
            ksp.max_it=200


            pc.setType(PETSc.PC.Type.PYTHON)
            if Solver == "LSC":
                pc.setPythonContext(NSprecond.LSCnew(W,A,L,Bd,dBt))
            elif Solver == "PCD":
                F = assemble(fp)
                F = CP.Assemble(F)
                pc.setPythonContext(NSprecond.PCDdirect(W, Mass, F, L))

            ksp.setOperators(A)
            # OptDB = PETSc.Options()
            # OptDB['pc_factor_shift_amount'] = 1
            # OptDB['pc_factor_mat_ordering_type'] = 'rcm'
            # OptDB['pc_factor_mat_solver_package']  = 'mumps'
            ksp.setFromOptions()



            toc()
            ksp.solve(bb, x)

            time = toc()
            x = x*scale
            print time
            SolutionTime = SolutionTime +time
            print ksp.its
            outerit += ksp.its

            # r = bb.duplicate()
            # A.MUlt(x, r)
            # r.aypx(-1, bb)
            # rnorm = r.norm()
            # PETSc.Sys.Print('error norm = %g' % rnorm,comm=PETSc.COMM_WORLD)

            uu = IO.vecToArray(x)
            UU = uu[0:Vdim[xx-1][0]]
            # time = time+toc()
            u1 = Function(V)
            u1.vector()[:] = u1.vector()[:] + UU

            pp = uu[Vdim[xx-1][0]:]
            # time = time+toc()
            p1 = Function(Q)
            n = pp.shape

            p1.vector()[:] = p1.vector()[:] +  pp
            diff = u1.vector().array()
            eps = np.linalg.norm(diff, ord=np.Inf)

            print '\n\n\niter=%d: norm=%g' % (iter, eps)
            print np.linalg.norm(p1.vector().array(),ord=np.inf)

            u2 = Function(V)
            u2.vector()[:] = u1.vector().array() + u_k.vector().array()
            p2 = Function(Q)
            p2.vector()[:] = p1.vector().array() + p_k.vector().array()
            u_k.assign(u2)
            p_k.assign(p2)

            uOld = np.concatenate((u_k.vector().array(),p_k.vector().array()), axis=0)
            r = IO.arrayToVec(uOld)
            if eps > 1e2 and iter>5:
                iter = 10000000000000
                break

            # del A,AA,PP,prec,L
        SolTime[xx-1] = SolutionTime/iter
        AvIt[xx-1] = np.ceil(outerit/iter)
        MUit[xx-1,yy-1] = AvIt[xx-1]
        # plot(u_k)


    # del  solver

print MUit
print MUsave

# print nonlinear



# print "Velocity Elements rate of convergence ", np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1])))
# print "Pressure Elements rate of convergence ", np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1])))


import pandas as pd
# # tableTitles = ["Total DoF","V DoF","Q DoF","AvIt","V-L2","V-order","P-L2","P-order"]
# # tableValues = np.concatenate((Wdim,Vdim,Qdim,AvIt,errL2u,l2uorder,errL2p,l2porder),axis=1)
# # df = pd.DataFrame(tableValues, columns = tableTitles)
# # pd.set_option('precision',3)
# # print df
# # print df.to_latex()

# # print "\n\n   Velocity convergence"
# # VelocityTitles = ["Total DoF","V DoF","Soln Time","AvIt","V-L2","L2-order","V-H1","H1-order"]
# # VelocityValues = np.concatenate((Wdim,Vdim,SolTime,AvIt,errL2u,l2uorder,errH1u,H1uorder),axis=1)
# # VelocityTable= pd.DataFrame(VelocityValues, columns = VelocityTitles)
# # pd.set_option('precision',3)
# # VelocityTable = MO.PandasFormat(VelocityTable,"V-L2","%2.4e")
# # VelocityTable = MO.PandasFormat(VelocityTable,'V-H1',"%2.4e")
# # VelocityTable = MO.PandasFormat(VelocityTable,"H1-order","%1.2f")
# # VelocityTable = MO.PandasFormat(VelocityTable,'L2-order',"%1.2f")
# # print VelocityTable

# print "\n\n   Pressure convergence"
# PressureTitles = ["Total DoF","P DoF","Soln Time","AvIt","P-L2","L2-order"]
# PressureValues = np.concatenate((Wdim,Qdim,SolTime,AvIt,errL2p,l2porder),axis=1)
# PressureTable= pd.DataFrame(PressureValues, columns = PressureTitles)
# pd.set_option('precision',3)
# PressureTable = MO.PandasFormat(PressureTable,"P-L2","%2.4e")
# PressureTable = MO.PandasFormat(PressureTable,'L2-order',"%1.2f")
# print PressureTable


# LatexTitles = ["l","DoFu","Dofp","AVit"]
# LatexValues = np.concatenate((level,Vdim,Qdim,MUit), axis=1)
# title = np.concatenate((np.array([[0,0,0]]),MUsave.T),axis=1)
# LatexValues = np.vstack((title,LatexValues))
# LatexTable = pd.DataFrame(LatexValues, columns = LatexTitles)

# print LatexTable.to_latex()
LatexTitles = ["l","DoFV","DoFP"]
for x in xrange(1,mm+1):
    LatexTitles.extend(["it"])
pd.set_option('precision',3)
LatexValues = np.concatenate((level,Vdim,Qdim,MUit), axis=1)
title = np.concatenate((np.array([[0,0,0]]),MUsave.T),axis=1)
MU = ["0","0","0"]
for x in xrange(1,mm+1):
    MU.extend(["Full"])
LatexValues = np.vstack((title,LatexValues))
LatexTable = pd.DataFrame(LatexValues, columns = LatexTitles)
# name = "Output/"+IterType+"kappatest"
# LatexTable.to_csv(name)
print LatexTable.to_latex()

# print MUsave

# plot(u_k)
# plot(interpolate(u0,V))

# plot(p_k)
# plot(interpolate(p0,Q))

interactive()

# plt.show()

