
#!/opt/local/bin/python

from dolfin import *
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

# from MatrixOperations import *
import numpy as np
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
import ExactSol
import Solver as S
import matplotlib.pylab as plt
import scipy.sparse as sp

m =4
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
nn = 2

dim = 2
Solver = 'PCD'
Saving = 'no'
case = 4
parameters['linear_algebra_backend'] = 'uBLAS'
# parameters = CP.ParameterSetup()
def LOG(arg):
    if INFO:
        print(arg)




for xx in xrange(1,m):
    print xx
    NN[xx-1] = xx
    nn = 2**(NN[xx-1])
    # Create mesh and define function space
    nn = int(nn)


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

    u_is = PETSc.IS().createGeneral(range(W.dim()-1))
    def boundary(x, on_boundary):
        return on_boundary


    u0, p0, Laplacian, Advection, gradPres = ExactSol.NS2D(case)

    R = 10.0
    # MU = Constant(0.01)
    # MU = 1000.0
    MU = 10./1
    bcc = DirichletBC(W.sub(0),u0, boundary)
    bcs = [bcc]

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)




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
    func = Function(V)
    func.vector()[:] += 1
    f = inner(func,v)*dx

    r11 = MU*inner(grad(v), grad(u_k))*dx + inner((grad(u_k)*u_k),v)*dx + (1/2)*div(u_k)*inner(u_k,v)*dx - (1/2)*inner(u_k,n)*inner(u_k,v)*ds
    r12 = div(v)*p_k*dx
    r21 = div(u_k)*q*dx
    RHSform = r11-r12-r21
    # RHSform = 0


    prec = a11 -a12

    bc = DirichletBC(W.sub(0),Expression(("0.0","0.0")), boundary)
    # bcp = DirichletBC(W.sub(1),Expression(("0.0")), boundary)
    bcs = [bc]

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-6       # tolerance
    iter = 0            # iteration counter
    maxiter = 10        # max no of iterations allowed
    # parameters = CP.ParameterSetup()
    outerit = 0

    parameters['linear_algebra_backend'] = 'uBLAS'
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
    elif Solver == "PCD":
        (pQ) = TrialFunction(Q)
        (qQ) = TestFunction(Q)
        print MU
        QQ = assemble(inner(pQ,qQ)*dx)
        L = assemble(inner(grad(pQ),grad(qQ))*dx)

        fp = MU*inner(grad(qQ), grad(pQ))*dx+inner((u_k[0]*grad(pQ)[0]+u_k[1]*grad(pQ)[1]),qQ)*dx + (1/2)*div(u_k)*inner(pQ,qQ)*dx - (1/2)*(u_k[0]*n[0]+u_k[1]*n[1])*inner(pQ,qQ)*ds
        L = CP.Assemble(L)
        # Mass = CP.Assemble(Mass)

    # print L
    SolutionTime = 0
    while eps > tol and iter < maxiter:
        iter += 1
        x = Function(W)

        uu = Function(W)
        tic()
        PP,Pb = assemble_system(prec, L1,bcs)
        AA, bb = assemble_system(a, L1 - RHSform,  bcs)
        A,b = CP.Assemble(AA,bb)
        P = CP.Assemble(PP)
        u = b.duplicate()
        print toc()
        ff = assemble(f)
        bc.apply(ff)
        ff = ff.array()
        ff = ff[0:V.dim()]
        low_values_indices = np.abs(ff) < 1e-10
        print A
        # b = b.getSubVector(t_is)
        plt.spy(PP.sparray())
        # plt.savefig("plt1")
        F = assemble(fp)
        # bcp.apply(F)

        # L = CP.Assemble(L)
        P = S.ExactPrecond(PP,QQ,L,F,[V,Q])
        Pblas = S.ExactPrecond(PP,QQ,L,F,[V,Q],'2')
        F = CP.Assemble(F)
        # bcp.apply(QQ)
        Mass = CP.Assemble(QQ)


        u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
        p_is = PETSc.IS().createGeneral(range(W.sub(0).dim(),W.sub(0).dim()+W.sub(1).dim()))

        diag = None
        kspLAMG = PETSc.KSP()
        kspLAMG.create(comm=PETSc.COMM_WORLD)
        pc = kspLAMG.getPC()
        kspLAMG.setType('preonly')
        pc.setType('cholesky')
        # pc.setFactorSolverPackage("pastix")
        OptDB = PETSc.Options()
        OptDB['pc_factor_shift_amount'] = .1
        OptDB['pc_factor_mat_ordering_type'] = 'rcm'
        OptDB['pc_factor_mat_solver_package']  = 'mumps'
        # kspLAMG.max_it = 1
        kspLAMG.setFromOptions()
        kspLAMG = kspLAMG
        # print kspLAMG.view()

        kspNLAMG = PETSc.KSP()
        kspNLAMG.create(comm=PETSc.COMM_WORLD)
        pc = kspNLAMG.getPC()
        kspNLAMG.setType('preonly')
        pc.setType('lu')
        # pc.setFactorSolverPackage("pastix")
        # kspNLAMG.max_it = 1
        kspNLAMG.setFromOptions()
        kspNLAMG = kspNLAMG
        # print kspNLAMG.view()

        kspQCG = PETSc.KSP()
        kspQCG.create(comm=PETSc.COMM_WORLD)
        pc = kspQCG.getPC()
        kspQCG.setType('preonly')
        pc.setType('cholesky')

        # pc.setType('icc')
        # pc.setFactorSolverPackage("pastix")

        # kspQCG.max_it = 4
        kspQCG.setFromOptions()
        kspQCG = kspQCG

        Bt = P.getSubMatrix(u_is,p_is)
        FF = P.getSubMatrix(u_is,u_is)

        kspNLAMG.setOperators(FF)
        kspLAMG.setOperators(L)
        kspQCG.setOperators(Mass)


        ExactP =  sp.rand(W.dim(),W.dim(), density=0.00, format='csr')
        print ExactP.shape
        for ii in xrange(0,W.dim()):
            b = np.zeros((W.dim(),1))
            b[ii] = 1
            b = IO.arrayToVec(b)

            x1 = b.getSubVector(u_is)
            y1 = x1.duplicate()
            x2 = b.getSubVector(p_is)
            y2 = x2.duplicate()
            yOut = y2.duplicate()

            kspLAMG.solve(x2, y2)
            yy2 = F*y2
            kspQCG.solve(-yy2, yOut)
            x1 = x1 - Bt*yOut
            kspNLAMG.solve(x1, y1)
            y = b.duplicate()
            ExactP[ii,:] = (np.concatenate([y1.array, yOut.array]))


        # ExactP = ExactP.transpose()

        # ExactP = PETSc.Mat().createAIJ(size=ExactP.shape,csr=(ExactP.indptr, ExactP.indices, ExactP.data))
        print (ExactP*Pblas)

        scipy.io.savemat('out.mat', mdict={'exon': ExactP*Pblas})
        ss

#         ksp = PETSc.KSP().create()
#         ksp.setTolerances(1e-5)
#         ksp.setOperators(A,P) #.getSubMatrix(u_is,u_is),P.getSubMatrix(u_is,u_is))
#         nsp = PETSc.NullSpace().create(constant=True)
#         ksp.setNullSpace(nsp)
# #         A.destroy()
# #         P.destroy()
#         reshist = {}
#         def monitor(ksp, its, fgnorm):
#             reshist[its] = fgnorm
#         ksp.setMonitor(monitor)
#         OptDB = PETSc.Options()
#         OptDB['pc_factor_shift_amount'] = "0.1"
#         OptDB['ksp_monitor_residual'] = ' '
#         OptDB['pc_factor_mat_ordering_type'] = 'rcm'
#         OptDB['pc_factor_mat_solver_package']  = 'umfpack'
#         # kspLAMG.max_it = 1
#         ksp.setFromOptions()
#         ksp.setType('gmres')
#         pc = ksp.getPC()
#         pc.setType(PETSc.PC.Type.LU)

#         ksp.max_it=1000
#         # PETSc.ViewerHDF5.DRAW(P)
#         toc()
#         ksp.solve(b,u) #.getSubVector(u_is), u.getSubVector(u_is))
#         NSits = ksp.its
#         ksp.destroy()
#         time = toc()
#         print ":::::::::::::::::::::::::::::::::::::::::::::",NSits
#         print time
#         SolutionTime = SolutionTime +time
#         outerit += NSits




        # ksp = PETSc.KSP()
        # ksp.create(comm=PETSc.COMM_WORLD)
        # ksp.setTolerances(1e-5)
        # ksp.setType('gmres')
        # pc = ksp.getPC()
        # ksp.max_it=1
        # reshist1 = {}
        # def monitor(ksp, its, fgnorm):
        #     reshist1[its] = fgnorm
        # ksp.setMonitor(monitor)

        # pc.setType(PETSc.PC.Type.PYTHON)
        # if Solver == "LSC":
        #     pc.setPythonContext(NSprecond.LSCnew(W,A,L,Bd,dBt))
        # elif Solver == "PCD":
        #     F = assemble(fp)
        #     F = CP.Assemble(F)
        #     pc.setPythonContext(NSprecond.PCD(W, Mass, F, L))

        # ksp.setOperators(A)
        # # OptDB = PETSc.Options()
        # # OptDB['pc_factor_shift_amount'] = 1
        # # OptDB['pc_factor_mat_ordering_type'] = 'rcm'
        # # OptDB['pc_factor_mat_solver_package']  = 'mumps'
        # ksp.setFromOptions()


        # x = b.duplicate()
        # toc()
        # ksp.solve(b, x)

        time = toc()
        print time
        SolutionTime = SolutionTime +time
        print ":::::::::::::::::::::::::::::::::::::::::::::",ksp.its
        outerit += ksp.its



        # print (A*u-b).array
        # print (A*x-b).array
        print np.linalg.norm(np.abs((A*x).array)-np.abs((A*u).array))
        # ss
        # for line in reshist.values():
        #     print line
        uu = IO.vecToArray(u)
        UU = uu[0:Vdim[xx-1][0]]
        # UU[low_values_indices] = 0.0
        # print UU
        # time = time+toc()
        u1 = Function(V)
        u1.vector()[:] = u1.vector()[:] + UU

        pp = uu[V.dim():V.dim()+Q.dim()]
        # print
        # time = time+toc()
        p1 = Function(Q)
        # n = pp.shape
        # pend = assemble(pa*dx)


        # pp = Function(Q)
        p1.vector()[:] = pp
        # p1.vector()[:] = p1.vector()[:] +  pp
        # p1.vector()[:] += - assemble(p1*dx)/assemble(ones*dx)
        diff = u1.vector().array()
        # print p1.vector().array()
        eps = np.linalg.norm(diff, ord=np.Inf) #+np.linalg.norm(p1.vector().array(),ord=np.Inf)

        print '\n\n\niter=%d: norm=%g' % (iter, eps)
        print np.linalg.norm( p1.vector().array(), ord=np.inf)
        # print u1.vector().array()
        u2 = Function(V)
        u2.vector()[:] = u1.vector().array() + u_k.vector().array()
        p2 = Function(Q)
        p2.vector()[:] = p1.vector().array() + p_k.vector().array()
        p2.vector()[:] += - assemble(p2*dx)/assemble(ones*dx)
        u_k.assign(u2)
        p_k.assign(p2)

        # plot(p_k)

        uOld = np.concatenate((u_k.vector().array(),p_k.vector().array()), axis=0)
        r = IO.arrayToVec(uOld)
        # plot(p_k)
    SolTime[xx-1] = SolutionTime/iter
    AvIt[xx-1] = np.ceil(outerit/iter)

    ue = u0
    pe = p0


    AvIt[xx-1] = np.ceil(outerit/iter)
    u = interpolate(ue,V)
    p = interpolate(pe,Q)

    ua = Function(V)
    ua.vector()[:] = u_k.vector().array()
    # nonlinear[xx-1] = assemble(inner((grad(ua)*ua),ua)*dx+(1/2)*div(ua)*inner(ua,ua)*dx- (1/2)*inner(ua,n)*inner(ua,ua)*ds)
    VelocityE = VectorFunctionSpace(mesh,"CG",5)
    u = interpolate(ue,VelocityE)

    PressureE = FunctionSpace(mesh,"CG",4)
    parameters["form_compiler"]["quadrature_degree"] = 5

    Nv  = ua.vector().array().shape

    X = IO.vecToArray(r)
    xu = X[0:V.dim()]
    ua = Function(V)
    ua.vector()[:] = xu

    pp = X[V.dim():V.dim()+Q.dim()]


    n = pp.shape
    pa = Function(Q)
    pa.vector()[:] = pp

    pend = assemble(pa*dx)

    ones = Function(Q)
    ones.vector()[:]=(0*pp+1)
    pp = Function(Q)
    pp.vector()[:] = pa.vector().array()- assemble(pa*dx)/assemble(ones*dx)

    pInterp = interpolate(pe,PressureE)
    pe = Function(PressureE)
    pe.vector()[:] = pInterp.vector().array()
    const = - assemble(pe*dx)/assemble(ones*dx)
    pe.vector()[:] = pe.vector()[:]+const




    ErrorU = Function(V)
    ErrorP = Function(Q)

    ErrorU = ue-ua
    ErrorP = pe-pp


    # errL2u[xx-1]= errornorm(ue, ua, norm_type='L2', degree_rise=8)
    # errH1u[xx-1]= errornorm(ue, ua, norm_type='H10', degree_rise=8)
    # errL2p[xx-1]= errornorm(pe, pp, norm_type='L2', degree_rise=8)

    errL2u[xx-1]= errornorm(ue, ua, norm_type='L2', degree_rise=8)
    # sqrt(abs(assemble(inner(ErrorU, ErrorU)*dx)))
    # errornorm(ue, ua, norm_type='L2', degree_rise=8)
    errH1u[xx-1]= errornorm(ue, ua, norm_type='H10', degree_rise=8)
    # sqrt(abs(assemble(inner(grad(ErrorU), grad(ErrorU))*dx)))
    # errornorm(ue, ua, norm_type='H10', degree_rise=8)
    errL2p[xx-1]= sqrt(abs(assemble(inner(ErrorP, ErrorP)*dx)))
    # errornorm(pe, pp, norm_type='L2', degree_rise=8)

    if xx == 1:
        l2uorder[xx-1] = 0
        l2porder[xx-1] = 0
    else:
        l2uorder[xx-1] =  np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1]))
        H1uorder[xx-1] =  np.abs(np.log2(errH1u[xx-2]/errH1u[xx-1]))

        l2porder[xx-1] =  np.abs(np.log2(errL2p[xx-2]/errL2p[xx-1]))
    print errL2u[xx-1]
    print errL2p[xx-1]
    # del  solver




print nonlinear



print "Velocity Elements rate of convergence ", np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1])))
print "Pressure Elements rate of convergence ", np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1])))


import pandas as pd
# tableTitles = ["Total DoF","V DoF","Q DoF","AvIt","V-L2","V-order","P-L2","P-order"]
# tableValues = np.concatenate((Wdim,Vdim,Qdim,AvIt,errL2u,l2uorder,errL2p,l2porder),axis=1)
# df = pd.DataFrame(tableValues, columns = tableTitles)
# pd.set_option('precision',3)
# print df
# print df.to_latex()

print "\n\n   Velocity convergence"
VelocityTitles = ["Total DoF","V DoF","Soln Time","AvIt","V-L2","L2-order","V-H1","H1-order"]
VelocityValues = np.concatenate((Wdim,Vdim,SolTime,AvIt,errL2u,l2uorder,errH1u,H1uorder),axis=1)
VelocityTable= pd.DataFrame(VelocityValues, columns = VelocityTitles)
pd.set_option('precision',3)
VelocityTable = MO.PandasFormat(VelocityTable,"V-L2","%2.4e")
VelocityTable = MO.PandasFormat(VelocityTable,'V-H1',"%2.4e")
VelocityTable = MO.PandasFormat(VelocityTable,"H1-order","%1.2f")
VelocityTable = MO.PandasFormat(VelocityTable,'L2-order',"%1.2f")
print VelocityTable

print "\n\n   Pressure convergence"
PressureTitles = ["Total DoF","P DoF","Soln Time","AvIt","P-L2","L2-order"]
PressureValues = np.concatenate((Wdim,Qdim,SolTime,AvIt,errL2p,l2porder),axis=1)
PressureTable= pd.DataFrame(PressureValues, columns = PressureTitles)
pd.set_option('precision',3)
PressureTable = MO.PandasFormat(PressureTable,"P-L2","%2.4e")
PressureTable = MO.PandasFormat(PressureTable,'L2-order',"%1.2f")
print PressureTable


LatexTitles = ["l","IT","DoFu","Dofp","V-L2","L2-order","V-H1","H1-order","P-L2","PL2-order"]
LatexValues = np.concatenate((NN,AvIt,Vdim,Qdim,errL2u,l2uorder,errH1u,H1uorder,errL2p,l2porder), axis=1)
LatexTable = pd.DataFrame(LatexValues, columns = LatexTitles)
pd.set_option('precision',3)
LatexTable = MO.PandasFormat(LatexTable,"V-L2","%2.4e")
LatexTable = MO.PandasFormat(LatexTable,'V-H1',"%2.4e")
LatexTable = MO.PandasFormat(LatexTable,"H1-order","%1.2f")
LatexTable = MO.PandasFormat(LatexTable,'L2-order',"%1.2f")
LatexTable = MO.PandasFormat(LatexTable,"P-L2","%2.4e")
LatexTable = MO.PandasFormat(LatexTable,'PL2-order',"%1.2f")
print LatexTable



# plot(u_k)
# # plot(interpolate(ue,V))

# plot(p_k)
# # plot(interpolate(p0,Q))

# interactive()


# plt.show()
