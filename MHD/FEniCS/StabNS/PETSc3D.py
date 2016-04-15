
#!/opt/local/bin/python

import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from dolfin import *
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
import NSpreconditioner
from scipy.sparse import  spdiags
import MatrixOperations as MO
import ExactSol
import NSprecondSetup
import matplotlib.pylab as plt
import pandas as pd
# parameters["form_compiler"]["optimize"]     = True
# parameters["form_compiler"]["cpp_optimize"] = True
#MO.SwapBackend('epetra')
#os.system("echo $PATH")
m = 5
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
NonLinearIts = np.zeros((m-1,1))
nn = 2

dim = 3
Solver = 'PCD'
Saving = 'no'
case = 1
# parameters['linear_algebra_backend'] = 'uBLAS'
# parameters = CP.ParameterSetup()
def LOG(arg):
    if INFO:
        print(arg)




for xx in xrange(1,m):
    print xx
    NN[xx-1] = xx+1
    nn = 2**(NN[xx-1])
    nn = int(nn)


    mesh = UnitCubeMesh(nn,nn,nn)
    # tic()
    parameters["form_compiler"]["quadrature_degree"] = -1

    parameters['reorder_dofs_serial'] = False
    V = VectorFunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DG", 0)
    ones = Function(Q)
    ones.vector()[:]=(0*ones.vector().array()+1)
    #QQ = VectorFunctionSpace(mesh,"B",3)
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


    u0, p0, Laplacian, Advection, gradPres = ExactSol.NS3D(case)

    R = 10.0
    # MU = Constant(0.01)
    # MU = 1000.0
    MU = 1.0
    bcc = DirichletBC(W.sub(0),u0, boundary)
    bcs = [bcc]

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)




    f = -MU*Laplacian+Advection+gradPres


    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    d = 0
    u_k,p_k = common.Stokes(V,Q,u0,f,[1,1,MU], FS = "DG",InitialTol = 1e-6)
    # p_k.vector()[:] = p_k.vector().array()
    pConst = - assemble(p_k*dx)/assemble(ones*dx)
    p_k.vector()[:] += pConst
    # u_k = Function(V)
    # p_k = Function(Q)
    # plot(u_k)
    # plot(p_k)
    uOld = np.concatenate((u_k.vector().array(),p_k.vector().array()), axis=0)
    r = IO.arrayToVec(uOld)

    a11 = MU*inner(grad(v), grad(u))*dx(mesh) + inner((grad(u)*u_k),v)*dx(mesh) + (1./2)*div(u_k)*inner(u,v)*dx(mesh) - (1./2)*inner(u_k,n)*inner(u,v)*ds(mesh)
    a12 = div(v)*p*dx
    a21 = div(u)*q*dx
    a22 = 0.1*h_avg*jump(p)*jump(q)*dS
    # L1  = inner(v-0.2*h*h*grad(q), f)*dx
    L1  = inner(v, f)*dx
    a = a11-a12-a21-a22


    r11 = MU*inner(grad(v), grad(u_k))*dx(mesh) + inner((grad(u_k)*u_k),v)*dx(mesh) + (1/2)*div(u_k)*inner(u_k,v)*dx(mesh) - (1/2)*inner(u_k,n)*inner(u_k,v)*ds(mesh)
    r12 = div(v)*p_k*dx
    r21 = div(u_k)*q*dx
    r22 = 0.1*h_avg*jump(p_k)*jump(q)*dS
    RHSform = r11-r12-r21-r22
    # -r22
    # RHSform = 0

    p11 = inner(u,v)*dx
    # p12 = div(v)*p*dx
    # p21 = div(u)*q*dx
    p22 = inner(p,q)*dx
    prec = p11 +p22
    bc = DirichletBC(W.sub(0),Expression(("0.0","0.0","0.0")), boundary)
    bcs = [bc]

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-4      # tolerance
    iter = 0            # iteration counter
    maxiter = 20        # max no of iterations allowed
    # parameters = CP.ParameterSetup()
    outerit = 0

    if Solver == "LSC":
        parameters['linear_algebra_backend'] = 'uBLAS'
        PP = assemble(inner(u,v)*dx-div(v)*p*dx)
        bc.apply(PP)
        PP = PP.sparray()
        X = PP[0:V.dim(),0:V.dim()]
        Xdiag = X.diagonal()
        # PP = assemble(-div(v)*p*dx)
        # bc.apply(PP)
        # PP = PP.sparray()
        # Xdiag = X.sum(1).A
        # print Xdiag



        Bt = PP[0:V.dim(),V.dim():W.dim()]
        d = spdiags(1.0/Xdiag, 0, len(Xdiag), len(Xdiag))
        dBt = (d*Bt).tocsr()
        print Bt.transpose()*dBt.todense()

        plt.spy(dBt)
        plt.show()
        BQB = Bt.transpose()*dBt
        dBt = PETSc.Mat().createAIJ(size=dBt.shape,csr=(dBt.indptr, dBt.indices, dBt.data))
        print dBt.size
        BQB = PETSc.Mat().createAIJ(size=BQB.tocsr().shape,csr=(BQB.tocsr().indptr, BQB.tocsr().indices, BQB.tocsr().data))
        # parameters['linear_algebra_backend'] = 'PETSc'
        kspBQB = NSprecondSetup.LSCKSPlinear(BQB)
    elif Solver == "PCD":
        N = FacetNormal(mesh)
        h = CellSize(mesh)
        h_avg =avg(h)
        alpha = 10.0
        gamma =10.0

        (pQ) = TrialFunction(Q)
        (qQ) = TestFunction(Q)
        print MU
        Mass = assemble(inner(pQ,qQ)*dx)
        L = MU*(inner(grad(qQ), grad(pQ))*dx(mesh) \
                        - inner(avg(grad(qQ)), outer(pQ('+'),N('+'))+outer(pQ('-'),N('-')))*dS(mesh) \
                        - inner(outer(qQ('+'),N('+'))+outer(qQ('-'),N('-')), avg(grad(pQ)))*dS(mesh) \
                        + alpha/h_avg*inner(outer(qQ('+'),N('+'))+outer(qQ('-'),N('-')),outer(pQ('+'),N('+'))+outer(pQ('-'),N('-')))*dS(mesh) \
                        - inner(outer(qQ,N), grad(pQ))*ds(mesh) \
                        - inner(grad(qQ), outer(pQ,N))*ds(mesh) \
                        + gamma/h*inner(qQ,pQ)*ds(mesh))

        O =inner(inner(grad(pQ),u_k),qQ)*dx(mesh)\
                - (1./2)*inner(u_k,n)*inner(qQ,pQ)*ds(mesh) \
                -(1/2)*(dot(u_k('+'),N('+'))+dot(u_k('-'),N('-')))*avg(inner(qQ,pQ))*ds(mesh) \
                -dot(avg(qQ),dot(outer(pQ('+'),N('+'))+outer(pQ('-'),N('-')),avg(u_k)))*dS(mesh)        
        Laplacian = assemble(L)
        Laplacian = CP.Assemble(Laplacian)
        Mass = CP.Assemble(Mass)
        kspA, kspQ = NSprecondSetup.PCDKSPlinear(Mass,Laplacian)

    u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
    p_is = PETSc.IS().createGeneral(range(W.sub(0).dim(),W.sub(0).dim()+W.sub(1).dim()))
    # print L
    SolutionTime = 0
    while eps > tol and iter < maxiter:
        iter += 1
        x = Function(W)

        uu = Function(W)
        tic()
        AA, bb = assemble_system(a, L1-RHSform, bcs)
        A,b = CP.Assemble(AA,bb)
        print toc()
        # b = b.getSubVector(t_is)
        F = A.getSubMatrix(u_is,u_is)

        kspF = NSprecondSetup.LSCKSPnonlinear(F)

        x = b.duplicate()
        ksp = PETSc.KSP()
        ksp.create(comm=PETSc.COMM_WORLD)
        # ksp.setTolerances(1e-5)
        ksp.setType('gmres')
        pc = ksp.getPC()
        Fp = assemble(L+O)
        Fp = CP.Assemble(Fp)


        pc.setType(PETSc.PC.Type.PYTHON)
        # if Solver == "LSC":
        pc.setPythonContext(NSpreconditioner.NSPCD(W, kspF, kspA, kspQ,Fp))
        # elif Solver == "PCD":
            # F = assemble(fp)
            # F = CP.Assemble(F)
        #     pc.setPythonContext(NSprecond.PCD(W, A, Mass, F, L))

        ksp.setOperators(A)
        OptDB = PETSc.Options()
        ksp.max_it = 100
        # OptDB['pc_factor_shift_amount'] = 1
        # OptDB['pc_factor_mat_ordering_type'] = 'rcm'
        # OptDB['pc_factor_mat_solver_package']  = 'mumps'
        ksp.setFromOptions()


        x = r

        toc()
        ksp.solve(b, x)

        time = toc()
        print time
        SolutionTime = SolutionTime +time
        outerit += ksp.its
        print "==============", ksp.its
        # r = bb.duplicate()
        # A.MUlt(x, r)
        # r.aypx(-1, bb)
        # rnorm = r.norm()
        # PETSc.Sys.Print('error norm = %g' % rnorm,comm=PETSc.COMM_WORLD)

        uu = IO.vecToArray(x)
        UU = uu[0:Vdim[xx-1][0]]
        # time = time+toc()
        u1 = Function(V)
        u1.vector()[:] = UU

        pp = uu[V.dim():V.dim()+Q.dim()]
        # time = time+toc()
        p1 = Function(Q)
        # n = pp.shape
        # pend = assemble(pa*dx)


        # pp = Function(Q)
        # p1.vector()[:] = pa.vector().array()- assemble(pa*dx)/assemble(ones*dx)
        p1.vector()[:] = pp
        p1.vector()[:] += - assemble(p1*dx)/assemble(ones*dx)
        diff = u1.vector().array()
        # print p1.vector().array()
        eps = np.linalg.norm(diff, ord=np.Inf) #+np.linalg.norm(p1.vector().array(),ord=np.Inf)

        print '\n\n\niter=%d: norm=%g' % (iter, eps)
        print np.linalg.norm(p1.vector().array(), ord=np.Inf)
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

    NonLinearIts[xx-1] = iter
    ue = u0
    pe = p0


    AvIt[xx-1] = float(outerit)/iter
#     u = interpolate(ue,V)
#     p = interpolate(pe,Q)

#     ua = Function(V)
#     ua.vector()[:] = u_k.vector().array()
#     VelocityE = VectorFunctionSpace(mesh,"CG",3)
#     u = interpolate(ue,VelocityE)

#     PressureE = FunctionSpace(mesh,"DG",2)
#     parameters["form_compiler"]["quadrature_degree"] = 5

#     Nv  = ua.vector().array().shape

#     X = IO.vecToArray(r)
#     xu = X[0:V.dim()]
#     ua = Function(V)
#     ua.vector()[:] = xu

#     pp = X[V.dim():V.dim()+Q.dim()]


#     n = pp.shape
#     pa = Function(Q)
#     pa.vector()[:] = pp

#     pend = assemble(pa*dx)

#     ones = Function(Q)
#     ones.vector()[:]=(0*pp+1)
#     pp = Function(Q)
#     pp.vector()[:] = pa.vector().array()- assemble(pa*dx)/assemble(ones*dx)

#     pInterp = interpolate(pe,PressureE)
#     pe = Function(PressureE)
#     pe.vector()[:] = pInterp.vector().array()
#     const = - assemble(pe*dx)/assemble(ones*dx)
#     pe.vector()[:] = pe.vector()[:]+const




#     ErrorU = Function(V)
#     ErrorP = Function(Q)

#     ErrorU = ue-ua
#     ErrorP = pe-pp


#     # errL2u[xx-1]= errornorm(ue, ua, norm_type='L2', degree_rise=8)
#     # errH1u[xx-1]= errornorm(ue, ua, norm_type='H10', degree_rise=8)
#     # errL2p[xx-1]= errornorm(pe, pp, norm_type='L2', degree_rise=8)

#     errL2u[xx-1]= errornorm(ue, ua, norm_type='L2', degree_rise=4)
#     # sqrt(abs(assemble(inner(ErrorU, ErrorU)*dx)))
#     # errornorm(ue, ua, norm_type='L2', degree_rise=8)
#     errH1u[xx-1]= errornorm(ue, ua, norm_type='H10', degree_rise=4)
#     # sqrt(abs(assemble(inner(grad(ErrorU), grad(ErrorU))*dx)))
#     # errornorm(ue, ua, norm_type='H10', degree_rise=8)
#     errL2p[xx-1]= sqrt(abs(assemble(inner(ErrorP, ErrorP)*dx)))
#     # errornorm(pe, pp, norm_type='L2', degree_rise=8)

#     if xx == 1:
#         l2uorder[xx-1] = 0
#         l2porder[xx-1] = 0
#     else:
#         l2uorder[xx-1] =  np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1]))
#         H1uorder[xx-1] =  np.abs(np.log2(errH1u[xx-2]/errH1u[xx-1]))

#         l2porder[xx-1] =  np.abs(np.log2(errL2p[xx-2]/errL2p[xx-1]))
#     print errL2u[xx-1]
#     print errL2p[xx-1]
#     # del  solver




# print nonlinear



# print "Velocity Elements rate of convergence ", np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1])))
# print "Pressure Elements rate of convergence ", np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1])))


# import pandas as pd
# # tableTitles = ["Total DoF","V DoF","Q DoF","AvIt","V-L2","V-order","P-L2","P-order"]
# # tableValues = np.concatenate((Wdim,Vdim,Qdim,AvIt,errL2u,l2uorder,errL2p,l2porder),axis=1)
# # df = pd.DataFrame(tableValues, columns = tableTitles)
# # pd.set_option('precision',3)
# # print df
# # print df.to_latex()

# print "\n\n   Velocity convergence"
# VelocityTitles = ["Total DoF","V DoF","Soln Time","AvIt","V-L2","L2-order","V-H1","H1-order"]
# VelocityValues = np.concatenate((Wdim,Vdim,SolTime,AvIt,errL2u,l2uorder,errH1u,H1uorder),axis=1)
# VelocityTable= pd.DataFrame(VelocityValues, columns = VelocityTitles)
# pd.set_option('precision',3)
# VelocityTable = MO.PandasFormat(VelocityTable,"V-L2","%2.4e")
# VelocityTable = MO.PandasFormat(VelocityTable,'V-H1',"%2.4e")
# VelocityTable = MO.PandasFormat(VelocityTable,"H1-order","%1.2f")
# VelocityTable = MO.PandasFormat(VelocityTable,'L2-order',"%1.2f")
# print VelocityTable

# print "\n\n   Pressure convergence"
# PressureTitles = ["Total DoF","P DoF","Soln Time","AvIt","P-L2","L2-order"]
# PressureValues = np.concatenate((Wdim,Qdim,SolTime,AvIt,errL2p,l2porder),axis=1)
# PressureTable= pd.DataFrame(PressureValues, columns = PressureTitles)
# pd.set_option('precision',3)
# PressureTable = MO.PandasFormat(PressureTable,"P-L2","%2.4e")
# PressureTable = MO.PandasFormat(PressureTable,'L2-order',"%1.2f")
# print PressureTable

# print "\n\n"

# LatexTitles = ["l","DoFu","Dofp","V-L2","L2-order","V-H1","H1-order","P-L2","PL2-order"]
# LatexValues = np.concatenate((NN,Vdim,Qdim,errL2u,l2uorder,errH1u,H1uorder,errL2p,l2porder), axis=1)
# LatexTable = pd.DataFrame(LatexValues, columns = LatexTitles)
# pd.set_option('precision',3)
# LatexTable = MO.PandasFormat(LatexTable,"V-L2","%2.4e")
# LatexTable = MO.PandasFormat(LatexTable,'V-H1',"%2.4e")
# LatexTable = MO.PandasFormat(LatexTable,"H1-order","%1.2f")
# LatexTable = MO.PandasFormat(LatexTable,'L2-order',"%1.2f")
# LatexTable = MO.PandasFormat(LatexTable,"P-L2","%2.4e")
# LatexTable = MO.PandasFormat(LatexTable,'PL2-order',"%1.2f")
# print LatexTable.to_latex()


print "\n\n\n\n"

LatexTitles = ["l","DoFu","Dofp","Soln Time","AvIt","Non-Lin its"]
LatexValues = np.concatenate((NN,Vdim,Qdim, SolTime,AvIt, NonLinearIts), axis=1)
LatexTable = pd.DataFrame(LatexValues, columns = LatexTitles)
pd.set_option('precision',3)
LatexTable = MO.PandasFormat(LatexTable,'AvIt',"%3.1f")
print LatexTable.to_latex()


# plot(ua)
# plot(interpolate(ue,V))

# plot(pp)
# plot(interpolate(p0,Q))

# interactive()

plt.show()

