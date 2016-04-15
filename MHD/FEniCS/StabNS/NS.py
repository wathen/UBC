
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
# import NSprecond
from scipy.sparse import  spdiags
import MatrixOperations as MO
import ExactSol
import NSprecondSetup as NSsetup
import NSpreconditioner as NSprecond
# parameters["form_compiler"]["optimize"]     = True
# parameters["form_compiler"]["cpp_optimize"] = True
#MO.SwapBackend('epetra')
#os.system("echo $PATH")
m = 7
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
Solver = 's'
Saving = 'no'
case = 4
# parameters['linear_algebra_backend'] = 'uBLAS'
# parameters = CP.ParameterSetup()
def LOG(arg):
    if INFO:
        print(arg)




for xx in xrange(1,m):
    print xx
    NN[xx-1] = xx+0
    nn = 2**(NN[xx-1])
    # Create mesh and define function space
    nn = int(nn)


    mesh = UnitSquareMesh(nn,nn)
    # tic()
    parameters["form_compiler"]["quadrature_degree"] = -1

    parameters['reorder_dofs_serial'] = False
    V = VectorFunctionSpace(mesh, "CG", 1)
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
    u_k,p_k = common.Stokes(V,Q,u0,f,[1,1,MU], FS = "CG", InitialTol=1e-10)
    # p_k.vector()[:] = p_k.vector().array()
    pConst = - assemble(p_k*dx)/assemble(ones*dx)
    p_k.vector()[:] += pConst
    # u_k = Function(V)
    # p_k = Function(Q)
    # plot(u_k)
    # plot(p_k)
    uOld = np.concatenate((u_k.vector().array(),p_k.vector().array()), axis=0)
    r = IO.arrayToVec(uOld)

    a11 = MU*inner(grad(v), grad(u))*dx + inner((grad(u)*u_k),v)*dx + (1./2)*div(u_k)*inner(u,v)*dx - (1./2)*inner(u_k,n)*inner(u,v)*ds(mesh)
    a12 = div(v)*p*dx
    a21 = div(u)*q*dx
    if Q._FunctionSpace___degree == 1:
        a22 = 0.1*inner(grad(p),grad(q))*dx
    elif Q._FunctionSpace___family == "DG":
        a22 = 0.2*h_avg*jump(p)*jump(q)*dS(mesh)
    else:
        a22 = 0
    # L1  = inner(v-0.2*h*h*grad(q), f)*dx
    L1  = inner(v, f)*dx
    a = a11-a12-a21-a22


    r11 = MU*inner(grad(v), grad(u_k))*dx + inner((grad(u_k)*u_k),v)*dx + (1./2)*div(u_k)*inner(u_k,v)*dx - (1./2)*inner(u_k,n)*inner(u_k,v)*ds(mesh)
    r12 = div(v)*p_k*dx
    r21 = div(u_k)*q*dx
    if Q._FunctionSpace___degree == 1:
        r22 = 0.1*inner(grad(p_k),grad(q))*dx
    elif Q._FunctionSpace___family == "DG":
        r22 = 0.2*h_avg*jump(p_k)*jump(q)*dS(mesh)
    else:
        r22 = 0
    RHSform = r11-r12-r21-r22
    # -r22
    # RHSform = 0

    p11 = inner(u,v)*dx
    # p12 = div(v)*p*dx
    # p21 = div(u)*q*dx
    p22 = inner(p,q)*dx
    prec = p11 +p22
    bc = DirichletBC(W.sub(0),Expression(("0.0","0.0")), boundary)
    bcs = [bc]

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-4      # tolerance
    iter = 0            # iteration counter
    maxiter = 20        # max no of iterations allowed
    # parameters = CP.ParameterSetup()
    outerit = 0
    Solver = "LSC"
    if Solver == "LSC":
        parameters['linear_algebra_backend'] = 'uBLAS'
        u = TrialFunction(V)
        v = TestFunction(V)
        p = TrialFunction(Q)
        Mass = assemble(inner(u,v)*dx)
        B = assemble(-div(v)*p*dx)
        bc = DirichletBC(V,Expression(("0.0","0.0")), boundary)
        #bc.apply(Mass)

        Mass = Mass.sparray()
        MassD = Mass.diagonal()
        del Mass
        B = B.sparray()
        
        d = spdiags(1.0/MassD, 0, len(MassD), len(MassD))
        QB = d*B
        L = B.transpose()*QB
        L = PETSc.Mat().createAIJ(size=L.shape,csr=(L.indptr, L.indices, L.data))
        QB = PETSc.Mat().createAIJ(size=QB.shape,csr=(QB.indptr, QB.indices, QB.data))
        KspL = NSsetup.Ksp(L)
    elif Solver == "PCD":
        (pQ) = TrialFunction(Q)
        (qQ) = TestFunction(Q)
        print MU
        Mass = assemble(inner(pQ,qQ)*dx)
        L = assemble(inner(grad(pQ),grad(qQ))*dx)

        fp = MU*inner(grad(qQ), grad(pQ))*dx+inner((u_k[0]*grad(pQ)[0]+u_k[1]*grad(pQ)[1]),qQ)*dx + (1./2)*div(u_k)*inner(pQ,qQ)*dx - (1./2)*(u_k[0]*n[0]+u_k[1]*n[1])*inner(pQ,qQ)*ds
        L = CP.Assemble(L)
        Mass = CP.Assemble(Mass)

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
        PP = assemble(prec)
        bcc.apply(PP)
        P = CP.Assemble(PP)


        b = bb.array()
        zeros = 0*b
        bb = IO.arrayToVec(b)
        x = IO.arrayToVec(zeros)
        ksp = PETSc.KSP()
        ksp.create(comm=PETSc.COMM_WORLD)
        # ksp.setTolerances(1e-5)
        ksp.setType('gmres')
        pc = ksp.getPC()

        u_is = PETSc.IS().createGeneral(range(V.dim()))
        F = A.getSubMatrix(u_is,u_is)
        KspF = NSsetup.Ksp(F)
        
        pc.setType('python')
        pc.setType(PETSc.PC.Type.PYTHON)
        
        if Solver == "LSC":
            pc.setPythonContext(NSprecond.NSLSC(W,KspF,KspL,QB))
        
        # elif Solver == "PCD":
            # F = assemble(fp)
            # F = CP.Assemble(F)
        #     pc.setPythonContext(NSprecond.PCD(W, A, Mass, F, L))

        ksp.setOperators(A)
        OptDB = PETSc.Options()
        # OptDB['pc_factor_shift_amount'] = 1
        OptDB['pc_factor_mat_ordering_type'] = 'rcm'
        OptDB['pc_factor_mat_solver_package']  = 'mumps'
        ksp.setFromOptions()



        toc()
        ksp.solve(bb, x)

        time = toc()
        print time
        SolutionTime = SolutionTime +time
        print '====================================',ksp.its
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

        pp = uu[V.dim():V.dim()+Q.dim()]
        # time = time+toc()
        p1 = Function(Q)
        # n = pp.shape
        # pend = assemble(pa*dx)


        # pp = Function(Q)
        # p1.vector()[:] = pa.vector().array()- assemble(pa*dx)/assemble(ones*dx)
        p1.vector()[:] = p1.vector()[:] +  pp
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
    AvIt[xx-1] = np.ceil(outerit/iter)

    ue = u0
    pe = p0


    AvIt[xx-1] = np.ceil(outerit/iter)
    u = interpolate(ue,V)
    p = interpolate(pe,Q)

    ua = Function(V)
    ua.vector()[:] = u_k.vector().array()
    # nonlinear[xx-1] = assemble(inner((grad(ua)*ua),ua)*dx+(1/2)*div(ua)*inner(ua,ua)*dx- (1/2)*inner(ua,n)*inner(ua,ua)*ds)
    VelocityE = VectorFunctionSpace(mesh,"CG",3)
    u = interpolate(ue,VelocityE)

    PressureE = FunctionSpace(mesh,"CG",2)
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

    errL2u[xx-1]= errornorm(ue, ua, norm_type='L2', degree_rise=4)
    # sqrt(abs(assemble(inner(ErrorU, ErrorU)*dx)))
    # errornorm(ue, ua, norm_type='L2', degree_rise=8)
    errH1u[xx-1]= errornorm(ue, ua, norm_type='H10', degree_rise=4)
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


LatexTitles = ["l","DoFu","Dofp","V-L2","L2-order","V-H1","H1-order","P-L2","PL2-order"]
LatexValues = np.concatenate((NN,Vdim,Qdim,errL2u,l2uorder,errH1u,H1uorder,errL2p,l2porder), axis=1)
LatexTable = pd.DataFrame(LatexValues, columns = LatexTitles)
pd.set_option('precision',3)
LatexTable = MO.PandasFormat(LatexTable,"V-L2","%2.4e")
LatexTable = MO.PandasFormat(LatexTable,'V-H1',"%2.4e")
LatexTable = MO.PandasFormat(LatexTable,"H1-order","%1.2f")
LatexTable = MO.PandasFormat(LatexTable,'L2-order',"%1.2f")
LatexTable = MO.PandasFormat(LatexTable,"P-L2","%2.4e")
LatexTable = MO.PandasFormat(LatexTable,'PL2-order',"%1.2f")
print LatexTable.to_latex()



# plot(ua)
# plot(interpolate(ue,V))

# plot(pp)
# plot(interpolate(p0,Q))

# interactive()

# plt.show()

