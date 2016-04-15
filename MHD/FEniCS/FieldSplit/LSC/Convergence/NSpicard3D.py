
#!/opt/local/bin/python

from dolfin import *
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

# from MatrixOperations import *
import numpy as np
#import matplotlib.pylab as plt
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
m = 3
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
case = 1
# parameters['linear_algebra_backend'] = 'uBLAS'
parameters = CP.ParameterSetup()
def LOG(arg):
    if INFO:
        print(arg)




for xx in xrange(1,m):
    print xx
    nn = 2**(xx+1)
    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = xx +1
    parameters["form_compiler"]["quadrature_degree"] = -1

    mesh = BoxMesh(0, 0, 0, 1, 1, 1, nn, nn,nn)
    # tic()

    parameters['reorder_dofs_serial'] = False
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
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

    if case == 1:
        u0 =Expression(("x[0]*sin(x[1])*exp(x[2])","cos(x[1])*exp(x[2])","sin(x[1])*cos(x[0])"))
        p0 = Expression("sin(x[0])*cos(x[1])*exp(x[2])")
        # u0 = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
        # p0 = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")
    # elif case == 2:
    #     u0 = Expression(("sin(x[1])*exp(x[0])","cos(x[1])*exp(x[0])"))
    #     p0 = Expression("sin(x[0])*cos(x[1])")

    R = 10.0
    # MU = Constant(0.01)
    MU = 10.0
    bcc = DirichletBC(W.sub(0),u0, boundary)
    bcs = [bcc]

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    if case == 1:
        Laplacian = -MU*Expression(("0","0","-2*cos(x[0])*sin(x[1])"))
        Advection = Expression(("x[0]*pow(exp(x[2]),2)+x[0]*cos(x[0])*pow(sin(x[1]),2)*exp(x[2])","-cos(x[1])*sin(x[1])*pow(exp(x[2]),2)+cos(x[0])*cos(x[1])*sin(x[1])*exp(x[2])","-x[0]*pow(sin(x[1]),2)*exp(x[2])*sin(x[0])+pow(cos(x[1]),2)*exp(x[2])*cos(x[0])"))
        gradPres = Expression(("cos(x[1])*cos(x[0])*exp(x[2])","-sin(x[1])*sin(x[0])*exp(x[2])","sin(x[0])*cos(x[1])*exp(x[2])"))

        f = Laplacian+Advection+gradPres




    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    d = 0
    u_k,p_k = common.Stokes(V,Q,u0,Laplacian+gradPres,[1,1,MU])

    uOld = np.concatenate((u_k.vector().array(),p_k.vector().array()), axis=0)
    r = IO.arrayToVec(uOld)

    a11 = MU*inner(grad(v), grad(u))*dx + inner((grad(u)*u_k),v)*dx + (1/2)*div(u_k)*inner(u,v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds
    a12 = div(v)*p*dx
    a21 = div(u)*q*dx
    L1  = inner(v, f)*dx
    a = a11-a12-a21


    r11 = MU*inner(grad(v), grad(u_k))*dx + inner((grad(u_k)*u_k),v)*dx + (1/2)*div(u_k)*inner(u_k,v)*dx- (1/2)*inner(u_k,n)*inner(u_k,v)*ds
    r12 = div(v)*p_k*dx
    r21 = div(u_k)*q*dx
    RHSform = r11-r12-r21


    p11 = inner(u,v)*dx
    # p12 = div(v)*p*dx
    # p21 = div(u)*q*dx
    p22 = inner(p,q)*dx
    prec = p11 +p22
    bc = DirichletBC(W.sub(0),Expression(("0","0","0")), boundary)
    bcs = [bc]

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-5       # tolerance
    iter = 0            # iteration counter
    maxiter = 10        # max no of iterations allowed
    parameters = CP.ParameterSetup()
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
        print 1
        (pQ) = TrialFunction(Q)
        (qQ) = TestFunction(Q)
        print MU
        Mass = assemble(inner(pQ,qQ)*dx)
        L = assemble(inner(grad(pQ),grad(qQ))*dx)

        fp = MU*inner(grad(qQ), grad(pQ))*dx+inner((u_k[0]*grad(pQ)[0]+u_k[1]*grad(pQ)[1]+u_k[2]*grad(pQ)[2]),qQ)*dx + (1/2)*div(u_k)*inner(pQ,qQ)*dx - (1/2)*(u_k[0]*n[0]+u_k[1]*n[1]+u_k[2]*n[2])*inner(pQ,qQ)*ds
        # print "hi"
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
        print A
        # b = b.getSubVector(t_is)



        b = bb.array()
        zeros = 0*b
        bb = IO.arrayToVec(b)
        x = IO.arrayToVec(zeros)
        ksp = PETSc.KSP()
        ksp.create(comm=PETSc.COMM_WORLD)
        # ksp.setTolerances(1e-4)
        ksp.setType('preonly')
        pc = ksp.getPC()
        # ksp.max_it("1000")
        # ksp.max_it=500

        pc.setType(PETSc.PC.Type.LU)
        if Solver == "LSC":
            pc.setPythonContext(NSprecond.LSCnew(W,A,L,Bd,dBt))
        elif Solver == "PCD":
            F = assemble(fp)
            F = CP.Assemble(F)
            pc.setPythonContext(NSprecond.PCD(W, A, Mass, F, L))

        ksp.setOperators(A)
        OptDB = PETSc.Options()
        OptDB['pc_factor_shift_amount'] = 0.01
        OptDB['pc_factor_mat_ordering_type'] = 'rcm'
        OptDB['pc_factor_mat_solver_package']  = 'mumps'
        ksp.setFromOptions()



        toc()
        ksp.solve(bb, x)

        time = toc()
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

        u2 = Function(V)
        u2.vector()[:] = u1.vector().array() + u_k.vector().array()
        p2 = Function(Q)
        p2.vector()[:] = p1.vector().array() + p_k.vector().array()
        u_k.assign(u2)
        p_k.assign(p2)

        uOld = np.concatenate((u_k.vector().array(),p_k.vector().array()), axis=0)
        r = IO.arrayToVec(uOld)

    del A, AA,ksp,pc, bb,b,

    SolTime[xx-1] = SolutionTime/iter
    AvIt[xx-1] = np.ceil(outerit/iter)

    if case == 1:
        ue = u0
        pe = p0
    elif case == 2:
        ue = u0
        pe = p0

    AvIt[xx-1] = np.ceil(outerit/iter)
    u = interpolate(ue,V)
    p = interpolate(pe,Q)

    ua = Function(V)
    ua.vector()[:] = u_k.vector().array()
    # nonlinear[xx-1] = assemble(inner((grad(ua)*ua),ua)*dx+(1/2)*div(ua)*inner(ua,ua)*dx- (1/2)*inner(ua,n)*inner(ua,ua)*ds)
    VelocityE = VectorFunctionSpace(mesh,"CG",6)
    u = interpolate(ue,VelocityE)

    PressureE = FunctionSpace(mesh,"CG",5)

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


    errL2u[xx-1]= errornorm(ue, ua, norm_type='L2', degree_rise=8)
    errH1u[xx-1]= errornorm(ue, ua, norm_type='H10', degree_rise=8)
    errL2p[xx-1]= errornorm(pe, pp, norm_type='L2', degree_rise=8)

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



plot(ua)
plot(interpolate(ue,V))

plot(pp)
plot(interpolate(pe,Q))

interactive()

#plt.show()

