#!/usr/bin/python
from dolfin import *
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
Print = PETSc.Sys.Print

import numpy as np
import scipy.sparse as sp
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pylab as plt
import PETScIO as IO
import scipy
import scipy.io as io
import CheckPetsc4py as CP
import MaxwellPrecond as MP
import StokesPrecond as SP
import time
import MatrixOperations as MO
import MatrixMulti as MM

def PETScCheck(A):
    if dolfin_version() == '1.6.0':
        return as_backend_type(A).mat()
    else:
        return PETSc.Mat().createAIJ(size=A.sparray().shape,csr=(A.sparray().indptr, A.sparray().indices, A.sparray().data))

def InitialGuess(Fspace,BC,RHS,params,HiptmairMatrices,Hiptmairtol,Neumann=None,options="Orig",
        FS = "CG", InitialTol = 1e-6,System = None):
    print System
    if System == None:
        V = Fspace[0]
        P = Fspace[1]
        M = Fspace[2]
        L = Fspace[3]

        uBC = BC[0]
        pBC = BC[1]
        bBC = BC[2]
        rBC = BC[3]

        Ls = RHS[0]
        Lmaxwell = RHS[1]

        u,p = Stokes(V,P,uBC,Ls,params,FS,InitialTol,Neumann)

        b, r  = Maxwell(u,M,L,[bBC,rBC],Lmaxwell,params,HiptmairMatrices,Hiptmairtol,InitialTol)

        if (options == "Orig"):
            return u, b
        elif (options == "New"):
            return u,p,b,r
    else:
        V = Fspace[0]
        P = Fspace[1]
        M = Fspace[2]
        L = Fspace[3]
        IS = MO.IndexSet(Fspace,'2by2')

        uBC = BC[0]
        pBC = BC[1]
        bBC = BC[2]
        rBC = BC[3]

        Ls = RHS[0]
        Lmaxwell = RHS[1]
#        Astokes  = System[0][0]
#        bstokes = System[1].getSubVector(IS[0])
        u, p = Stokes(V,P,uBC,Ls,params,FS,InitialTol,Neumann=None ,A= System[0][0], b = System[1].getSubVector(IS[0]))

        b, r  = Maxwell(u,M,L,[bBC,rBC],Lmaxwell,params,HiptmairMatrices,Hiptmairtol,InitialTol)
        if (options == "Orig"):
            return u, b
        elif (options == "New"):
            return u,p,b,r


def Stokes(V,Q,BC,f,params,FS,InitialTol,Neumann=None,A = 0, b = 0):
    if A == 0:
        # parameters['linear_algebra_backend'] = 'uBLAS'

        # parameters = CP.ParameterSetup()
        # parameters["form_compiler"]["quadrature_degree"] = 6
        parameters['reorder_dofs_serial'] = False
        Split = 'No'
        if Split == 'No':
            W = V*Q

            (u, p) = TrialFunctions(W)
            (v, q) = TestFunctions(W)
            print FS
            def boundary(x, on_boundary):
                return on_boundary
            bcu = DirichletBC(W.sub(0), BC, boundary)
            u_k = Function(V)

            if FS == "DG":
                h = CellSize(V.mesh())
                h_avg =avg(h)
                a11 = inner(grad(v), grad(u))*dx
                a12 = div(v)*p*dx
                a21 = div(u)*q*dx
                a22 = 0.1*h_avg*jump(p)*jump(q)*dS
                L1  =  inner(v,f)*dx
                a = params[2]*a11-a12-a21-a22
            else:
                if W.sub(0).__str__().find("Bubble") == -1 and W.sub(0).__str__().find("CG1") != -1:

                    print "Bubble Bubble Bubble Bubble Bubble"
                    a11 = inner(grad(v), grad(u))*dx
                    a12 = div(v)*p*dx
                    a21 = div(u)*q*dx
                    h = CellSize(V.mesh())
                    beta  = 0.2
                    delta = beta*h*h
                    a22 = delta*inner(grad(p),grad(q))*dx
                    a = params[2]*a11-a12-a21-a22

                    L1  =  inner(v - delta*grad(q),f)*dx
                else:
                    a11 = inner(grad(v), grad(u))*dx
                    a12 = div(v)*p*dx
                    a21 = div(u)*q*dx
                    L1  =  inner(v,f)*dx
                    a = params[2]*a11-a12-a21

            tic()
            AA, bb = assemble_system(a, L1, bcu)
            A,b = CP.Assemble(AA,bb)
            del AA
            x = b.duplicate()

            pp = params[2]*inner(grad(v), grad(u))*dx + (1./params[2])*p*q*dx
            PP, Pb = assemble_system(pp,L1,bcu)
            P = CP.Assemble(PP)
            del PP
        else:
            u = TrialFunction(V)
            v = TestFunction(V)
            p = TrialFunction(Q)
            q = TestFunction(Q)
            def boundary(x, on_boundary):
                return on_boundary
            W = V*Q
            bcu = DirichletBC(V,BC, boundary)
            u_k = Function(V)

            if FS == "DG":
                h = CellSize(V.mesh())
                h_avg =avg(h)
                a11 = inner(grad(v), grad(u))*dx
                a12 = div(v)*p*dx
                a21 = div(u)*q*dx
                a22 = 0.1*h_avg*jump(p)*jump(q)*dS
                L1  =  inner(v,f)*dx
            else:
                if V.__str__().find("Bubble") == -1 and V.__str__().find("CG1") != -1:
                    a11 = params[2]*inner(grad(v), grad(u))*dx
                    a12 = -div(v)*p*dx
                    a21 = div(u)*q*dx
                    L1  =  inner(v,f)*dx
                    h = CellSize(V.mesh())
                    beta  = 0.2
                    delta = beta*h*h
                    a22 = delta*inner(grad(p),grad(q))*dx
                else:
                    a11 = params[2]*inner(grad(v), grad(u))*dx
                    a12 = -div(v)*p*dx
                    a21 = div(u)*q*dx
                    L1  =  inner(v,f)*dx


            AA = assemble(a11)
            bcu.apply(AA)
            b = assemble(L1)
            bcu.apply(b)

            AA, b = assemble_system(a11,L1,bcu)
            BB = assemble(a12)
            AAA = AA
            AA = AA.sparray()
            mesh = V.mesh()
            dim = mesh.geometry().dim()
            LagrangeBoundary = np.zeros(dim*mesh.num_vertices())
            Row = (AA.sum(0)[0,:dim*mesh.num_vertices()]-AA.diagonal()[:dim*mesh.num_vertices()])
            VelocityBoundary = np.abs(Row.A1) > 1e-4
            LagrangeBoundary[VelocityBoundary] = 1
            BubbleDOF = np.ones(dim*mesh.num_cells())
            VelocityBoundary = np.concatenate((LagrangeBoundary,BubbleDOF),axis=1)
            BC = sp.spdiags(VelocityBoundary,0,V.dim(),V.dim())
            B = BB.sparray().T*BC
            A=CP.Scipy2PETSc(sp.bmat([[AA,B.T],
                      [B,None]]))
            io.savemat("A.mat", {"A":sp.bmat([[AA,B.T],
                      [B,None]])})
            b = IO.arrayToVec(np.concatenate((b.array(),np.zeros(Q.dim())),axis=0))

            mass = (1./params[2])*p*q*dx
            mass1 = assemble(mass).sparray()
            mass2 = assemble(mass)
            W = V*Q
            io.savemat("P.mat",{"P":sp.bmat([[AA,None],
                      [None,(mass1)]])})
            P=CP.Scipy2PETSc(sp.bmat([[AA,None],
                      [None,(mass1)]]))
    else:
        W = V*Q

        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)
        print FS
        def boundary(x, on_boundary):
            return on_boundary
        bcu = DirichletBC(W.sub(0),BC, boundary)
        u_k = Function(V)


        pp = params[2]*inner(grad(v), grad(u))*dx + (1./params[2])*p*q*dx
        P = assemble(pp)
        bcu.apply(P)
        P = CP.Assemble(P)


    ksp = PETSc.KSP().create()
    pc = ksp.getPC()

    ksp.setType(ksp.Type.MINRES)
    ksp.setTolerances(InitialTol)
    pc.setType(PETSc.PC.Type.PYTHON)

    if Split == "Yes":
        A = PETSc.Mat().createPython([W.dim(),W.dim()])
        A.setType('python')
        print AAA
        a = MM.Mat2x2multi(W,[CP.Assemble(AAA),CP.Scipy2PETSc(B)])
        A.setPythonContext(a)
        pc.setPythonContext(SP.ApproxSplit(W,CP.Assemble(AAA),CP.Assemble(mass2)))
    else:
        pc.setPythonContext(SP.Approx(W,P))

    ksp.setOperators(A,P)
    x = b.duplicate()
    tic()
    ksp.solve(b, x)
    print ("{:40}").format("Stokes solve, time: "), " ==>  ",("{:4f}").format(toc()),("{:9}").format("   Its: "), ("{:4}").format(ksp.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    X = x.array
    print np.linalg.norm(x)
    # print X
    ksp.destroy()
    x = X[0:V.dim()]
    p =  X[V.dim():]
    # x =
    u = Function(V)
    u.vector()[:] = x
    # print u.vector().array()
    pp = Function(Q)
    n = p.shape
    pp.vector()[:] = p

    ones = Function(Q)
    ones.vector()[:]=(0*ones.vector().array()+1)
    pp.vector()[:] += -assemble(pp*dx)/assemble(ones*dx)

#
#    PP = io.loadmat('Ptrue.mat')["Ptrue"]
#    PP = CP.Scipy2PETSc(PP)
#    x = IO.arrayToVec(np.random.rand(W.dim()))
#    y = x.duplicate()
#    yy = x.duplicate()
#    SP.ApproxFunc(W,PP,x,y)
#    SP.ApproxSplitFunc(W,CP.Scipy2PETSc(AA),CP.Scipy2PETSc(mass),x,yy)
#    print np.linalg.norm(y.array-yy.array)
#    sssss

    return u, pp


def Maxwell(u,V,Q,BC,f,params,HiptmairMatrices,Hiptmairtol,InitialTol,Neumann=None):
    # parameters['linear_algebra_backend'] = 'uBLAS'
    dim = f.shape()[0]
    W = V*Q

    (b, r) = TrialFunctions(W)
    (c,s) = TestFunctions(W)

    def boundary(x, on_boundary):
        return on_boundary

    bcb = DirichletBC(W.sub(0),BC[0], boundary)
    bcr = DirichletBC(W.sub(1),BC[1], boundary)
    if params[0] == 0:
        a11 = params[1]*inner(curl(c),curl(b))*dx
    else:
        if dim == 2:
            a11 = params[1]*params[0]*inner(curl(c),curl(b))*dx
        elif dim == 3:
            a11 = params[1]*params[0]*inner(curl(c),curl(b))*dx

    a12 = inner(c,grad(r))*dx
    a21 = inner(b,grad(s))*dx
    Lmaxwell  = inner(c, f)*dx
    maxwell = a11+a12+a21



    bcs =  [bcb,bcr]
    tic()
    AA, bb = assemble_system(maxwell, Lmaxwell, bcs)
    A,bb = CP.Assemble(AA,bb)
    del AA
    x = bb.duplicate()

    u_is = PETSc.IS().createGeneral(range(V.dim()))
    p_is = PETSc.IS().createGeneral(range(V.dim(),V.dim()+Q.dim()))




    ksp = PETSc.KSP().create()
    ksp.setTolerances(InitialTol)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    # ksp.setType('preonly')
    # pc.setType('lu')
    # [G, P, kspVector, kspScalar, kspCGScalar, diag]
    reshist = {}
    def monitor(ksp, its, rnorm):
        print rnorm
        reshist[its] = rnorm


    # ksp.setMonitor(monitor)
    if V.__str__().find("N1curl2") == -1:
        ksp.setOperators(A)
        pc.setPythonContext(MP.Hiptmair(W, HiptmairMatrices[3], HiptmairMatrices[4], HiptmairMatrices[2], HiptmairMatrices[0], HiptmairMatrices[1], HiptmairMatrices[6],Hiptmairtol))
    else:
        p = params[1]*params[0]*inner(curl(c),curl(b))*dx+inner(c,b)*dx + inner(grad(r),grad(s))*dx
        P = assemble(p)
        for bc in bcs:
            bc.apply(P)
        P = CP.Assemble(P)
        ksp.setOperators(A,P)
        pc.setType(PETSc.PC.Type.LU)
#        pc.setPythonContext(MP.Direct(W))

    scale = bb.norm()
    bb = bb/scale
    start_time = time.time()
    ksp.solve(bb, x)
    print ("{:40}").format("Maxwell solve, time: "), " ==>  ",("{:4f}").format(time.time() - start_time),("{:9}").format("   Its: "), ("{:4}").format(ksp.its),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    x = x*scale


    ksp.destroy()
    X = IO.vecToArray(x)
    x = X[0:V.dim()]
    ua = Function(V)
    ua.vector()[:] = x


    p =  X[V.dim():]
    pa = Function(Q)
    pa.vector()[:] = p
    del ksp,pc
    return ua, pa
