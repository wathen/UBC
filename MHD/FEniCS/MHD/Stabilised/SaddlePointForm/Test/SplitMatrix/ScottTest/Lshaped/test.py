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
from scipy.sparse import coo_matrix, csr_matrix, spdiags, bmat

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
        level[xx-1] = xx + 2
        nn = 2**(level[xx-1])



        # Create mesh and define function space
        nn = int(nn)
        NN[xx-1] = nn/2
        # parameters["form_compiler"]["quadrature_degree"] = 6
        # parameters = CP.ParameterSetup()
        mesh = UnitSquareMesh(nn,nn)
        # mesh = RectangleMesh(0,0,2*np.pi,2*np.pi,nn,nn)
        order = 1
        parameters['reorder_dofs_serial'] = False
        Velocity = VectorFunctionSpace(mesh, "CG", order)
        Pressure = FunctionSpace(mesh, "CG", order)
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
        bcb = DirichletBC(Magnetic,Expression(('0','0')), boundary)
        bcr = DirichletBC(Lagrange,Expression(('0')), boundary)

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
        print HiptmairMatrices
        C = CP.PETSc2Scipy(HiptmairMatrices[0])
        Px = CP.PETSc2Scipy(HiptmairMatrices[1][0])
        Py = CP.PETSc2Scipy(HiptmairMatrices[1][1])

        # VecV = VectorFunctionSpace(mesh,"CG",1)

        # f = Expression(('sin(x[0])','sin(x[1])'))
        # F = interpolate(f,VecV)
        # Fvec = F
        # bcrVec = DirichletBC(VecV,Expression(('0','0')), boundary)

        # bcrVec.apply(F.vector())

        # Fmag = interpolate(F,Magnetic)
        # bcb.apply(Fmag.vector())
        # print Fmag.vector().array()
        # print bmat([[Px,Py]]).shape
        # print bmat([[Px,Py]])*F.vector().array()
        # FmagP = bmat([[Px,Py]])*F.vector().array()
        # print np.max(abs(bmat([[Px,Py]])*F.vector().array()-Fmag.vector().array()))

        # print "\n\n\n\n"
        # Fmag = interpolate(f,Magnetic)
        # # plot(Fmag)
        # bcb.apply(Fmag.vector())
        # Forig = interpolate(Fmag,VecV)
        # # plot(Forig)
        # bcrVec.apply(Forig.vector())
        # print Forig.vector().array()
        # print "\n\n"
        # # print Fmag.vector().array()
        # print (bmat([[Px,Py]]).T*Fmag.vector().array())
        # print "\n\n\n\n"
        # print np.max(abs(bmat([[Px,Py]]).T*Fmag.vector().array()-Forig.vector().array()))

        u = TrialFunction(Magnetic)
        v = TestFunction(Magnetic)
        f = Expression('(x[0])')
        F = interpolate(f,Lagrange)
        # f = np.zeros(Lagrange.dim())
        f = np.zeros((Lagrange.dim(),1))[:,0]
        f[0] = 1.0
        F = Function(Lagrange)
        F.vector()[:] = f
        # bcrVec = DirichletBC(VecV,Expression(('0','0')), boundary)

        bcr.apply(F.vector())

        M = (inner(u,v)*dx)
        B = (inner(v, grad(F))*dx)

        u = Function(Magnetic)
        solve(M == B, u,bcb)
        projection = project(grad(F),Magnetic, solver_type = "lu")
        print projection
        u = bcb.apply(projection.vector())
        print projection.vector().array()
        print C*F.vector().array()
        print "\n\n\n"
        print abs(C*F.vector().array() - projection.vector().array())<1e-6
        print np.max(abs(projection.vector().array()-C*F.vector().array()))#,

        f = Expression(('(x[0])','(x[1])'))
        F = interpolate(f,Magnetic)
        bcb.apply(F.vector())
        print F.vector().array()
        projection = project(div((F)),Lagrange, solver_type = "lu")
        bcr.apply(projection.vector())
        print projection.vector().array()
        print C.T*F.vector().array()
        print "\n\n\n"
        print abs(C.T*F.vector().array() - projection.vector().array())
        print np.max(abs(projection.vector().array()-C.T*F.vector().array()))#,



    interactive()
foo()
