#!/usr/bin/python
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
import time

import BiLinear as forms
import IterOperations as Iter
import MatrixOperations as MO
import CheckPetsc4py as CP
import ExactSol
import Solver as S
import MHDmatrixPrecondSetup as PrecondSetup
import NSprecondSetup
import MHDpreconditioner


def TensorMass(b_k,params,FS):
    
    MO.PrintStr("Tensor Mass matrix construction",5,"=","\n","\n")
    b_t = TestFunction(FS)
    c_t = TrialFunction(FS)
    dim = FS.mesh().geometry().dim()
    def boundary(x, on_boundary):
        return on_boundary
    if dim == 2:
        u = Expression(("0.0","0.0"))
        mat =  as_matrix([[b_k[1]*b_k[1],-b_k[1]*b_k[0]],[-b_k[1]*b_k[0],b_k[0]*b_k[0]]])
    else:
        u = Expression(("0.0","0.0","0.0"))
        mat =  as_matrix([[b_k[2]*b_k[2]+b[1]*b[1],-b_k[1]*b_k[0],-b_k[0]*b_k[2]],
                          [-b_k[1]*b_k[0],b_k[0]*b_k[0]+b_k[2]*b_k[2],-b_k[2]*b_k[1]],
                          [-b_k[0]*b_k[2],-b_k[1]*b_k[2],b_k[0]*b_k[0]+b_k[1]*b_k[1]]])


    bc = DirichletBC(FS,u,boundary)
    if FS.dolfin_element().signature().find("Lagrange") == -1:
        tic()
#        a = params[2]*inner(grad(v), grad(u))*dx + inner((grad(u)*u_k),v)*dx +(1/2)*div(u_k)*inner(u,v)*dx - (1/2)*inner(u_k,n)*inner(u,v)*ds
        ShiftedMass = assemble(a + params[0]*params[0]/params[2]*inner(mat*b_t,c_t)*dx)
        bc.apply(ShiftedMass)
        print ("{:40}").format("Magnetic mass construction, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    else:
        tic()
        ShiftedMass = assemble(params[0]/params[1]*inner(mat*b_t,c_t)*dx)
        bc.apply(ShiftedMass)
        print ("{:40}").format("Magnetic mass construction, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    
    print "\n\n"
    FF = CP.Assemble(ShiftedMass)
    return FF




