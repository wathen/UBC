
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
from dolfin import *
# from MatrixOperations import *
import numpy as np
#import matplotlib.pylab as plt

from scipy.sparse import coo_matrix, csr_matrix, spdiags, bmat
import os, inspect
from HiptmairSetup import BoundaryEdge
import matplotlib
import CheckPetsc4py as CP
import MatrixOperations as MO
import time
import PETScIO as IO
import MHDmulti

def BoundaryIndices(mesh):

    
    dim = mesh.geometry().dim()

    if dim == 3:
        EdgeBoundary = BoundaryEdge(mesh)
        EdgeBoundary = np.sort(EdgeBoundary)[::2]
    else:
        B = BoundaryMesh(mesh,"exterior",False)
        EdgeBoundary = B.entity_map(1).array()

    MagneticBoundary = np.ones(mesh.num_edges())
    MagneticBoundary[EdgeBoundary.astype("int32")] = 0
    Magnetic = spdiags(MagneticBoundary,0,mesh.num_edges(),mesh.num_edges())

    B = BoundaryMesh(mesh,"exterior",False)
    NodalBoundary = B.entity_map(0).array()#.astype("int","C")
    LagrangeBoundary = np.ones(mesh.num_vertices())
    LagrangeBoundary[NodalBoundary] = 0
    Lagrange = spdiags(LagrangeBoundary,0,mesh.num_vertices(),mesh.num_vertices())
    
    if dim == 3:
        VelocityBoundary = np.concatenate((LagrangeBoundary,LagrangeBoundary,LagrangeBoundary),axis=1)
    else:
        VelocityBoundary = np.concatenate((LagrangeBoundary,LagrangeBoundary),axis=1)
    Velocity = spdiags(VelocityBoundary,0,dim*mesh.num_vertices(),dim*mesh.num_vertices())

    return [Velocity, Magnetic, Lagrange]
    



def Assemble(W, NS, Maxwell, Couple, L_ns, L_m, RHSform, BC, Type, IterType):
    
    tic() 
    if Type == 'NonLinear':
        F = assemble(NS[0])
        BC[0].apply(F)
        F = F.sparray()
        if IterType == 'Full':
            C = assemble(Couple[0])
            C = BC[4]*C.sparray()*BC[3]
        else:
            C = None
        if RHSform == 0:
            bu = assemble(L_ns)
            bp = Function(W[1]).vector() 
            print bp.array()
            bb = assemble(L_m)
            br = Function(W[3]).vector()
            BC[0].apply(bu)
            BC[1].apply(bb)
            BC[2].apply(br)
        else:
            bu = assemble(L_ns-RHSform[0])
            bp = assemble(-RHSform[1])
            bb = assemble(L_m-RHSform[2])
            br = assemble(-RHSform[3])
            BC[0].apply(bu)
            BC[1].apply(bb)
            BC[2].apply(br)
        b = np.concatenate((bu.array(),bp.array(),bb.array(),br.array()),axis = 0)

        MO.StrTimePrint("MHD non-linear matrix assembled, time: ",toc())
        return [F, C],b
    elif Type == 'Linear':

        M = assemble(Maxwell[0])
        D = assemble(Maxwell[2])
        SS = assemble(Maxwell[3])

        B = assemble(NS[2])
        S = assemble(NS[3])


        SS = 0*SS
        BC[1].apply(M)
        BC[2].apply(SS)  

        
        B = B.sparray()*BC[3]
        S = S.sparray()

        M = M.sparray()
        D = BC[4]*D.sparray()*BC[5]
        SS = SS.sparray()
        
        MO.StrTimePrint("MHD linear matrix assembled, time: ",toc())
        return [B,M,D,S,SS]
    else:
        bu = assemble(L_ns-RHSform[0])
        bp = assemble(-RHSform[1])
        bb = assemble(L_m-RHSform[2])
        br = assemble(-RHSform[3])
        BC[0].apply(bu)
        BC[1].apply(bb)
        BC[2].apply(br)
        b = np.concatenate((bu.array(),bp.array(),bb.array(),br.array()),axis = 0)
        return IO.arrayToVec(b)

   

def SystemAssemble(W,A,b,SetupType,IterType):
    tic()
    if SetupType == 'Matrix':
        if IterType == 'Full':
            A = CP.Scipy2PETSc(bmat([[A[0],A[2].T,-A[1].T,None],
                      [A[2],A[5],None,None],
                      [A[1],None,A[3],A[4]],
                      [None,None,A[4].T,A[6]]]))
        else:
            A = [CP.Scipy2PETSc(bmat([[A[0],A[2].T],
                      [A[2],A[5]]])),CP.Scipy2PETSc(bmat([[A[3],A[4]],
                      [A[4].T,A[6]]]))]
        b = IO.arrayToVec(b)
        MO.StrTimePrint("MHD system assemble, time: ",toc()) 
        return A,b
    else:
        for i in range(len(A)):
            if A[i] != None:
                A[i] = CP.Scipy2PETSc(A[i])
        if IterType == 'Full':
            P = PETSc.Mat().createPython([W[0].dim()+W[1].dim()+W[2].dim()+W[3].dim(),W[0].dim()+W[1].dim()+W[2].dim()+W[3].dim()])
            P.setType('python')
            p = MHDmulti.MHDmat(W,A)
            P.setPythonContext(p)
        else: 
            MatFluid = PETSc.Mat().createPython([W[0].dim()+W[1].dim(), W[0].dim()+W[1].dim()])
            MatFluid.setType('python')
            pFluid = MHDmulti.MatFluid([W[0],W[1]],A)
            MatFluid.setPythonContext(pFluid)
            
            MatMag = PETSc.Mat().createPython([W[2].dim()+W[3].dim(), W[2].dim()+W[3].dim()])
            MatMag.setType('python')
            pMag = MHDmulti.MatMag([W[2],W[3]],A)
            MatMag.setPythonContext(pMag)
            P = [MatFluid,MatMag]
        b = IO.arrayToVec(b)
        MO.StrTimePrint("MHD mult-class setup, time: ",toc())
        return P,b





