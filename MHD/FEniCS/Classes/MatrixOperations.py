
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc

from dolfin import *
from numpy import *

import scipy as Sci
#import scipy.linalg
from math import pi,sin,cos,sqrt
import scipy.sparse as sps
import scipy
import pandas as pd
#import petsc4py
#import sys
#petsc4py.init(sys.argv)
#from petsc4py import PETSc
import numpy as np
#import PETScIO as IO
import scipy.io
import time

def shift(A,shift):
    n = A.size[0]
    ones = np.ones(n)
    ones = IO.arrayToVec(ones)
    Identity = PETSc.Mat()
    Identity.create()
    Identity.setSizes([n, n])
    Identity.setUp()
    Identity.assemble()
    Identity.setDiagonal(ones)
    Aout = A + shift*Identity
    return Aout

def StoreMatrix(A,name):
    sA = A
    test ="".join([name,".mat"])
    scipy.io.savemat( test, {name: sA},oned_as='row')

def SaveEpertaMatrix(A,name):
    from PyTrilinos import EpetraExt
    from numpy import array
    import scipy.sparse as sps
    import scipy.io
    test ="".join([name,".txt"])
    EpetraExt.RowMatrixToMatlabFile(test,A)
    data = loadtxt(test)
    col,row,values = data[:,0]-1,data[:,1]-1,data[:,2]
    Asparse = sps.csr_matrix((values, (row, col)))
    testmat ="".join([name,".mat"])
    scipy.io.savemat( testmat, {name: Asparse},oned_as='row')

def LabelStucture( Values,Label):
    for x,y in zip(Values,Label):
        d.setdefault(y, []).append(x)
    return d

def PandasFormat(table,field,format):
    table[field] = table[field].map(lambda x: format %x)
    return table


def  PrintStr(string,indent,boarder,preLines="",postLines=""):
    AppendedString  = ""
    for i in range(indent):
        AppendedString = " "+AppendedString

    StringPrint = AppendedString+string
    if indent < 2:
        Outerboarder = ""
        if indent == 1:
            for i in range(len(string)+indent+1):
                Outerboarder += boarder
        else:
            for i in range(len(string)+indent):
                Outerboarder += boarder

    else:
        AppendedString  = ""
        for i in range(indent-2):
            AppendedString = " "+AppendedString
        Outerboarder = AppendedString
        for i in range(len(string)+4):
            Outerboarder += boarder
    print preLines+Outerboarder
    print StringPrint
    print Outerboarder+postLines

    def IndexSet(W):
        print W
        if str(W.__class__).find('list') == -1:
            n = W.num_sub_spaces()
            IS = [0]*n
            Start = 0
            End = W.sub(0).dim()
            for i in range(n):
                if i>0:
                    Start += W.sub(i-1).dim()
                    End += W.sub(i).dim()
                IS[i] = PETSc.IS().createGeneral(range(Start,End))
        else:
            n = len(W)
            IS = [0]*n
            Start = 0
            End = W[0].dim()
            for i in range(n):
                if i>0:
                    Start += W[i-1].dim()
                    End += W[i].dim()
                IS[i] = PETSc.IS().createGeneral(range(Start,End))
        return IS



def IndexSet(W,type = 'standard'):
    if type == 'standard':
        if str(W.__class__).find('dolfin.functions.functionspace.FunctionSpace') != -1:
            n = W.num_sub_spaces()
            IS = [0]*n
            Start = 0
            End = W.sub(0).dim()
            for i in range(n):
                if i>0:
                    Start += W.sub(i-1).dim()
                    End += W.sub(i).dim()
                IS[i] = PETSc.IS().createGeneral(range(Start,End))
        elif str(W.__class__).find('dict') != -1:
            n = len(W)
            IS = {}
            Start = 0
            End = W.values()[0].dim()
            for i in range(n):
                if i>0:
                    Start += W.values()[i-1].dim()
                    End += W.values()[i].dim()
                IS[W.keys()[i]] = PETSc.IS().createGeneral(range(Start,End))
        else:
            n = len(W)
            IS = [0]*n
            Start = 0
            End = W[0].dim()
            for i in range(n):
                if i>0:
                    Start += W[i-1].dim()
                    End += W[i].dim()
                IS[i] = PETSc.IS().createGeneral(range(Start,End))
        return IS
    else:
        if str(W.__class__).find('list') == -1:
            IS = [0]*2
            IS[0] =  PETSc.IS().createGeneral(range(0,W.sub(0).dim()+W.sub(1).dim()))
            IS[1] =  PETSc.IS().createGeneral(range(W.sub(0).dim()+W.sub(1).dim(),W.sub(0).dim()+W.sub(1).dim()+W.sub(2).dim()+W.sub(3).dim()))


        else:
            IS = [0]*2
            IS[0] =  PETSc.IS().createGeneral(range(0,W[0].dim()+W[1].dim()))
            IS[1] =  PETSc.IS().createGeneral(range(W[0].dim()+W[1].dim(),W[0].dim()+W[1].dim()+W[2].dim()+W[3].dim()))
        return IS

def PETScMultiDuplications(b,num):
    A = [0]*num
    for i in range(num):
        A[i] = b.duplicate()

    return A



def StrTimePrint(String,EndTime):
    print ("{:40}").format(String), " ==>  ",("{:4f}").format(EndTime),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
