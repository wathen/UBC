import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
from scipy.sparse import identity, bmat
import scipy.io

def IndexSet(W,type = 'standard'):
    if type == 'standard':
        if str(W.__class__).find('MixedFunctionSpace') != -1:
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

def BlockPerm(ordering, FS):
    Nu = FS['Velcity'].dim()
    Np = FS['Pressure'].dim()
    Nb = FS['Magnetic'].dim()
    Nr = FS['Multiplier'].dim()
    u = identity(Nu)
    p = identity(Np)
    b = identity(Nb)
    r = identity(Nr)

    if ordering == ['u', 'b', 'p', 'r']:
        A = bmat([[u, None, None, None],
                [None, None, p, None]
                [None, b, None, None]
                [None, None, None, r]])
    # elif ordering == ['b', 'u', 'p', 'r']:
    #     A = bmat([[None, u, None, None],
    #             [b, None, None, None]
    #             [None, None, p, None]
    #             [None, None, None, r]])
    # elif ordering == ['b', 'u', 'r', 'p']:
    #     A = bmat([[None, u, None, None],
    #             [b, None, None, None]
    #             [None, None, None, r]
    #             [None, None, p, None]])
    # elif ordering == ['u', 'b', 'r', 'p']:
    #     A = bmat([[u, None, None, None],
    #             [None, None, b, r]
    #             [None, b, None, None]
    #             [None, None, p, None]])


    return 1

def StoreMatrix(A,name):
    sA = A
    test ="".join([name,".mat"])
    scipy.io.savemat( test, {name: sA},oned_as='row')
