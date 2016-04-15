from dolfin import has_petsc4py, parameters, as_backend_type, Function, assemble, dx
import petsc4py
import sys
petsc4py.init(sys.argv)
import petsc4py.PETSc as PETSc
from PackageName.GeneralFunc import common
import scipy.sparse as sp

def PETScMultiDuplications(b,num):
    A = [0]*num
    for i in range(num):
        A[i] = b.duplicate()

    return A
def Assemble(AA,bb=None):
    backend = AA.str(False)


    if AA.__str__().find('uBLAS') == -1:
        if AA.size(1) == AA.size(0):
            A = as_backend_type(AA).mat()
        else:
            A = as_backend_type(AA).vec()
        if bb:
            b = as_backend_type(bb).vec()
    else:
        if AA.size(1) == AA.size(0):
            As = AA.sparray()
            As.eliminate_zeros()
            row, col, value = As.indptr, As.indices, As.data
            A = PETSc.Mat().createAIJ(size=(AA.size(0),AA.size(1)),csr=(row.astype('int32'), col.astype('int32'), value))
        else:
            A = arrayToVec(AA.array())
        if bb:
            b = arrayToVec(bb.array())
    if bb:
        return A,b
    else:
        return A

def Scipy2PETSc(A):
    A = A.tocsr()
    return PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))

def PETSc2Scipy(A):
    row, col, value = A.getValuesCSR()
    return sp.csr_matrix((value, col, row), shape=A.size)

def arrayToVec(vecArray):

    vec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    vec.setSizes(len(vecArray))
    vec.setUp()
    (Istart,Iend) = vec.getOwnershipRange()
    return vec.createWithArray(vecArray[Istart:Iend],
            comm=PETSc.COMM_WORLD)
    vec.destroy()

def PETScToNLiter(x,FS):
    n = len(FS)
    IS = common.IndexSet(FS)
    u = {}
    for i in range(n):
        v = Function(FS.values()[i])
        v.vector()[:] = x.getSubVector(IS.values()[i]).array
        if FS.keys()[i] == 'Pressure':
            ones = Function(FS['Pressure'])
            ones.vector()[:] = 1
            vv = Function(FS['Pressure'])
            vv.vector()[:] = v.vector().array() - assemble(v*dx)/assemble(ones*dx)
            u[FS.keys()[i]] = vv
        else:
            u[FS.keys()[i]] = v
    return u
