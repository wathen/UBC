from dolfin import has_petsc4py, parameters, as_backend_type
from PETScIO import arrayToVec
import petsc4py
import sys
petsc4py.init(sys.argv)
import petsc4py.PETSc as PETSc
import scipy.sparse as sp


def ParameterSetup(parameters, backend):
    parameters['linear_algebra_backend'] = backend


def Assemble(AA, bb=None):

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
            A = PETSc.Mat().createAIJ(size=(AA.size(0), AA.size(1)), csr=(
                row.astype('int32'), col.astype('int32'), value))
        else:
            A = arrayToVec(AA.array())
        if bb:
            b = arrayToVec(bb.array())
    if bb:
        return A, b
    else:
        return A


def Scipy2PETSc(A):
    A = A.tocsr()
    return PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))


def PETSc2Scipy(A):
    row, col, value = A.getValuesCSR()
    return sp.csr_matrix((value, col, row), shape=A.size)
