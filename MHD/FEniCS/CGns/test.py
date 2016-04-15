from petsc4py import PETSc
from PyTrilinos import EpetraExt, Epetra
from numpy import array,loadtxt
import scipy.sparse as sps
import scipy.io
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
import ipdb
from dolfin import *
import matplotlib.pylab as plt
import PETScIO as io
# data = loadtxt("A_epetra.txt")
# col,row,values = data[:,0]-1,data[:,1]-1,data[:,2]
n = 1e6
A = scipy.sparse.rand(n, n, density=1/(n**1.4), format='csr')

# plt.spy(A)
# plt.show()
comm = Epetra.PyComm()
tic()
As = scipy_csr_matrix2CrsMatrix(A, comm)
print toc()
tic()
Ap = PETSc.Mat().createAIJ(size=A.shape,csr=(A.indptr, A.indices, A.data))
# Ap = io.arrayToMat(As)
print toc()

tic
Anew = io.matToSparse(Ap)
# from petsc4py import PETSc as _PETSc
# data = As.getValuesCSR()
# (Istart,Iend) = As.getOwnershipRange()
# columns = As.getSize()[0]
# sparseSubMat = sps.csr_matrix(data[::-1],shape=(Iend-Istart,columns))
# comm = _PETSc.COMM_WORLD

# sparseSubMat = comm.tompi4py().allgather(sparseSubMat)
# A = sps.vstack(sparseSubMat)
print toc()

plt.spy(A)
plt.show()