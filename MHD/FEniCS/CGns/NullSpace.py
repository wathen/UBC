from PyTrilinos import EpetraExt, Epetra
from numpy import array,loadtxt
import scipy.sparse as sps
import scipy.io
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
import ipdb
from dolfin import *
tic()
data = loadtxt("A_epetra.txt")
print toc()
tic()
col,row,values = data[:,0]-1,data[:,1]-1,data[:,2]
print toc()
tic()
A = sps.csr_matrix((values, (row, col)))
print toc()
tic()
comm = Epetra.PyComm()
print toc()
tic()
A = scipy_csr_matrix2CrsMatrix(A, comm)
print toc()
zz