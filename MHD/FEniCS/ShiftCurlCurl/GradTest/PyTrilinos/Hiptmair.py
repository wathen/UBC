import scipy.io
from scipy2Trilinos import scipy_csr_matrix2CrsMatrix
from PyTrilinos import Epetra, ML, AztecOO
import TrilinosIO


System = scipy.io.loadmat("System16.mat")


MLList = {
  "default values":"maxwell",
  "max levels":10,
  "prec type":"MGV",
  "increasing or decreasing":"decreasing",

  "aggregation: type":"Uncoupled-MIS",
  "aggregation: damping factor":1.333,
  "eigen-analysis: type":"cg",
  "eigen-analysis: iterations":10,
  "aggregation: edge prolongator drop threshold":0.0,

  "smoother: sweeps":1,
  "smoother: damping factor":1.0,
  "smoother: pre or post":"both",
  "smoother: type (level 1)":"Hiptmair",
  "smoother: type (level 2)":"Hiptmair",
  "smoother: type (level 3)":"Hiptmair",
  "smoother: type (level 4)":"Hiptmair",
  "smoother: type (level 5)":"Hiptmair",
  "smoother: type (level 6)":"Hiptmair",

  "smoother: Hiptmair efficient symmetric":True,
  "subsmoother: type": "Chebyshev",
  "subsmoother: Chebyshev alpha": 27.0,
  "subsmoother: node sweeps":4,
  "subsmoother: edge sweeps":4,

  "coarse: type":"Amesos-KLU",
  "coarse: max size":128,
  "coarse: pre or post":"post",
  "coarse: sweeps":1

}

comm = Epetra.PyComm()
C = scipy_csr_matrix2CrsMatrix(System["C"].tocsr(), comm)
CurlCurl = scipy_csr_matrix2CrsMatrix(System["CurlCurl"].tocsr(), comm)
node = scipy_csr_matrix2CrsMatrix(System["node"].tocsr(), comm)

ML_Hiptmair = ML.MultiLevelPreconditioner(CurlCurl,C,node,MLList)
ML_Hiptmair.ComputePreconditioner()
x = System["rhs"][0]
b_epetra = TrilinosIO._numpyToTrilinosVector(x)
x_epetra = TrilinosIO._numpyToTrilinosVector(System["rhs"]*0)

solver = AztecOO.AztecOO(CurlCurl, x_epetra, b_epetra)
solver.SetPrecOperator(ML_Hiptmair)
solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_cg);
solver.SetAztecOption(AztecOO.AZ_output, 50);
err = solver.Iterate(155000, 1e-10)


