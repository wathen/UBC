import PyTrilinos.ML as ml
# import PyTrilinos.AztecOO as AztecOO
from Maxwell import *
from FullMaxwell import *
from dolfin import *
from numpy import *
import scipy as Sci
#import scipy.linalg
from math import pi,sin,cos,sqrt
import scipy.sparse as sps
import scipy.io as save
import scipy
import ipdb



print "FEniCS routine to solve Maxwell subproblem \n \n"
n = 1000
FM = FMaxwell(n)
print "Create mesh.... \n "
M = Maxwell(n)
mesh = M.mesh

print "Creating Function space with trial and test functions.... \n "
VV,u,v= M.CreateTrialTestFuncs(mesh)

# Define the boundary condition
u0 = Expression(('0','0'))

f=Expression(("2+x[1]*(1-x[1])","2+x[0]*(1-x[0])"))

print "Assemble System.... \n "
tic()
AAA,b =  M.AssembleSystem(VV,u,v,f,u0,-1,"Epetra")
print 'time to create system ',toc(),' size of system ',b.size()
# Solve System.... "
parameters.linear_algebra_backend = "Epetra"
# u = M.SolveSystem(A,b,V,"cg","amg",1e-6,1e-6,1)
# MLList = {
#     "default values" : "maxwell",
#     "max levels"                                     : 10,
#     "prec type"                                        : "MGV",
#     "increasing or decreasing"               : "decreasing",
#     "aggregation: type"                          : "Uncoupled-MIS",
#     "aggregation: damping factor"         : 4.0/3.0,
#     "eigen-analysis: type"                      : "cg",
#     "eigen-analysis: iterations"              : 10,
#     "smoother: sweeps"                          : 1,
#     "smoother: damping factor"              : 1.0,
#     "smoother: pre or post"                     : "both",
#     "smoother: type"                               : "Hiptmair",
#     "subsmoother: type"                         : "Chebyshev",
#     "subsmoother: Chebyshev alpha"    : 27.0,
#     "subsmoother: node sweeps"           : 4,
#     "subsmoother: edge sweeps"           : 4,
#     "PDE equations" : 1,
#     "coarse: type"                                   : "Amesos-MUMPS",
#     "coarse: max size"                           : 128

# }

MLList = {

    "default values" : "maxwell",
    "max levels" : 1,
    "output" : 10,
    "PDE equations" : 1,
    "increasing or decreasing" : "decreasing",
    "aggregation: type" : "Uncoupled-MIS",
    "aggregation: damping factor" : 1.3333,
    "coarse: max size" : 75,
    "aggregation: threshold" : 0.0,
    "smoother: sweeps" : 2,
    "smoother: damping factor" : 0.67,
    "smoother: type" : "MLS",
    "smoother: MLS polynomial order" : 4,
    "smoother: pre or post" : "both",
    "coarse: type" : "Amesos-KLU",
    "prec type" : "MGV",
    "print unused" : -2
}


#Q = FunctionSpace(mesh, "CG", 1)
#p = TestFunction(Q)
#B = dolfin.inner(grad(p),u)*dx
#BB = assemble(B)
#bc.apply(BB)
tic()
V,Q,u,p,v,q= FM.CreateTrialTestFuncs(mesh)
print 'time to create function spaces  ',toc()
# Define the boundary condition
u0 = Expression(('0','0'))
p0 = Expression('0')


f=Expression(("2+x[1]*(1-x[1])","2+x[0]*(1-x[0])"))

print "Assemble SysteMax.... \n "

tic()
AA,bb,P=  FM.AssembleSystem(V,Q,u,p,v,q,f,u0,p0,0.5)
print 'time to assemble saddle point system' , toc()


[[A,B],[C,_]] = AA
[B1,B2] =bb
[[P11,_],[_,P22]] =P

# tic()
# AA = AAA.down_cast().mat()
# FM.SaveEpertaMatrix(AA,"A")
# print toc()

# FM.StoreMatrix(b,"b")
tic()
ML_Hiptmair = ml.MultiLevelPreconditioner(P11.down_cast().mat(),B.down_cast().mat(),P22.down_cast().mat(),MLList)
# ML_Hiptmair = ml.MultiLevelPreconditioner(A.down_cast().mat(),MLList)
ML_Hiptmair.ComputePreconditioner()
x = Function(V)
print 'time to create preconditioner ', toc()
A_epetra = down_cast(AAA).mat()
b_epetra = down_cast(b).vec()
x_epetra = down_cast(x.vector()).vec()

tic()
#u = M.SolveSystem(A,b,V,"cg","amg",1e-6,1e-6,1)
print toc()

import PyTrilinos.AztecOO as AztecOO
solver = AztecOO.AztecOO(A_epetra, x_epetra, b_epetra)
solver.SetPrecOperator(ML_Hiptmair)
solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres);
solver.SetAztecOption(AztecOO.AZ_output, 50);
err = solver.Iterate(155000, 1e-10)


# # Defining exact solution
ue = Expression(("x[1]*(1-x[1])","x[0]*(1-x[0])"))

print "\nCalculating error"
#error = M.Error(u,ue)

#print error

#plot(u, interactive=True)
# A = {'hi': 1, 'two': 2}
# A['hi'] =



from PyTrilinos import Amesos, Epetra
xx = Function(V)
xx_epetra = down_cast(xx.vector()).vec()

problem = Epetra.LinearProblem(A_epetra,xx_epetra,b_epetra)
print '\n\n\n\n\n\n'
factory = Amesos.Factory()
solver = factory.Create("Amesos_Umfpack", problem)
amesosList = {"PrintTiming" : True, "PrintStatus" : True }
solver.SetParameters(amesosList)
solver.SymbolicFactorization()
solver.NumericFactorization()
solver.Solve()
soln = problem.GetLHS()
print "||x_computed||_2 =", soln.Norm2()
solver.PrintTiming()
print '\n\n\n\n\n\n'
