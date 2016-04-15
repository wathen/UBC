import PyTrilinos.ML as ml
from FullMaxwell import *
from dolfin import *
from numpy import *
from block import *
from block.iterative import *
import block.algebraic.trilinos as bat
import scipy as Sci
#import scipy.linalg
from math import pi,sin,cos,sqrt
import scipy.sparse as sps
import scipy.io as save
import scipy
import ipdb
import threading


print "FEniCS routine to solve Maxwell subproblem \n \n"


print "Create mesh.... \n "
Max = FMaxwell(32)
mesh = Max.mesh

print "Creating Function space with trial and test functions.... \n "
V,Q,u,p,v,q= Max.CreateTrialTestFuncs(mesh)

# Define the boundary condition
u0 = Expression(('0','0'))
p0 = Expression('0')


f=Expression(("2+x[1]*(1-x[1])","2+x[0]*(1-x[0])"))

print "Assemble SysteMax.... \n "

lock = threading.Lock()
lock.acquire()
AA,bb,P=  Max.AssembleSystem(V,Q,u,p,v,q,f,u0,p0,0)
lock.release()



[[A,B],[C,_]] = AA
[B1,B2] =bb
[[P11,_],[_,P22]] =P
# ML_Hiptmair(P11

MLList = {
    "max levels"                                     : 1,
    "prec type"                                        : "MGV",
    "increasing or decreasing"               : "decreasing",

    "aggregation: type"                          : "Uncoupled-MIS",
    "aggregation: damping factor"         : 4.0/3.0,

    "eigen-analysis: type"                      : "cg",
    "eigen-analysis: iterations"              : 10,

    "smoother: sweeps"                          : 1,
    "smoother: damping factor"              : 1.0,
    "smoother: pre or post"                     : "both",
    "smoother: type"                               : "Hiptmair",

    "subsmoother: type"                         : "Chebyshev",
    "subsmoother: Chebyshev alpha"    : 27.0,
    "subsmoother: node sweeps"           : 4,
    "subsmoother: edge sweeps"           : 4,

    "coarse: type"                                   : "Amesos-KLU",
    "coarse: max size"                           : 128

}

ML_Hiptmair = ml.MultiLevelPreconditioner(P11.down_cast().mat(),B.down_cast().mat(),P22.down_cast().mat(),MLList)
# ML_Hiptmair = ml.MultiLevelPreconditioner(P11.down_cast().mat(),MLList)
ML_Hiptmair.ComputePreconditioner()

DoF =B1.array().size + B2.array().size
PP = block_mat([[ConjGrad(P11,precond=bat.ML(P11),tolerance=1e-10,maxiter=50000,show=1),0],[0,ConjGrad(P22,precond=bat.ML(P22),tolerance=1e-10,maxiter=50000,show=1)]])

# PP = block_mat([[(P11),0],[0,MinRes(P22,precond=block.algebraic.trilinos.ML(P22),tolerance=1e-6,maxiter=50000,show=1)]])
print "DoF:",DoF


# PP = block_mat([[(P11),0],[0,(P22)]])
# Max.minres(AA, bb,0*bb, PP, 1e-10,50000,1e-8)

AAinv = MinRes(AA, precond = PP, tolerance=1e-8,maxiter=50000)
uu, pp = AAinv * bb

ue = Expression(("x[1]*(1-x[1])","x[0]*(1-x[0])"))
pe = Expression('0')

erru = ue - Function(V,uu)
errp = pe - Function(Q,pp)

print sqrt(assemble(dolfin.inner(erru,erru)*dx))
print sqrt(assemble(dolfin.inner(errp,errp)*dx))


# print AAinv
# print uu
# print pp
# plot (Function(V,uu))
# plot (Function(Q,pp))

# interactive()
# Max.StoreSystem(A,b)

# AAinv = LGMRES(A,  tolerance=1e-5, maxiter=50, show=2)
# Solve SysteMax.... "
# u = Max.SolveSystem(A,b,V,"cg","amg",1e-6,1e-6,1)

# # Defining exact solution
# ue = Expression(("x[1]*(1-x[1])","x[0]*(1-x[0])"))

# print "\nCalculating error"
# error = Max.Error(u,ue)

# print error


# A = {'hi': 1, 'two': 2}
# A['hi'] =
