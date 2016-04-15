from Maxwell import *
from dolfin import *
from numpy import *
import scipy as Sci
import scipy.linalg
from math import pi,sin,cos,sqrt
import scipy.sparse as sps
import scipy.io as save
import scipy
import pdb
from matrix2latex import *
from matplotlib.pylab import *


Mcycle = 8
n= 2

time = zeros((Mcycle,1))
N = zeros((Mcycle,1))
DoF = zeros((Mcycle,1))
table = zeros((Mcycle,4))
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
for i in xrange(1,Mcycle+1):
    print "Cycle # = ",i,"\n"
    n = 2*n
    N[i-1,0] = n
    mesh = MeshGenerator(n)
    V,u,v = CreateTrialTestFuncs(mesh)
    u0 = Expression(('0','0'))
    f=Expression(("2+x[1]*(1-x[1])","2+x[0]*(1-x[0])"))
    tic()
    A,b =  AssembleSystem(V,u,v,f,u0,1,"PETSc")
    time[i-1,0] = toc()
    DoF[i-1,0] = b.size()
    # u = SolveSystem(A,b,V,"cg","amg",1e-6,1e-6,1)
    # ue = Expression(("x[1]*(1-x[1])","x[0]*(1-x[0])"))
    # error = Error(u,ue)
    table[i-1,0] = i-1
    table[i-1,1] = n
    table[i-1,2] = b.size()
    table[i-1,3] = time[i-1,0]


print matrix2latex(table)
loglog(DoF,time)
show()