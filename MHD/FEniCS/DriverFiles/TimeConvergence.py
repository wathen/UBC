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

print "FEniCS routine to solve Maxwell subproblem \n \n"


Mcycle = 10
n = 2

time = zeros((Mcycle,1))
N = zeros((Mcycle,1))
DoF = zeros((Mcycle,1))
cycle = zeros((Mcycle,1))
table = zeros((Mcycle,4))

for j in xrange(1,Mcycle+1):
    print "cycle #",j-1,"\n"
    n = 2*n
    M = Maxwell(n)
    mesh = M.mesh
    N[j-1,0] = n
    tic()
    V,u,v= M.CreateTrialTestFuncs(mesh)
    time[j-1,0] = toc()
    u0 = Expression(('0','0'))
    f=Expression(("2+x[1]*(1-x[1])","2+x[0]*(1-x[0])"))
    A,b =  M.AssembleSystem(V,u,v,f,u0,1,"PETSc")
    DoF[j-1,0] = b.size()
    cycle[j-1,0] = j-1
    table[j-1,:] = [j-1,n,b.size(),time[j-1,0]]

print matrix2latex(table)

