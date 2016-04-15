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


Mcycle = 8
n = 2

error = zeros((Mcycle,1))
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
    V,u,v= M.CreateTrialTestFuncs(mesh)
    u0 = Expression(('0','0'))
    f=Expression(("2+x[1]*(1-x[1])","2+x[0]*(1-x[0])"))
    A,b =  M.AssembleSystem(V,u,v,f,u0,1,"PETSc")
    # u = M.SolveSystem(A,b,V,"cg","icc",1e-6,1e-6,1)
    u = Function(V)
    solve(A,u.vector(),b,'mumps')
    ue = Expression(("x[1]*(1-x[1])","x[0]*(1-x[0])"))
    error[j-1,0] = M.Error(u,ue)
    DoF[j-1,0] = b.size()
    cycle[j-1,0] = j-1
    table[j-1,:] = [j-1,n,b.size(),error[j-1,0]]

print matrix2latex(table)

