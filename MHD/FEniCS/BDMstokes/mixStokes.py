import scipy
from MatrixOperations import *
import os
import cProfile
from dolfin import *
import numpy
import scipy.sparse as sps
import scipy.io as save
from PyTrilinos import ML,AztecOO, Epetra, Amesos

MO = MatrixOperations()
parameters['linear_algebra_backend'] = 'Epetra'

# mesh = UnitSquareMesh(16,16)
n = 16
mesh = RectangleMesh(-1, -1, 1, 1, n, n)
# parameters['reorder_dofs_serial'] = False
tic()
V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)
print 'time to create function spaces', toc(),'\n\n'
W = V*Q

def boundary(x, on_boundary):
    return on_boundary

u0 = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
# p0 = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")

# u0 = Expression(("0","0"))

bc = DirichletBC(W.sub(0),u0, boundary)
# bc1 = DirichletBC(W.sub(1), p0, boundary)
bcs = [bc]
# v, u = TestFunction(V), TrialFunction(V)
# q, p = TestFunction(Q), TrialFunction(Q)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = -Expression(('0','0'))
# f = -Expression(("- 8*x[0]*x[0]*(x[0] - 1)*(x[0] - 1)*(x[1] - 1) - 4*x[0]*x[0]*(2*x[1] - 1)*(x[0] - 1)**2 - 8*x[0]*x[1]*(x[0] - 1)*(x[0] - 1) - 4*x[1]*(2*x[1] - 1)*(x[0] - 1)*(x[0] - 1)*(x[1] - 1) - 4*x[0]*x[0]*x[1]*(2*x[1] - 1)*(x[1] - 1) - 8*x[0]*x[1]*(2*x[0] - 2)*(2*x[1] - 1)*(x[1] - 1)","- 8*x[1]*x[1]*(x[0] - 1)*(x[1] - 1)*(x[1] - 1) - 4*x[1]*x[1]*(2*x[0] - 1)*(x[1] - 1)*(x[1] - 1) - 8*x[0]*x[1]*x[1]*(x[1] - 1)*(x[1] - 1) - 4*x[0]*(2*x[0] - 1)*(x[0] - 1)*(x[1] - 1)*(x[1] - 1) - 4*x[0]*x[1]*x[1]*(2*x[0] - 1)*(x[0] - 1) - 8*x[0]*x[1]*(2*x[0] - 1)*(2*x[1] - 2)*(x[0] - 1)"))

u_k = Function(V)
mu = Constant(1e-0)

N = FacetNormal(mesh)
t = as_vector((-N[0], N[1]))
h = CellSize(mesh)
h_avg =avg(h)
alpha = 10.0
gamma =10.0


# constq = assemble(q*dx)
# q = q - constq

a11 = inner(grad(v), grad(u))*dx \
    - inner(avg(grad(v)), outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
    - inner(outer(v('+'),N('+'))+outer(v('-'),N('-')), avg(grad(u)))*dS \
    + alpha/h_avg*inner(outer(v('+'),N('+'))+outer(v('-'),N('-')),outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
    - inner(outer(v,N), grad(u))*ds \
    - inner(grad(v), outer(u,N))*ds \
    + gamma/h*inner(v,u)*ds
a12 = div(v)*p*dx
a21 = div(u)*q*dx
L1  =  inner(v,f)*dx + gamma/h*inner(u0,v)*ds - inner(grad(v),outer(u0,N))*ds
# L1  =  inner(v,f)*dx + gamma/h*inner(u0,t)*inner(v,t)*ds - inner(grad(v),outer(u0,N))*ds
a = -a11+a12+a21

i = p*q*dx
# AA = assemble(a11)

uu = Function(W)
tic()
# AA= block_assemble([[a11, a12],[a21,  0 ]], bcs=bcs)
# bb = block_assemble([L1, 0], bcs=bcs)
AA, bb = assemble_system(a, L1, bcs)
PP, btmp = assemble_system(i+a11, L1, bcs)
print 'time to create linear system', toc(),'\n\n'


# MO.SaveEpertaMatrix(AA.down_cast().mat(),"A")
# MO.SaveEpertaMatrix(P.down_cast().mat(),"P")

print "DoF ",W.dim()

# MO.StoreMatrix(bb,"rhs")
x_epetra = Epetra.Vector(0*bb.array())
A_epetra = down_cast(AA).mat()
P_epetra = down_cast(PP).mat()
b_epetra = down_cast(bb).vec()
# mlList = {"max levels"        : 200,
#       "output"            : 10,
#       "smoother: type"    : "symmetric Gauss-Seidel",
#       "aggregation: type" : "Uncoupled"
#       }

# prec = ML.MultiLevelPreconditioner(P_epetra, False)
# prec.SetParameterList(mlList)
# prec.ComputePreconditioner()

# solver = AztecOO.AztecOO(A_epetra, x_epetra, b_epetra)
# solver.SetPrecOperator(prec)
# solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres);
# solver.SetAztecOption(AztecOO.AZ_output, 100);
# err = solver.Iterate(20000, 1e-10)

tic()
problem = Epetra.LinearProblem(A_epetra,x_epetra,b_epetra)
print '\n\n\n\n\n\n'
factory = Amesos.Factory()
solver = factory.Create("Amesos_Umfpack", problem)
# solver = factory.Create("MUMPS", problem)
amesosList = {"PrintTiming" : True, "PrintStatus" : True }
solver.SetParameters(amesosList)
solver.SymbolicFactorization()
solver.NumericFactorization()
solver.Solve()
soln = problem.GetLHS()
print "||x_computed||_2 =", soln.Norm2()
solver.PrintTiming()
print '\n\n\n\n\n\n'






# solver = KrylovSolver("minres", "amg")

# # Associate operator (A) and preconditioner matrix (P)
# solver.set_operators(AA, P)

# Solve
# U = Function(W)
# solve(AA,U.vector(), bb)


# solve(a==L1,uu,bcs)
print 'time to solve linear system', toc(),'\n\n'


ue = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
pe = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")


# ue = Expression(("-2*x[0]*x[1]*(x[0]-1)*(x[1]-1)*x[0]*(x[0]-1)*(2*x[1]-1)","-2*x[0]*x[1]*(x[0]-1)*(x[1]-1)*x[1]*(x[1]-1)*(2*x[0]-1)"))
# pe = Expression("0")
# ua, pa = uu.split(True)
# int_p = pa * dx
# average_p = assemble ( int_p , mesh = mesh )
# p_array = pa.vector().array() - (average_p)
# pa.vector()[:] = p_array

# erru = ue - ua
# errp = pe - pa
Vdim = V.dim()
pp = x_epetra[Vdim:]
pa = Function(Q)
pa1 = Function(Q)
pa2 = Function(Q)
pa1.vector()[:] = pp.array
pa2.vector()[:] = 0*pp.array+1
pa2.vector().array()
pa.vector()[:] = pp.array + assemble(pa1*dx)/assemble(pa2*dx)

uu = x_epetra[0:Vdim]
ua = Function(V)
ua.vector()[:] = uu.array

# print sqrt(assemble(dolfin.inner(erru,erru)*dx))
# print sqrt(assemble(dolfin.inner(errp,errp)*dx))



erru= errornorm(ue,ua,norm_type="L2", degree_rise=4,mesh=mesh)
errp= errornorm(pe,pa,norm_type="L2", degree_rise=4,mesh=mesh)


print erru
print errp


# Plot solution
plot(ua)
plot(pa)
interactive()
