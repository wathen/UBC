from dolfin import *
import ipdb

parameters['linear_algebra_backend'] = "Epetra"

# Create mesh and define function space
mesh = UnitCubeMesh(128,128,128)
tic()
V =VectorFunctionSpace(mesh, "CG", 1 )
print 'time to create function spaces',toc(),'\n\n'
# Define test and trial functions
v = TestFunction(V)
u = TrialFunction(V)

def boundary(x, on_boundary):
    return on_boundary


u0 = Expression(('0','0','0'))
# # p0 = ('0')

bcs = DirichletBC(V,u0, boundary)

# Define normal component, mesh size and right-hand side
# f = Expression(('- 2*(x[1]*x[1]-x[1])*-2*(x[0]*x[0]-x[0])','-2*(x[0]*x[0]-x[0]) - 2*(x[1]*x[1]-x[1])'))
f = Expression(('- 2*(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])-2*(x[0]*x[0]-x[0])*(x[2]*x[2]-x[2])-2*(x[2]*x[2]-x[2])*(x[2]*x[2]-x[2])', \
    '- 2*(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])-2*(x[0]*x[0]-x[0])*(x[2]*x[2]-x[2])-2*(x[2]*x[2]-x[2])*(x[2]*x[2]-x[2])', \
    '- 2*(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])-2*(x[0]*x[0]-x[0])*(x[2]*x[2]-x[2])-2*(x[2]*x[2]-x[2])*(x[2]*x[2]-x[2])'))


# Define parameters


# Define variational problem
tic()
a = inner(grad(v), grad(u))*dx
L = inner(v,f)*dx
AA,bb = assemble_system(a,L,bcs)
print 'time to creat linear system',toc(),'\n\n'

# Compute solution
u = Function(V)
tic()
set_log_level(PROGRESS)
solver = KrylovSolver("cg","ml_amg")
solver.parameters["relative_tolerance"] = 1e-10
solver.parameters["absolute_tolerance"] = 1e-10
solver.solve(AA,u.vector(),bb)
set_log_level(PROGRESS)


print 'time to solve linear system', toc(),'\n\n'
ue = Expression(('x[0]*x[1]*x[2]*(x[1]-1)*(x[2]-1)*(x[0]-1)','x[0]*x[1]*x[2]*(x[1]-1)*(x[2]-1)*(x[0]-1)','x[0]*x[1]*x[2]*(x[1]-1)*(x[2]-1)*(x[0]-1)'))
erru = ue- Function(V,u)

print sqrt(assemble(dolfin.inner(erru,erru)*dx))

# Plot solution
# plot(u)
# plot(interpolate(ue,V))

# interactive()
