from dolfin import *

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Defing boundaries
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)

# Initialize sub-domain instances
left = Left()
top = Top()
right = Right()
bottom = Bottom()

# Initialize mesh function for boundary domains
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
left.mark(boundaries, 1)
top.mark(boundaries, 2)
right.mark(boundaries, 1)
bottom.mark(boundaries, 2)

# Define Diriclet boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundaries,1)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("(pow(x[0]-0.5,2)-4*pow(x[1]-0.5,4))")
g = Expression("(5*x[0])")
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds(2)

# Compute solution
u = Function(V)
solve(a == L, u, bc, solver_parameters=dict(linear_solver="cg",
                             preconditioner="amg"))

u = Function(V)
A, b = assemble_system(a,L,bc)
solve(A,u.vector(),b)

# Plot solution
plot(u, interactive=True)