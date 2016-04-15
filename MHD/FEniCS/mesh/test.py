from dolfin import *
import ipdb
# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# Load mesh from file
mesh = Mesh()

domain_vertices = [Point(1.0, -1.0),
                   Point(6.0, -1.0),
                   Point(6.0, 1.0),
                   Point(1.0, 1.0),
                   Point(1.0, 6.0),
                   Point(-1.0, 6.0),
                   Point(-1.0, 1.0),
                   Point(-6.0, 1.0),
                   Point(-6.0, -1.0),
                   Point(-1.0, -1.0),
                   Point(-1.0, -6.0),
                   Point(1.0, -6.0),
                   Point(1.0,-1.0),
                   Point(1.0, -1.0)]

# Generate mesh and plot
PolygonalMeshGenerator.generate(mesh, domain_vertices, 0.75);


cell_markers = CellFunction("bool", mesh)
cell_markers.set_all(False)
origin = Point(0.0, 0.0)

for cell in cells(mesh):
  p = cell.midpoint()
  # print p
  if p.distance(origin) < 2:
      cell_markers[cell] = True


mesh = refine(mesh, cell_markers)


cell_markers = CellFunction("bool", mesh)
cell_markers.set_all(False)
origin = Point(0.0, 0.0)

for cell in cells(mesh):
  p = cell.midpoint()
  # print p
  if p.distance(origin) < 1:
      cell_markers[cell] = True




mesh = refine(mesh, cell_markers)

class noflow1(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-6.0,-1.0)) and near(x[0],1.0)


class noflow4(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (1.0, 6.0)) and near(x[0],1.0)


class noflow5(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-6.0,-1.0)) and near(x[0],-1.0)


class noflow8(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (1.0, 6.0)) and near(x[0],-1.0)


class noflow2(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (1.0,6.0)) and near(x[1],-1.0)


class noflow3(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (1.0,6.0)) and near(x[1],1.0)


class noflow6(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-6.0,-1.0)) and near(x[1],-1.0)


class noflow7(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-6.0,-1.0)) and near(x[1],1.0)


Noflow1 = noflow1()
Noflow2 = noflow2()
Noflow3 = noflow3()
Noflow4 = noflow4()
Noflow5 = noflow5()
Noflow6 = noflow6()
Noflow7 = noflow7()
Noflow8 = noflow8()


class inflow1(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (-1.0,1.0)) and near(x[1],-6.0)

class inflow2(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (-1.0,1.0)) and near(x[1],6.0)

class outflow1(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-1.0,1.0)) and near(x[0],6.0)

class outflow2(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-1.0,1.0)) and near(x[0],-6.0)

boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)

Noflow1.mark(boundaries, 1)
Noflow2.mark(boundaries, 1)
Noflow3.mark(boundaries, 1)
Noflow4.mark(boundaries, 1)
Noflow5.mark(boundaries, 1)
Noflow6.mark(boundaries, 1)
Noflow7.mark(boundaries, 1)
Noflow8.mark(boundaries, 1)

Inflow1 = inflow1()
Inflow2 = inflow2()

Outlow1 = outflow1()
Outlow2 = outflow2()

Inflow1.mark(boundaries, 2)
Inflow2.mark(boundaries, 2)

Outlow1.mark(boundaries, 3)
Outlow2.mark(boundaries, 3)

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# Set parameter values
dt = 0.01
T = 3
nu = 0.01

# Define time-dependent pressure boundary condition
p_in = Expression("-sin(3.0*t)", t=0.0)
p_in2 = Expression("sin(3.0*t)", t=0.0)

# Define boundary conditions
noslip  = DirichletBC(V, (0, 0),boundaries,1)
inflow  = DirichletBC(Q, p_in, boundaries,2)
outflow = DirichletBC(Q, p_in2, boundaries,3)
bcu = [noslip]
bcp = [inflow, outflow]

# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0))

# Tentative velocity step
F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
     nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(p), grad(q))*dx
L2 = -(1/k)*div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

# Time-stepping
t = dt
while t < T + DOLFIN_EPS:

    # Update pressure boundary condition
    p_in.t = t

    # Compute tentative velocity step
    begin("Computing tentative velocity")
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "gmres", "default")
    end()

    # Pressure correction
    begin("Computing pressure correction")
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    solve(A2, p1.vector(), b2, "gmres", prec)
    end()

    # Velocity correction
    begin("Computing velocity correction")
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "gmres", "default")
    end()

    # Plot solution
    plot(p1, title="Pressure", rescale=True)
    plot(u1, title="Velocity", scalarbar=True)

    # Save to file
    ufile << u1
    pfile << p1

    # Move to next time step
    u0.assign(u1)
    t += dt
    print "t =", t

# Hold plot
interactive()
