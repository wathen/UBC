
from dolfin import *

if not has_cgal():   # does not work without CGAL
    print "DOLFIN must be compiled with CGAL to run this demo."
    exit(0)

TOL = 1e-4           # error tolerance for stopping adaptive iterations
REFINE_RATIO = 0.50  # refine 50 % of the cells in each iteration
MAX_ITER = 20        # maximal number of iterations

# create initial mesh
msh = UnitSquareMesh(2,2)

# set rhs f, no expression for exact solution known
source_str = "exp(-100.0*(pow(x[0], 2) + pow(x[1], 2)))"

ufile = File("out/dpgsol.pvd")

# adaptive algorithm
for level in xrange(MAX_ITER):

    # define the lowest order DPG method
    ER  = FunctionSpace(msh,"DG",2)   # DPG error estimator
    CG  = FunctionSpace(msh,"CG",1)   # linear solution u
    Q   = FunctionSpace(msh,"RT",1,restriction="facet")
                                      # constant fluxes q
    X = MixedFunctionSpace( [ER,CG,Q] )
    (e,u,q) = TrialFunctions(X)
    (y,z,r) = TestFunctions(X)
    n = FacetNormal(msh)

    # the Y-inner product
    yip = dot(grad(e),grad(y))*dx + e*y*dx

    # b( (u,q), y ) = (grad u, grad y) - <q.n, y>
    b1  = dot( grad(u),grad(y) ) * dx               \
        - dot(q('+'),n('+')) * (y('+')-y('-')) * dS \
        - dot(q,n)*y*ds

    # b( (z,r), e)
    b2 =  dot( grad(e),grad(z) ) * dx               \
        - (e('+')-e('-')) * dot(r('+'),n('+')) * dS \
        - e*dot(r,n)*ds

    a = yip + b1 + b2
    f = Expression(source_str)
    b = f*y*dx

    # solve
    x = Function(X)
    bc = DirichletBC(X.sub(1), Constant(0.0), DomainBoundary())
    solve( a==b, x, bcs=bc)
    e,u,q = x.split()

    # compute error indicators from the DPG mixed variable e
    PC = FunctionSpace(msh,"DG", 0)
    c  = TestFunction(PC)             # p.w constant fn
    g  = assemble((dot(grad(e),grad(e))+e*e)*c*dx)
    g  = g.array()                    # element-wise norms of e
    E  = sqrt( assemble((dot(grad(e),grad(e))+e*e)*dx))
    print "Level %d: Estimated error E = %g (TOL = %g)"%\
        (level, E, TOL)

    # Check convergence
    if E < TOL:
        print "Success, solution converged after %d iterations"% level
        break

    # Mark cells for refinement
    cell_markers = MeshFunction("bool", msh, msh.topology().dim())
    g0 = sorted(g, reverse=True)[int(len(g)*REFINE_RATIO)]
    for c in cells(msh):
        cell_markers[c] = g[c.index()] > g0

    # Refine mesh
    msh = refine(msh, cell_markers)

    # Plot mesh
    plot(u,wireframe=True,scalarbar=False)
    ufile << u

# Hold plot
interactive()