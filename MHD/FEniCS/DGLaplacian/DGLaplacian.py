from dolfin import *
import ipdb
import numpy as np
import matplotlib.pylab as plt

m =9
err = np.zeros((m-1,1))
N = np.zeros((m-1,1))
errh1 = np.zeros((m-1,1))
nn = 2

for xx in xrange(1,m):
    # Create mesh and define function space

    n = 2**xx
    N[xx-1] = n
    mesh = UnitSquareMesh(n,n)
    tic()
    V = FunctionSpace(mesh, "DG",2 )
    print 'time to create function spaces',toc(),'\n\n'
    # Define test and trial functions
    v = TestFunction(V)
    u = TrialFunction(V)

    def boundary(x, on_boundary):
        return on_boundary


    u0 = Expression('x[0]*x[1]')
    # # p0 = ('0')

    # bcs = DirichletBC(V,u0, boundary)

    # Define normal component, mesh size and right-hand side
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg = (h('+') + h('-'))/2
    f = Expression('-2*(x[0]*x[0]-x[0]) - 2*(x[1]*x[1]-x[1])')
    # Define parameters
    alpha = 10.0
    gamma = 10.0

    # Define variational problem
    tic()
    a = dot(grad(v), grad(u))*dx \
       - dot(avg(grad(v)), jump(u, n))*dS \
       - dot(jump(v, n), avg(grad(u)))*dS \
       + alpha/h_avg*dot(jump(v, n), jump(u, n))*dS \
       - dot(v*n, grad(u))*ds \
        - dot(grad(v), u*n)*ds \
       + gamma/h*v*u*ds
    L = v*f*dx + gamma/h*u0*v*ds - inner(grad(v),n)*u0*ds
    AA,bb = assemble_system(a,L)
    print 'time to creat linear system',toc(),'\n\n'

    # Compute solution
    u = Function(V)
    tic()
    set_log_level(PROGRESS)
    solver = KrylovSolver("cg","hypre_amg")
    solver.parameters["relative_tolerance"] = 1e-6
    solver.parameters["absolute_tolerance"] = 1e-6
    solver.solve(AA,u.vector(),bb)
    set_log_level(PROGRESS)
    print 'time to solve linear system', toc(),'\n\n'

    # solve(a == L,u,bcs)

    ue = Expression('x[0]*x[1]*(x[1]-1)*(x[0]-1) + x[0]*x[1]')
    # ue = Expression('x[0]*x[1]*x[2]*(x[1]-1)*(x[2]-1)*(x[0]-1)')
    erru = ue- Function(V,u)
    err[xx-1]=errornorm(ue,Function(V,u),norm_type="L2", degree_rise=3,mesh=mesh)
    errh1[xx-1]=errornorm(ue,Function(V,u),norm_type="H1", degree_rise=3,mesh=mesh)
    print 'L2',err[xx-1]
    print 'H1',errh1[xx-1]


    # print sqrt(assemble(dolfin.dot(grad(erru),grad(erru))*dx))

# Plot solution
# plot(u, interactive=True)

plt.loglog(N,err)
plt.title('Error plot for DG1 elements - L2 convergence = %f' % np.log2(np.average((err[0:m-2]/err[1:m-1]))))
plt.xlabel('N')
plt.ylabel('L2 error')

plt.figure()

plt.loglog(N,err)
plt.title('Error plot for DG1 elements - H1 convergence = %f' % np.log2(np.average((errh1[0:m-2]/errh1[1:m-1]))))
plt.xlabel('N')
plt.ylabel('H1 error')

plt.show()