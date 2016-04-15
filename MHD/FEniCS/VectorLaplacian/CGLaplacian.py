# from MatrixOperations import *
from dolfin import *
#import ipdb
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
# from MatrixOperations import *

# MO = MatrixOperations()

m = 9
err = np.zeros((m-1,1))
N = np.zeros((m-1,1))
DoF = np.zeros((m-1,1))
nn = 2


dim = 2
Solving = 'yes'
Saving = 'no'

# if Saving == 'yes':
#     parameters['linear_algebra_backend'] = 'Epetra'
# else:
#     parameters['linear_algebra_backend'] = 'Epetra'

for xx in xrange(1,m):
    print xx
    nn = 2**xx
    N[xx-1] = nn
    # Create mesh and define function space
    nn = int(nn)
    if dim == 3:
        mesh = UnitCubeMesh(nn,nn,nn)
    else:
        mesh = UnitSquareMesh(nn,nn)
    # tic()
    parameters['reorder_dofs_serial'] = False
    V =VectorFunctionSpace(mesh, "CG", 2 )

    v = TestFunction(V)
    u = TrialFunction(V)

    def boundary(x, on_boundary):
        return on_boundary

    if dim == 3:
        u0 = Expression(('0','0','0'))
    else:
        u0 = Expression(('x[0]*x[1]','x[0]*x[1]'))


    # # p0 = ('0')

    bcs = DirichletBC(V,u0, boundary)

    if dim == 3:
        f = Expression(('- 2*(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])-2*(x[0]*x[0]-x[0])*(x[2]*x[2]-x[2])-2*(x[1]*x[1]-x[1])*(x[2]*x[2]-x[2])', \
        '- 2*(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])-2*(x[0]*x[0]-x[0])*(x[2]*x[2]-x[2])-2*(x[1]*x[1]-x[1])*(x[2]*x[2]-x[2])', \
        '- 2*(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])-2*(x[0]*x[0]-x[0])*(x[2]*x[2]-x[2])-2*(x[1]*x[1]-x[1])*(x[2]*x[2]-x[2])'))
    else:
        f = Expression(('- 2*(x[1]*x[1]-x[1])-2*(x[0]*x[0]-x[0])','-2*(x[0]*x[0]-x[0]) - 2*(x[1]*x[1]-x[1])'))



    # tic()
    a = inner(grad(v), grad(u))*dx
    L = inner(v,f)*dx
    AA,bb = assemble_system(a,L,bcs)
    # print 'time to creat linear system',toc(),'\n\n'
    DoF[xx-1] = bb.array().size
    # Compute solution
    u = Function(V)
    if Solving == 'yes':
        tic()
        set_log_level(PROGRESS)
        solver = KrylovSolver("cg","amg")
        solver.parameters["relative_tolerance"] = 1e-8
        solver.parameters["absolute_tolerance"] = 1e-8
        solver.solve(AA,u.vector(),bb)
        set_log_level(PROGRESS)
        print 'time to solve linear system', toc(),'\n\n'
        # solve(a==L,u,bcs)

        if dim == 3:
            ue = Expression(('x[0]*x[1]*x[2]*(x[1]-1)*(x[2]-1)*(x[0]-1)','x[0]*x[1]*x[2]*(x[1]-1)*(x[2]-1)*(x[0]-1)','x[0]*x[1]*x[2]*(x[1]-1)*(x[2]-1)*(x[0]-1)'))
        else:
            ue = Expression(('(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])+x[0]*x[1]','(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])+x[0]*x[1]'))


        e = ue- Function(V,u)
        uu= Function(V,u)
        err[xx-1]=errornorm(ue,Function(V,u),norm_type="L2", degree_rise=3,mesh=mesh)
        print err[xx-1]
    # uE = interpolate(ue,V)
    # ue = uE.vector().array()
    # u = u.vector().array()

    # print scipy.linalg.norm(u-ue)
# # Plot solution
# plot(u)
# plot(interpolate(ue,V))

# interactive()
# print N,err


# print '\n\n'
# print (err[0:m-2]/err[1:m-1])
# print '\n\n'
if Saving == 'yes':
    MO.SaveEpertaMatrix(AA.down_cast().mat(),"A2d")
else:
    plt.loglog(N,err)
    plt.title('Error plot for P2 elements - L2 convergence = %f' % np.log2(np.average((err[0:m-2]/err[1:m-1]))))
    plt.xlabel('N')
    plt.ylabel('L2 error')
    plt.show()