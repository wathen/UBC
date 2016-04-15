# from MatrixOperations import *
from dolfin import *
import ipdb
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
# from MatrixOperations import *

# MO = MatrixOperations()

m =7
errL2 = np.zeros((m-1,1))
errDIV= np.zeros((m-1,1))
errH1 = np.zeros((m-1,1))
errDG = np.zeros((m-1,1))

NN = np.zeros((m-1,1))
DoF = np.zeros((m-1,1))
nn = 2


dim = 2
Solving = 'yes'
Saving = 'no'

if Saving == 'yes':
    parameters['linear_algebra_backend'] = 'Epetra'
else:
    parameters['linear_algebra_backend'] = 'PETSc'

for xx in xrange(1,m):
    print xx
    nn = 2**xx
    NN[xx-1] = nn
    # Create mesh and define function space
    nn = int(nn)
    if dim == 3:
        mesh = UnitCubeMesh(nn,nn,nn)
    else:
        mesh = UnitSquareMesh(nn,nn)

    V =FunctionSpace(mesh, "BDM", 1 )

    # creating trial and test function s
    v = TestFunction(V)
    u = TrialFunction(V)


    def boundary(x, on_boundary):
        return on_boundary

    # Creating expressions along the boundary
    if dim == 3:
        u0 = Expression(("0","0","0"))
    else:
        u0 = Expression(('(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])+x[0]*x[1]','(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])+x[0]*x[1]'))

    N = FacetNormal(mesh)
    # defining boundary conditions
    bcs = DirichletBC(V,u0, boundary)

    #  Creating RHS function
    if dim == 3:
        f = Expression(('- 2*(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])-2*(x[0]*x[0]-x[0])*(x[2]*x[2]-x[2])-2*(x[1]*x[1]-x[1])*(x[2]*x[2]-x[2])', \
        '- 2*(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])-2*(x[0]*x[0]-x[0])*(x[2]*x[2]-x[2])-2*(x[1]*x[1]-x[1])*(x[2]*x[2]-x[2])', \
        '- 2*(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])-2*(x[0]*x[0]-x[0])*(x[2]*x[2]-x[2])-2*(x[1]*x[1]-x[1])*(x[2]*x[2]-x[2])'))
    else:
        f = Expression(('- 2*(x[1]*x[1]-x[1])-2*(x[0]*x[0]-x[0])','-2*(x[0]*x[0]-x[0]) - 2*(x[1]*x[1]-x[1])'))
        # f = Expression(("0","0")

    # defining normal component

    h = CellSize(mesh)
    h_avg =avg(h)
    alpha = 10.0
    gamma =10.0
    t = as_vector((-N[0], N[1]))
    inside = avg(outer(N,grad(inner(v,t))))
     # tic()
    # a =  inner(grad(v), grad(u))*dx \
    # - inner(avg(outer(N,grad(inner(v,t)))), outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
    # - inner(outer(v('+'),N('+'))+outer(v('-'),N('-')), avg(outer(N,grad(inner(u,t)))))*dS \
    # + alpha/h_avg*inner(outer(v('+'),N('+'))+outer(v('-'),N('-')),outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
    # - inner(v, grad(inner(u,N)))*ds \
    # - inner(grad(v,in), u)*ds \
    # + gamma/h*inner(v,u)*ds

    a =  inner(grad(v), grad(u))*dx \
    - inner(avg(grad(v)), outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
    - inner(outer(v('+'),N('+'))+outer(v('-'),N('-')), avg(grad(u)))*dS \
    + alpha/h_avg*inner(outer(v('+'),N('+'))+outer(v('-'),N('-')),outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
    - inner(outer(v,N), grad(u))*ds \
    - inner(grad(v), outer(u,N))*ds \
    + gamma/h*inner(v,u)*ds

    # - inner(outer(v,N), grad(u))*ds \
    # - inner(grad(v), outer(u,N))*ds \


    L = inner(v,f)*dx+ gamma/h*inner(u0,v)*ds - inner(grad(v),outer(u0,N))*ds

        # - inner(grad(v), outer(u,N))*ds \

    # b =- dot(outer(v,n), grad(u))*ds \
    #     - dot(grad(v), outer(u,n))*ds
    # assemebling system
    AA,bb = assemble_system(a,L,bcs)

    DoF[xx-1] = bb.array().size


    u = Function(V)
    if Solving == 'yes':
        # tic()
        # set_log_level(PROGRESS)
        # solver = KrylovSolver("cg","amg")
        # solver.parameters["relative_tolerance"] = 1e-10
        # solver.parameters["absolute_tolerance"] = 1e-10
        # solver.solve(AA,u.vector(),bb)
        # set_log_level(PROGRESS)
        # print 'time to solve linear system', toc(),'\n\n'
        print 'DoF', DoF[xx-1]
        solve(a==L,u)

        if dim == 3:
            ue = Expression(('x[0]*x[1]*x[2]*(x[1]-1)*(x[2]-1)*(x[0]-1)','x[0]*x[1]*x[2]*(x[1]-1)*(x[2]-1)*(x[0]-1)','x[0]*x[1]*x[2]*(x[1]-1)*(x[2]-1)*(x[0]-1)'))
        else:
            #ue = Expression(('(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])','(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])'))
            ue = Expression(('(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])+x[0]*x[1]','(x[1]*x[1]-x[1])*(x[0]*x[0]-x[0])+x[0]*x[1]'))
            # ue = Expression(('x[0]*x[1]','x[0]*x[1]'))


        e = ue- Function(V,u)
        uu= Function(V,u)
        errL2[xx-1]=errornorm(ue,Function(V,u),norm_type="L2", degree_rise=4,mesh=mesh)
        errDIV[xx-1]=errornorm(ue,Function(V,u),norm_type="Hdiv", degree_rise=4,mesh=mesh)
        errH1[xx-1]=errornorm(ue,Function(V,u),norm_type="H1", degree_rise=4,mesh=mesh)
        errDG[xx-1] = errL2[xx-1] +errH1[xx-1]
        print errL2[xx-1],errDIV[xx-1],errH1[xx-1],errDG[xx-1]




####if Saving == 'yes':
    #MO.SaveEpertaMatrix(AA.down_cast().mat(),"A2d")

# plot(u)
# plot(interpolate(ue,V))
# interactive()
plt.loglog(NN,errL2)
plt.title('Error plot for BDM2 elements - L2 convergence = %f' % np.log2(np.average((errL2[0:m-2]/errL2[1:m-1]))))
plt.xlabel('N')
plt.ylabel('L2 error')


plt.figure()
plt.loglog(NN,errDIV)
plt.title('Error plot for BDM2 elements - Hdiv convergence = %f' % np.log2(np.average((errDIV[0:m-2]/errDIV[1:m-1]))))
plt.xlabel('N')
plt.ylabel('Hdiv error')

plt.figure()
plt.loglog(NN,errH1)
plt.title('Error plot for BDM2 elements - H1 convergence = %f' % np.log2(np.average((errH1[0:m-2]/errH1[1:m-1]))))
plt.xlabel('N')
plt.ylabel('H1 error')

plt.figure()
plt.loglog(NN,errDG)
plt.title('Error plot for BDM2 elements - DG convergence = %f' % np.log2(np.average((errDG[0:m-2]/errDG[1:m-1]))))
plt.xlabel('N')
plt.ylabel('H1 error')


plt.show()
