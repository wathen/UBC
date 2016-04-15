from dolfin import *
import numpy as np
import time

m =6
errL2u = np.zeros((m-1,1))
errL2p = np.zeros((m-1,1))
NN = np.zeros((m-1,1))
DoF = np.zeros((m-1,1))
Vdim = np.zeros((m-1,1))
Qdim = np.zeros((m-1,1))
Wdim = np.zeros((m-1,1))
l2uorder = np.zeros((m-1,1))
l2porder = np.zeros((m-1,1))
nonlinear = np.zeros((m-1,1))
nn = 2

dim = 2
Solving = 'No'
Saving = 'no'



for xx in xrange(1,m):
    print xx
    nn = 2**(xx)
    # Create mesh and define function space
    nn = int(nn)
    NN[xx-1] = nn

    mesh = RectangleMesh(-1, -1, 1, 1, nn, nn,'right')
    # tic()

    parameters['reorder_dofs_serial'] = False
    V = FunctionSpace(mesh, "BDM", 2)
    Q = FunctionSpace(mesh, "DG", 1)
    parameters['reorder_dofs_serial'] = False
    # print 'time to create function spaces', toc(),'\n\n'
    W = V*Q
    Vdim[xx-1] = V.dim()
    Qdim[xx-1] = Q.dim()
    Wdim[xx-1] = W.dim()
    print "\n\nV:  ",Vdim[xx-1],"Q:  ",Qdim[xx-1],"W:  ",Wdim[xx-1],"\n\n"

    def boundary(x, on_boundary):
        return on_boundary


    u0 = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
    p0 = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)")


    bc = DirichletBC(W.sub(0),u0, boundary)
    bcs = [bc]

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)


    f = Expression(("120*x[0]*x[1]*(1-mu)+ 400*x[0]*pow(x[1],6)+(5*pow(x[0],4)-5*pow(x[1],4))*60*x[0]*x[1]*x[1]","60*(pow(x[0],2)-pow(x[1],2))*(1-mu)+400*pow(x[0],4)*pow(x[1],3)-(5*pCostumes are mandatory or a donation for the local food bank or donations of booze to the house or pizza or falafel!
ow(x[0],4)-5*pow(x[1],4))*20*x[1]*x[1]*x[1]"), mu = 1e0)



    u_k = Function(V)
    mu = Constant(1e0)
    u_k.vector()[:] = u_k.vector()[:]*0
    N = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)
    alpha = 10.0
    gamma =10.0
    n = FacetNormal(mesh)
    h = CellSize(mesh)
    h_avg =avg(h)

    a11 = inner(grad(v), grad(u))*dx \
        - inner(avg(grad(v)), outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
        - inner(outer(v('+'),N('+'))+outer(v('-'),N('-')), avg(grad(u)))*dS \
        + alpha/h_avg*inner(outer(v('+'),N('+'))+outer(v('-'),N('-')),outer(u('+'),N('+'))+outer(u('-'),N('-')))*dS \
        - inner(outer(v,N), grad(u))*ds \
        - inner(grad(v), outer(u,N))*ds \
        + gamma/h*inner(v,u)*ds

    O = inner((grad(u)*u_k),v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds \
     -(1/2)*(inner(u_k('+'),N('+'))+inner(u_k('-'),N('-')))*avg(inner(u,v))*ds \
    -dot(avg(v),dot(outer(u('+'),N('+'))+outer(u('-'),N('-')),avg(u_k)))*dS

    a12 = div(v)*p*dx
    a21 = div(u)*q*dx
    L1  = inner(v, f)*dx + gamma/h*inner(u0,v)*ds - inner(grad(v),outer(u0,N))*ds
    a = a11+O-a12-a21

    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-4       # tolerance
    iter = 0            # iteration counter
    maxiter = 100        # max no of iterations allowed

    while eps > tol and iter < maxiter:
            iter += 1
            x = Function(W)

            uu = Function(W)
            solve(a == L1,uu,bcs)
            uu1=uu.vector().array()[:V.dim()]
            u1 = Function(V)
            u1.vector()[:] = u1.vector()[:] + uu1
            diff = u1.vector().array() - u_k.vector().array()
            eps = np.linalg.norm(diff, ord=np.Inf)

            print '\n\n\niter=%d: norm=%g' % (iter, eps)
            u_k.assign(u1)

#

    ue = Expression(("20*x[0]*pow(x[1],3)","5*pow(x[0],4)-5*pow(x[1],4)"))
    pe = Expression("60*pow(x[0],2)*x[1]-20*pow(x[1],3)+5")




    ua = Function(V)
    ua.vector()[:] = u_k.vector().array()
    nonlinear[xx-1] = assemble(inner((grad(ua)*ua),ua)*dx- (1/2)*inner(ua,n)*inner(ua,ua)*ds \
     -(1/2)*(inner(ua('+'),N('+'))+inner(ua('-'),N('-')))*avg(inner(ua,ua))*ds \
    -dot(avg(ua),dot(outer(ua('+'),N('+'))+outer(ua('-'),N('-')),avg(ua)))*dS)

    u = interpolate(ue,V)
    p = interpolate(pe,Q)

    uu1=uu.vector().array()[:V.dim()]
    ua = Function(V)
    ua.vector()[:] = ua.vector()[:] + uu1



    pp1=uu.vector().array()[V.dim():W.dim()]

    ones = Function(Q)
    ones.vector()[:]=(0*pp1+1)
    pp = Function(Q)
    pa = Function(Q)

    pa.vector()[:] =  pp1

    pp.vector()[:] = pa.vector().array() - assemble(pa*dx)/assemble(ones*dx)

    pInterp = interpolate(pe,Q)
    pe = Function(Q)
    pe.vector()[:] = pInterp.vector().array()
    const = - assemble(pe*dx)/assemble(ones*dx)
    pe.vector()[:] = pe.vector()[:]+const

    errL2u[xx-1] = errornorm(ua,u,norm_type="L2", degree_rise=4,mesh=mesh)
    errL2p[xx-1] = errornorm(pp,pe,norm_type="L2", degree_rise=4,mesh=mesh)
    if xx == 1:
        l2uorder[xx-1] = 0
        l2porder[xx-1] = 0
    else:
        l2uorder[xx-1] =  np.abs(np.log2(errL2u[xx-2]/errL2u[xx-1]))
        l2porder[xx-1] =  np.abs(np.log2(errL2p[xx-2]/errL2p[xx-1]))

    print errL2u[xx-1]
    print errL2p[xx-1]




print "Velocity Elements rate of convergence ", np.log2(np.average((errL2u[0:m-2]/errL2u[1:m-1])))
print "Pressure Elements rate of convergence ", np.log2(np.average((errL2p[0:m-2]/errL2p[1:m-1])))


import pandas as pd
tableTitles = ["Total DoF","V DoF","Q DoF","V-L2","V-order","P-L2","P-order"]
tableValues = np.concatenate((Wdim,Vdim,Qdim,errL2u,l2uorder,errL2p,l2porder),axis=1)
df = pd.DataFrame(tableValues, columns = tableTitles)
pd.set_option('precision',3)
print df


plot(pp)
plot(interpolate(pe,Q))

interactive()

