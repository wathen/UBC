from dolfin import *
import numpy as np
import scipy.sparse as sp

def PETSc2Scipy(A):
    row, col, value = A.getValuesCSR()
    return sp.csr_matrix((value, col, row), shape=A.size)

def check(nu, u_k, p_k, mesh, boundaries, domains):
    print "Boundary Modified Check: PCD"
    P = FunctionSpace(mesh, "CG", 1)
    p = TestFunction(P)
    q = TrialFunction(P)
    h = CellSize(mesh)
    N = FacetNormal(mesh)

    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    print "\nAssemble boundary modified Ap:"
    ApBM = assemble(nu*inner(grad(q), grad(p))*dx(0) - inner(grad(q),N)*p*ds(2))
    print "Assemble non-boundary modified Ap:"
    Ap = assemble(nu*inner(grad(q), grad(p))*dx(0))

    print "Boundary modified Mat-Vec"
    BM = assemble(nu*inner(grad(p_k), grad(p))*dx(0) - inner(grad(p_k),N)*p*ds(2))
    print "Non-boundary modified Mat-Vec"
    nBM = assemble(nu*inner(grad(p_k), grad(p))*dx(0))

    Ap = as_backend_type(Ap).mat()
    BM = as_backend_type(BM).vec()
    ApBM = as_backend_type(ApBM).mat()
    nBM = as_backend_type(nBM).vec()
    Ap = PETSc2Scipy(Ap)*p_k.vector().array()
    ApBM = PETSc2Scipy(ApBM)*p_k.vector().array()

    print "\n norm(Ap-(Ap)p):     ", np.linalg.norm(Ap-nBM)
    print "\n norm(Ap_B-(Ap)p):   ", np.linalg.norm(ApBM-nBM)
    print "\n norm(Ap-(Ap_B)p):   ", np.linalg.norm(Ap-BM)
    print "\n norm(Ap_B-(Ap_B)p): ", np.linalg.norm(ApBM-BM)


    print "\n\nAssemble boundary modified Fp:"
    FpBM = assemble(nu*inner(grad(q), grad(p))*dx(0) + inner(grad(p),u_k)*q*dx(0) + (1./2)*div(u_k)*inner(p,q)*dx(0) - (1./2)*inner(u_k,N)*inner(p,q)*ds(0) + (-mu*inner(grad(q),N)*p + inner(u_k, n)*q)*p*ds(2))
    print "Assemble non-boundary modified Fp:"
    Fp = assemble(nu*inner(grad(q), grad(p))*dx(0) + inner(grad(q),u_k)*p*dx(0) + (1./2)*div(u_k)*inner(p,q)*dx(0) - (1./2)*inner(u_k,N)*inner(p,q)*ds(0))

    print "Boundary modified Mat-Vec"
    BM = assemble(nu*inner(grad(p_k), grad(p))*dx(0) + inner(grad(p_k),u_k)*q*dx(0) + (1./2)*div(u_k)*inner(p,p_k)*dx(0) - (1./2)*inner(u_k,N)*inner(p,p_k)*ds(0) + (-mu*inner(grad(p_k),N)*p + inner(u_k, n)*p_k)*p*ds(2))
    print "Non-boundary modified Mat-Vec"
    nBM = assemble(nu*inner(grad(p_k), grad(p))*dx(0) + inner(grad(p_k),u_k)*p*dx(0) + (1./2)*div(u_k)*inner(p,p_k)*dx(0) - (1./2)*inner(u_k,N)*inner(p,p_k)*ds(0))
    Fp = as_backend_type(Fp).mat()
    BM = as_backend_type(BM).vec()
    FpBM = as_backend_type(FpBM).mat()
    nBM = as_backend_type(nBM).vec()
    Fp = PETSc2Scipy(Fp)*p_k.vector().array()
    FpBM = PETSc2Scipy(FpBM)*p_k.vector().array()

    print "\n norm(Fp-(Fp)p):     ", np.linalg.norm(Fp-nBM)
    print "\n norm(Fp_B-(Fp)p):   ", np.linalg.norm(FpBM-nBM)
    print "\n norm(Fp-(Fp_B)p):   ", np.linalg.norm(Fp-BM)
    print "\n norm(Fp_B-(Fp_B)p): ", np.linalg.norm(FpBM-BM)

