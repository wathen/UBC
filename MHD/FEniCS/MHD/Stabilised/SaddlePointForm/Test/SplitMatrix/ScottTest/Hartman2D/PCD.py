from dolfin import *
import numpy as np

def check(nu, u_k, p_k, mesh, boundaries, domains):
    print "Boundary Modified Check: PCD"
    P = FunctionSpace(mesh, "CG", 1)
    q = TestFunction(P)
    p = TrialFunction(P)
    h = CellSize(mesh)

    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    print "\nAssemble boundary modified Ap:"
    ApBM = assemble(assemble(nu*inner(grad(q), grad(p))*dx(0) - inner(grad(p),N)*q*ds(2)))
    print "Assemble non-boundary modified Ap:"
    Ap = assemble(assemble(nu*inner(grad(q), grad(p))*dx(0)))

    print "Boundary modified Mat-Vec"
    BM = assemble(assemble(nu*inner(grad(p_k), grad(p))*dx(0) - inner(grad(p_k),N)*p*ds(2)))
    print "Non-boundary modified Mat-Vec"
    nBM = assemble(assemble(nu*inner(grad(p_k), grad(p))*dx(0)))

    print "\n norm(Ap-(Ap)p):     ", np.linalg.norm(Ap-nBM)
    print "\n norm(Ap_B-(Ap)p):   ", np.linalg.norm(ApBM-nBM)
    print "\n norm(Ap-(Ap_B)p):   ", np.linalg.norm(Ap-BM)
    print "\n norm(Ap_B-(Ap_B)p): ", np.linalg.norm(ApBM-BM)