from dolfin import *

def check(nu, u_k, p_k, mesh, boundaries, domains):
    print "Boundary Modified Check: PCD"
    P = FunctionSpace(mesh, "CG", 1)
    q = TestFunction(P)
    p = TrialFunction(P)
    h = CellSize(Pressure.mesh())

    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    print "Assemble boundary modified Ap:"
    ApBM = assemble(assemble(nu*(inner(grad(q), grad(p))*dx(Pressure.mesh())) + inner(grad(p),N)*q*ds(2)))
    print "\nAssemble non-boundary modified Ap:"
    Ap = assemble(assemble(nu*(inner(grad(q), grad(p))*dx(Pressure.mesh()))))

    print "Boundary modified Mat-Vec"
    ApBM = assemble(assemble(nu*(inner(grad(p_k), grad(p))*dx(Pressure.mesh())) + inner(grad(q),N)*p*ds(2)))
