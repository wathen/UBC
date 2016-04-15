from dolfin import *
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import HiptmairSetup
import time
import MatrixOperations as MO
import NSprecondSetup


def FluidLinearSetup(Pressure,mu):
    MO.PrintStr("Preconditioning Fluid linear setup",3,"=","\n\n")
    parameters['linear_algebra_backend'] = 'uBLAS'
    p = TrialFunction(Pressure)
    q = TestFunction(Pressure)

    N = FacetNormal(Pressure.mesh())
    h = CellSize(Pressure.mesh())
    h_avg =avg(h)

    alpha = 10.0
    gamma =10.0
    tic()
    L = assemble(mu*(inner(grad(q), grad(p))*dx(Pressure.mesh()) \
                            - inner(avg(grad(q)), outer(p('+'),N('+'))+outer(p('-'),N('-')))*dS(Pressure.mesh()) \
                            - inner(outer(q('+'),N('+'))+outer(q('-'),N('-')), avg(grad(p)))*dS(Pressure.mesh()) \
                            + alpha/h_avg*inner(outer(q('+'),N('+'))+outer(q('-'),N('-')),outer(p('+'),N('+'))+outer(p('-'),N('-')))*dS(Pressure.mesh()) \
                            - inner(outer(q,N), grad(p))*ds(Pressure.mesh()) \
                            - inner(grad(q), outer(p,N))*ds(Pressure.mesh()) \
                            + gamma/h*inner(q,p)*ds(Pressure.mesh())))
    L = PETSc.Mat().createAIJ(size=L.sparray().shape,csr=(L.sparray().indptr, L.sparray().indices, L.sparray().data))
    print ("{:40}").format("DG scalar Laplacian assemble, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    tic()
    Q = assemble((1./mu)*inner(p,q)*dx)
    Q = PETSc.Mat().createAIJ(size=Q.sparray().shape,csr=(Q.sparray().indptr, Q.sparray().indices, Q.sparray().data))
    print ("{:40}").format("DG scalar mass matrix assemble, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    tic()
    kspA, kspQ = NSprecondSetup.PCDKSPlinear(Q, L)
    print ("{:40}").format("Linear fluid precond setup, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    return [kspA,kspQ], [L,Q]

def FluidNonLinearSetup(Pressure,mu, u_k):
    MO.PrintStr("Preconditioning Fluid linear setup",3,"=")
    parameters['linear_algebra_backend'] = 'uBLAS'
    p = TrialFunction(Pressure)
    q = TestFunction(Pressure)
    mesh = Pressure.mesh()
    N = FacetNormal(Pressure.mesh())
    h = CellSize(Pressure.mesh())
    h_avg =avg(h)

    alpha = 10.0
    gamma =10.0

    tic()
    Fp = assemble(mu*(inner(grad(q), grad(p))*dx(mesh) \
                            - inner(avg(grad(q)), outer(p('+'),N('+'))+outer(p('-'),N('-')))*dS (Pressure.mesh())\
                            - inner(outer(q('+'),N('+'))+outer(q('-'),N('-')), avg(grad(p)))*dS (Pressure.mesh())\
                            +
                            alpha/h_avg*inner(outer(q('+'),N('+'))+outer(q('-'),N('-')),outer(p('+'),N('+'))+outer(p('-'),N('-')))*dS(Pressure.mesh()) \
                            - inner(outer(q,N), grad(p))*ds(mesh) \
                            - inner(grad(q), outer(p,N))*ds(mesh) \
                            + gamma/h*inner(q,p)*ds(mesh)) \
                            + inner(inner(grad(p),u_k),q)*dx(mesh)- (1/2)*inner(u_k,N)*inner(q,p)*ds(mesh) \
                            -(1/2)*(inner(u_k('+'),N('+'))+inner(u_k('-'),N('-')))*avg(inner(q,p))*ds(mesh) \
                            -dot(avg(q),dot(outer(p('+'),N('+'))+outer(p('-'),N('-')),avg(u_k)))*dS(Pressure.mesh()))
    Fp = PETSc.Mat().createAIJ(size=Fp.sparray().shape,csr=(Fp.sparray().indptr, Fp.sparray().indices, Fp.sparray().data))
    print ("{:40}").format("DG convection-diffusion assemble, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    tic()
    kspFp= NSprecondSetup.PCDKSPnonlinear(Fp)
    print ("{:40}").format("Non-linear fluid precond, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    print "\n\n"
    return kspFp, Fp

def MagneticSetup(Magnetic, Lagrange, u0, p0, CGtol,params):
    MO.PrintStr("Preconditioning Magnetic setup",3,"=")

    parameters['linear_algebra_backend'] = 'uBLAS'
    C, P = HiptmairSetup.HiptmairMatrixSetupBoundary(Magnetic.mesh(), Magnetic.dim(), Lagrange.dim(),Magnetic.mesh().geometry().dim())
    G, P = HiptmairSetup.HiptmairBCsetupBoundary(C,P,Magnetic.mesh())


    u = TrialFunction(Magnetic)
    v = TestFunction(Magnetic)
    p = TrialFunction(Lagrange)
    q = TestFunction(Lagrange)


    def boundary(x, on_boundary):
        return on_boundary
    bcp = DirichletBC(Lagrange, p0, boundary)
    bcu = DirichletBC(Magnetic, u0, boundary)

    tic()
    ScalarLaplacian, b1 = assemble_system(inner(grad(p),grad(q))*dx,inner(p0,q)*dx,bcp)
    VectorLaplacian, b2 = assemble_system(inner(grad(p),grad(q))*dx+inner(p,q)*dx,inner(p0,q)*dx,bcp)
    del b1, b2
    print ("{:40}").format("Hiptmair Laplacians BC assembled, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    tic()
    VectorLaplacian = PETSc.Mat().createAIJ(size=VectorLaplacian.sparray().shape,csr=(VectorLaplacian.sparray().indptr, VectorLaplacian.sparray().indices, VectorLaplacian.sparray().data))
    ScalarLaplacian = PETSc.Mat().createAIJ(size=ScalarLaplacian.sparray().shape,csr=(ScalarLaplacian.sparray().indptr, ScalarLaplacian.sparray().indices, ScalarLaplacian.sparray().data))
    print ("{:40}").format("PETSc Laplacians assembled, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    tic()
    CurlCurlShift, b2 = assemble_system(params[0]*params[1]*inner(curl(u),curl(v))*dx+inner(u,v)*dx,inner(u0,v)*dx,bcu)
    CurlCurlShift = PETSc.Mat().createAIJ(size=CurlCurlShift.sparray().shape,csr=(CurlCurlShift.sparray().indptr, CurlCurlShift.sparray().indices, CurlCurlShift.sparray().data))
    print ("{:40}").format("Shifted Curl-Curl assembled, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    tic()
    kspVector, kspScalar, kspCGScalar, diag = HiptmairSetup.HiptmairKSPsetup(VectorLaplacian, ScalarLaplacian, CurlCurlShift, CGtol)
    del VectorLaplacian, ScalarLaplacian
    print ("{:40}").format("Hiptmair Setup time:"), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])


    return [G, P, kspVector, kspScalar, kspCGScalar, diag, CurlCurlShift]
