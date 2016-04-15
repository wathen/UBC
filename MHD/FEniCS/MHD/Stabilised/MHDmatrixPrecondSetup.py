from dolfin import *
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import HiptmairSetup
import time


def FluidSetup(Velocity, Pressure):
    print "Fluid setup...."





def MagneticSetup(Magnetic, Lagrange, u0, p0, CGtol):
    print "Magnetic setup...."
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
    CurlCurlShift, b2 = assemble_system(inner(curl(u),curl(v))*dx+inner(u,v)*dx,inner(u0,v)*dx,bcu)
    CurlCurlShift = PETSc.Mat().createAIJ(size=CurlCurlShift.sparray().shape,csr=(CurlCurlShift.sparray().indptr, CurlCurlShift.sparray().indices, CurlCurlShift.sparray().data))
    print ("{:40}").format("Shifted Curl-Curl assembled, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    tic()
    kspVector, kspScalar, kspCGScalar, diag = HiptmairSetup.HiptmairKSPsetup(VectorLaplacian, ScalarLaplacian, CurlCurlShift, CGtol)
    del VectorLaplacian, ScalarLaplacian
    print ("{:40}").format("Hiptmair Setup time:"), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])


    return [G, P, kspVector, kspScalar, kspCGScalar, diag, CurlCurlShift]
