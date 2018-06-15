from dolfin import *
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import HiptmairSetup
import time
import MatrixOperations as MO
import NSprecondSetup
import CheckPetsc4py as CP

parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

def FluidLinearSetup(Pressure,mu,mesh):
    MO.PrintStr("Preconditioning Fluid linear setup",3,"=","\n\n")
    # parameters['linear_algebra_backend'] = 'uBLAS'
    q = TrialFunction(Pressure)
    p = TestFunction(Pressure)

    tic()
    L = assemble(mu*inner(grad(q), grad(p))*dx, form_compiler_parameters=ffc_options)
    L = CP.Assemble(L)
    print ("{:40}").format("CG scalar Laplacian assemble, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    tic()
    Q = assemble((1./mu)*inner(p,q)*dx, form_compiler_parameters=ffc_options)
    Q = CP.Assemble(Q)
    print ("{:40}").format("DG scalar mass matrix assemble, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    tic()
    kspL, kspQ = NSprecondSetup.PCDKSPlinear(L, Q)
    print ("{:40}").format("Linear fluid precond setup, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    return [kspL, kspQ], [L,Q]

def FluidNonLinearSetup(Pressure,mu, u_k, mesh):
    MO.PrintStr("Preconditioning Fluid linear setup",3,"=")
    # parameters['linear_algebra_backend'] = 'uBLAS'
    p = TrialFunction(Pressure)
    q = TestFunction(Pressure)
    mesh = Pressure.mesh()
    N = FacetNormal(Pressure.mesh())
    h = CellSize(Pressure.mesh())
    h_avg =avg(h)

    alpha = 10.0
    gamma =10.0
    tic()
    if Pressure.__str__().find("CG") == -1:
        Fp = assemble(mu*(jump(q)*jump(p)*dx(mesh)) \
                                + inner(inner(grad(p),u_k),q)*dx(mesh)- (1./2)*inner(u_k,N)*inner(q,p)*ds(mesh) \
                                -(1./2)*(inner(u_k('+'),N('+'))+inner(u_k('-'),N('-')))*avg(inner(q,p))*ds(mesh) \
                                -dot(avg(q),dot(outer(p('+'),N('+'))+outer(p('-'),N('-')),avg(u_k)))*dS(Pressure.mesh()), form_compiler_parameters=ffc_options)
    else:
        if mesh.topology().dim() == 2:
            Fp = assemble(mu*inner(grad(q), grad(p))*dx(mesh)+inner((u_k[0]*grad(p)[0]+u_k[1]*grad(p)[1]),q)*dx(mesh) + (1./2)*div(u_k)*inner(p,q)*dx(mesh) + (1./2)*(u_k[0]*N[0]+u_k[1]*N[1])*inner(p,q)*ds(mesh), form_compiler_parameters=ffc_options)
        else:
            Fp = assemble(mu*inner(grad(q), grad(p))*dx(mesh)+inner((u_k[0]*grad(p)[0]+u_k[1]*grad(p)[1]+u_k[2]*grad(p)[2]),q)*dx(mesh) + (1./2)*div(u_k)*inner(p,q)*dx(mesh) - (1./2)*(u_k[0]*N[0]+u_k[1]*N[1]+u_k[2]*N[2])*inner(p,q)*ds(mesh), form_compiler_parameters=ffc_options)# + (-mu*inner(grad(q),N)*p + inner(u_k, N)*q*p)*ds(2))

    Fp = CP.Assemble(Fp)
    print ("{:40}").format("DG convection-diffusion assemble, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    tic()
    kspFp= NSprecondSetup.PCDKSPnonlinear(Fp)
    print ("{:40}").format("Non-linear fluid precond, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    print "\n\n"
    return kspFp, Fp

def MagneticSetup(mesh, Magnetic, Lagrange, u0, p0, CGtol,params):
    MO.PrintStr("Preconditioning Magnetic setup",3,"=")
    Magnetic = FunctionSpace(mesh, Magnetic)
    Lagrange = FunctionSpace(mesh, Lagrange)

    # parameters['linear_algebra_backend'] = 'uBLAS'
    if Magnetic.__str__().find("N1curl2") == -1:
        G, P = HiptmairSetup.HiptmairMatrixSetupBoundary(Magnetic.mesh(), Magnetic.dim(), Lagrange.dim(),Magnetic.mesh().geometry().dim())
        G, P = HiptmairSetup.HiptmairBCsetupBoundary(G, P, Magnetic.mesh())
    else:
        G = None
        P = None

    u = TrialFunction(Magnetic)
    v = TestFunction(Magnetic)
    p = TrialFunction(Lagrange)
    q = TestFunction(Lagrange)


    def boundary(x, on_boundary):
        return on_boundary
    bcp = DirichletBC(Lagrange, p0, boundary)
    bcu = DirichletBC(Magnetic, u0, boundary)

    tic()
    ScalarLaplacian, b1 = assemble_system(inner(grad(p),grad(q))*dx,inner(p0,q)*dx,bcp, form_compiler_parameters=ffc_options)
    VectorLaplacian, b2 = assemble_system((params[0]*params[1]*inner(grad(p),grad(q))*dx+params[0]*params[1]*inner(p,q)*dx),inner(p0,q)*dx,bcp, form_compiler_parameters=ffc_options)
    del b1, b2
    print ("{:40}").format("Hiptmair Laplacians BC assembled, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    tic()
    VectorLaplacian = CP.Assemble(VectorLaplacian)
    ScalarLaplacian = CP.Assemble(ScalarLaplacian)
    print ("{:40}").format("PETSc Laplacians assembled, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    tic()
    if params[0] == 0:
        CurlCurlShift, b2 = assemble_system(params[1]*inner(curl(u),curl(v))*dx+inner(u,v)*dx,inner(u0,v)*dx,bcu, form_compiler_parameters=ffc_options)
    else:
        CurlCurlShift, b2 = assemble_system(params[0]*params[1]*inner(curl(u),curl(v))*dx+inner(u,v)*dx,inner(u0,v)*dx,bcu, form_compiler_parameters=ffc_options)
    CurlCurlShift = CP.Assemble(CurlCurlShift)
    print ("{:40}").format("Shifted Curl-Curl assembled, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    tic()
    kspVector, kspScalar, kspCGScalar, diag = HiptmairSetup.HiptmairKSPsetup(VectorLaplacian, ScalarLaplacian, CurlCurlShift, CGtol, G, mesh)
    del VectorLaplacian, ScalarLaplacian
    print ("{:40}").format("Hiptmair Setup time:"), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])


    return [G, P, kspVector, kspScalar, kspCGScalar, diag, CurlCurlShift]
