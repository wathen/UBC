from dolfin import *
import PETScIO as IO
import numpy as np


def DirectErrors(x,mesh,FSpaces,ExactSolution,k,dim):

    Vdim = dim[0]
    Pdim = dim[1]
    Mdim = dim[2]
    Rdim = dim[3]

    VelocityE = VectorFunctionSpace(mesh,"CG",k+2)
    u = interpolate(ExactSolution[0],VelocityE)

    PressureE = FunctionSpace(mesh,"CG",k+1)


    MagneticE = FunctionSpace(mesh,"N1curl",k+2)
    b = interpolate(ExactSolution[2],MagneticE)

    LagrangeE = FunctionSpace(mesh,"CG",k+2)
    r = interpolate(ExactSolution[3],LagrangeE)

    X = IO.vecToArray(x)
    xu = X[0:Vdim]
    ua = Function(FSpaces[0])
    ua.vector()[:] = xu

    pp = X[Vdim:Vdim+Pdim-1]
    # xp[-1] = 0
    # pa = Function(Pressure)
    # pa.vector()[:] = xp

    n = pp.shape
    pp = np.insert(pp,n,0)
    pa = Function(FSpaces[1])
    pa.vector()[:] = pp

    pend = assemble(pa*dx)

    ones = Function(FSpaces[1])
    ones.vector()[:]=(0*pp+1)
    pp = Function(FSpaces[1])
    pp.vector()[:] = pa.vector().array()- assemble(pa*dx)/assemble(ones*dx)

    pInterp = interpolate(ExactSolution[1],PressureE)
    pe = Function(PressureE)
    pe.vector()[:] = pInterp.vector().array()
    const = - assemble(pe*dx)/assemble(ones*dx)
    pe.vector()[:] = pe.vector()[:]+const


    xb = X[Vdim+Pdim-1:Vdim+Pdim+Mdim-1]
    ba = Function(FSpaces[2])
    ba.vector()[:] = xb

    xr = X[Vdim+Pdim+Mdim-1:]
    ra = Function(FSpaces[3])
    ra.vector()[:] = xr

    ErrorU = Function(FSpaces[0])
    ErrorP = Function(FSpaces[1])
    ErrorB = Function(FSpaces[2])
    ErrorR = Function(FSpaces[3])

    ErrorU = u-ua
    ErrorP = pe-pp
    ErrorB = b-ba
    ErrorR = r-ra


    errL2u= sqrt(abs(assemble(inner(ErrorU, ErrorU)*dx)))
    errH1u= sqrt(abs(assemble(inner(grad(ErrorU), grad(ErrorU))*dx)))
    errL2p= sqrt(abs(assemble(inner(ErrorP, ErrorP)*dx)))
    errL2b= sqrt(abs(assemble(inner(ErrorB, ErrorB)*dx)))
    errCurlb = sqrt(abs(assemble(inner(curl(ErrorB), curl(ErrorB))*dx)))
    errL2r= sqrt(abs(assemble(inner(ErrorR, ErrorR)*dx)))
    errH1r= sqrt(abs(assemble(inner(grad(ErrorR), grad(ErrorR))*dx)))

    return errL2u, errH1u, errL2p, errL2b, errCurlb, errL2r, errH1r






# def IterativeErrors():
