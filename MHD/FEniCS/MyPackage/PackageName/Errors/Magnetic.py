from dolfin import *
import numpy as np

def Magnetic(x,FS,ExactSolution):

    parameters["form_compiler"]["quadrature_degree"] = 6
    mesh = FS[0].mesh()
    Vdim = FS[0].dim()
    Pdim = FS[1].dim()


    MagneticE = FunctionSpace(mesh, FS[0].ufl_element().family(),FS[0].ufl_element().degree()+3)
    b = interpolate(ExactSolution[0],MagneticE)

    MultiplierE = FunctionSpace(mesh, FS[1].ufl_element().family(),FS[1].ufl_element().degree()+3)
    r = interpolate(ExactSolution[1],MultiplierE)

    X = x.array

    xb = X[:Vdim]
    ba = Function(FS[0])
    ba.vector()[:] = xb

    xr = X[Vdim:]
    ra = Function(FS[1])
    ra.vector()[:] = xr

    ErrorU = Function(MagneticE)
    ErrorP = Function(MultiplierE)

    ErrorB = b-ba
    ErrorR = r-ra

    errL2b = sqrt(abs(assemble(inner(ErrorB, ErrorB)*dx)))
    errCurlb = sqrt(abs(assemble(inner(curl(ErrorB), curl(ErrorB))*dx)))
    errL2r = sqrt(abs(assemble(inner(ErrorR, ErrorR)*dx)))
    errH1r = sqrt(abs(assemble(inner(grad(ErrorR), grad(ErrorR))*dx)))

    return errL2b, errCurlb, errL2r, errH1r


