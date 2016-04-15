from dolfin import *
import numpy as np

def Fluid(x,FS,ExactSolution):


    mesh = FS[0].mesh()
    Vdim = FS[0].dim()
    Pdim = FS[1].dim()

    if FS[0].ufl_element().family() == 'EnrichedElement':
        VelocityE = VectorFunctionSpace(mesh, 'CG',3) + VectorFunctionSpace(mesh, 'B', 5)
    else:
        VelocityE = VectorFunctionSpace(mesh, FS[0].ufl_element().family(),FS[0].ufl_element().degree()+2)
    u = interpolate(ExactSolution[0],VelocityE)

    PressureE = FunctionSpace(mesh, FS[1].ufl_element().family(),FS[1].ufl_element().degree()+2)
    pInterp = interpolate(ExactSolution[1],PressureE)
    X = x.array
    xu = X[:Vdim]
    ua = Function(FS[0])
    ua.vector()[:] = xu

    pp = X[Vdim:]


    pa = Function(FS[1])
    pa.vector()[:] = pp

    ones = Function(FS[1])
    ones.vector()[:]=(0*pp+1)
    pp = Function(FS[1])
    pp.vector()[:] = pa.vector().array() - assemble(pa*dx)/assemble(ones*dx)


    pe = Function(PressureE)
    pe.vector()[:] = pInterp.vector().array()
    const = - assemble(pe*dx)/assemble(ones*dx)
    pe.vector()[:] = pe.vector()[:]+const

    ErrorU = Function(FS[0])
    ErrorP = Function(FS[1])

    ErrorU = u-ua
    ErrorP = pe-pp

    errL2u = sqrt(abs(assemble(inner(ErrorU, ErrorU)*dx)))
    errH1u = sqrt(abs(assemble(inner(grad(ErrorU), grad(ErrorU))*dx)))
    errL2p = sqrt(abs(assemble(inner(ErrorP, ErrorP)*dx)))

    return errL2u, errH1u, errL2p




