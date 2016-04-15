from dolfin import Function, assemble, dx
from numpy.linalg import norm
from PackageName.GeneralFunc import common
from PackageName.GeneralFunc import PrintFuncs

def NLtol(x, u, FS, Type = None):
    IS = common.IndexSet(FS)

    if Type == 'Update':
        v = x.getSubVector(IS['Velocity']).array
        p = x.getSubVector(IS['Pressure']).array

        pa = Function(FS['Pressure'])
        pa.vector()[:] = p

        ones = Function(FS['Pressure'])
        ones.vector()[:]=(0*p+1)
        pp = Function(FS['Pressure'])
        pp.vector()[:] = pa.vector().array() - assemble(pa*dx)/assemble(ones*dx)

        vnorm = norm(v)
        pnorm = norm(pp.vector().array())

        V = [vnorm, pnorm]
        eps = PrintFuncs.NormPrint(V, Type)

        x.axpy(1.0,u)
        return x, eps
    else:
        vcurrent = x.getSubVector(IS['Velocity']).array
        pcurrent = x.getSubVector(IS['Pressure']).array
        vprev = u.getSubVector(IS['Velocity']).array
        pprev = u.getSubVector(IS['Pressure']).array

        pa = Function(FS['Pressure'])
        pa.vector()[:] = pcurrent

        ones = Function(FS['Pressure'])
        ones.vector()[:]=(0*pcurrent+1)
        pp = Function(FS['Pressure'])
        pp.vector()[:] = pa.vector().array() - assemble(pa*dx)/assemble(ones*dx)

        vnorm = norm(vcurrent - vprev)
        pnorm = norm(pp.vector().array() - pprev)

        V = [vnorm, pnorm]
        eps = PrintFuncs.NormPrint(V, Type)

        return x, eps


