from dolfin import *
import PETScIO as IO
import numpy as np
import scipy.linalg as splin
import scipy
import petsc4py
import sys
import time
petsc4py.init(sys.argv)
# import matplotlib.pylab as plt
from petsc4py import PETSc
import MatrixOperations as MO

def StoreMatrix(A,name):
      test ="".join([name,".mat"])
      scipy.io.savemat( test, {name: A},oned_as='row')



def Errors(X,mesh,FSpaces,ExactSolution,k,dim, FS = "CG"):

    Vdim = dim[0]
    Pdim = dim[1]
    Mdim = dim[2]
    Rdim = dim[3]
    # k +=2
    VelocityE = VectorFunctionSpace(mesh,"CG",3)
    u = interpolate(ExactSolution[0],VelocityE)

    PressureE = FunctionSpace(mesh,FS,2)
    # parameters["form_compiler"]["quadrature_degree"] = 8
    # X = x.array()
    xu = X[0:Vdim]
    ua = Function(FSpaces[0])
    ua.vector()[:] = xu

    pp = X[Vdim:Vdim+Pdim]


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

    ErrorU = Function(FSpaces[0])
    ErrorP = Function(FSpaces[1])

    ErrorU = u-ua
    ErrorP = pe-pp
    tic()
    errL2u= sqrt(abs(assemble(inner(ErrorU, ErrorU)*dx)))
    MO.StrTimePrint("Velocity L2 error, time: ", toc())
    tic()
    errH1u= sqrt(abs(assemble(inner(grad(ErrorU), grad(ErrorU))*dx)))
    MO.StrTimePrint("Velocity H1 error, time: ", toc())
    tic()
    errL2p= sqrt(abs(assemble(inner(ErrorP, ErrorP)*dx)))
    MO.StrTimePrint("Pressure L2 error, time: ", toc())

    # parameters["form_compiler"]["quadrature_degree"] = 5

    MagneticE = FunctionSpace(mesh,"N1curl",2)
    LagrangeE = FunctionSpace(mesh,"CG",2)
    b = interpolate(ExactSolution[2],MagneticE)
    r = interpolate(ExactSolution[3],LagrangeE)


    xb = X[Vdim+Pdim:Vdim+Pdim+Mdim]
    ba = Function(FSpaces[2])
    ba.vector()[:] = xb

    xr = X[Vdim+Pdim+Mdim:]
    ra = Function(FSpaces[3])
    ra.vector()[:] = xr


    ErrorB = Function(FSpaces[2])
    ErrorR = Function(FSpaces[3])


    # plot(ua)
    # plot(pp)
    # plot(ba)
    # plot(ra)

    # plot(u)
    # plot(pe)
    # plot(b)
    # plot(r)

    # print curl(b).vector().array()
    # ssss
    # print b.vector().array()-ba.vector().array()
    # ssss
    ErrorB = b-ba
    ErrorR = r-ra
    # print '               Exact solution curl   ', assemble(curl(b)*dx), '    assemble(curl(b)*dx)'
    # print '               Approx solution curl  ', assemble(curl(ba)*dx), '    assemble(curl(ba)*dx)'
    # print '               Error curl            ', assemble(curl(ErrorB)*dx), '    assemble(curl(ErrorB)*dx)'
    # # print '               Error                 ', assemble((ErrorB)*dx), '    assemble((ErrorB)*dx)'
    # print '               Error curl-curl       ', assemble(curl(ErrorB)*curl(ErrorB)*dx), '    assemble(curl(ErrorB)*curl(ErrorB)*dx)'
    # print '               Error inner curl-curl ', assemble(inner(curl(ErrorB),curl(ErrorB))*dx), '    assemble(inner(curl(ErrorB),curl(ErrorB))*dx)'
    tic()
    errL2b= sqrt(abs(assemble(inner(ErrorB, ErrorB)*dx)))
    MO.StrTimePrint("Magnetic L2 error, time: ", toc())
    tic()
    errCurlb = sqrt(abs(assemble(inner(curl(ErrorB),curl(ErrorB))*dx)))
    MO.StrTimePrint("Magnetic Curl error, time: ", toc())
    tic()
    errL2r= sqrt(abs(assemble(inner(ErrorR, ErrorR)*dx)))
    MO.StrTimePrint("Multiplier L2 error, time: ", toc())
    tic()
    errH1r= sqrt(abs(assemble(inner(grad(ErrorR), grad(ErrorR))*dx)))
    MO.StrTimePrint("Multiplier H1 error, time: ", toc())


    # errL2b= errornorm(b, ba, norm_type='L2', degree_rise=4)
    # errCurlb = errornorm(b, ba, norm_type='Hcurl0', degree_rise=4)
    # errL2r= errornorm(r, ra, norm_type='L2', degree_rise=4)
    # errH1r= errornorm(r, ra, norm_type='H10', degree_rise=4)

    return errL2u, errH1u, errL2p, errL2b, errCurlb, errL2r, errH1r




def PicardTolerance(x,u_k,b_k,FSpaces,dim,NormType,iter):
    X = IO.vecToArray(x)
    uu = X[0:dim[0]]
    bb = X[dim[0]+dim[1]:dim[0]+dim[1]+dim[2]]

    u = Function(FSpaces[0])
    u.vector()[:] = u.vector()[:] + uu
    diffu = u.vector().array() - u_k.vector().array()

    b = Function(FSpaces[2])
    b.vector()[:] = b.vector()[:] + bb
    diffb = b.vector().array() - b_k.vector().array()
    if (NormType == '2'):
        epsu = splin.norm(diffu)/sqrt(dim[0])
        epsb = splin.norm(diffb)/sqrt(dim[0])
    elif (NormType == 'inf'):
        epsu = splin.norm(diffu, ord=np.Inf)
        epsb = splin.norm(diffb, ord=np.Inf)
    else:
        print "NormType must be 2 or inf"
        quit()

    print 'iter=%d: u-norm=%g   b-norm=%g ' % (iter, epsu,epsb)
    u_k.assign(u)
    b_k.assign(b)


    return u_k,b_k,epsu,epsb


def PicardToleranceDecouple(x,U,FSpaces,dim,NormType,iter,SaddlePoint = "No"):
    X = IO.vecToArray(x)
    uu = X[0:dim[0]]

    if SaddlePoint == "Yes":
        bb = X[dim[0]:dim[0]+dim[1]]
        pp = X[dim[0]+dim[1]:dim[0]+dim[1]+dim[2]]
    else:
        pp = X[dim[0]:dim[0]+dim[1]]
        bb = X[dim[0]+dim[1]:dim[0]+dim[1]+dim[2]]

    rr = X[dim[0]+dim[1]+dim[2]:]

    u = Function(FSpaces[0])
    u.vector()[:] = uu
    u_ = assemble(inner(u,u)*dx)
    diffu = u.vector().array()

    # if SaddlePoint == "Yes":
    #     p = Function(FSpaces[2])
    #     p.vector()[:] = pp
    #     ones = Function(FSpaces[2])
    #     ones.vector()[:]=(0*ones.vector().array()+1)
    #     pp = Function(FSpaces[2])
    #     print ones
    #     pp.vector()[:] = p.vector().array()- assemble(p*dx)/assemble(ones*dx)
    #     p = pp.vector().array()
    #     b = Function(FSpaces[1])
    #     b.vector()[:] = bb
    #     diffb = b.vector().array()
    # else:
    print pp.shape
    p = Function(FSpaces[1])
    print FSpaces[1].dim()
    p.vector()[:] = pp
    p_ = assemble(p*p*dx)

    ones = Function(FSpaces[1])
    ones.vector()[:]=(0*ones.vector().array()+1)
    pp = Function(FSpaces[1])
    pp.vector()[:] = p.vector().array() - assemble(p*dx)/assemble(ones*dx)
    p_ = assemble(p*p*dx)
    p = pp.vector().array()
    b = Function(FSpaces[2])
    b.vector()[:] = bb
    b_ = assemble(inner(b,b)*dx)
    diffb = b.vector().array()

    r = Function(FSpaces[3])
    r.vector()[:] = rr
    r_ = assemble(r*r*dx)
    # print diffu
    if (NormType == '2'):
        epsu = splin.norm(diffu)/sqrt(dim[0])
        epsp = splin.norm(pp.vector().array())/sqrt(dim[1])
        epsb = splin.norm(diffb)/sqrt(dim[2])
        epsr = splin.norm(r.vector().array())/sqrt(dim[3])
    elif (NormType == 'inf'):
        epsu = splin.norm(diffu, ord=np.Inf)
        epsp = splin.norm(pp.vector().array(),ord=np.inf)
        epsb = splin.norm(diffb, ord=np.Inf)
        epsr = splin.norm(r.vector().array(),ord=np.inf)

    else:
        print "NormType must be 2 or inf"
        quit()
    # U.axpy(1,x)
    p = Function(FSpaces[1])
    RHS = IO.vecToArray(U+x)

    if SaddlePoint == "Yes":
        u.vector()[:] = RHS[0:dim[0]]
        p.vector()[:] = pp.vector().array()+U.array[dim[0]+dim[1]:dim[0]+dim[1]+dim[2]]
        b.vector()[:] = RHS[dim[0]:dim[0]+dim[1]]
        r.vector()[:] = RHS[dim[0]+dim[1]+dim[2]:]
    else:
        u.vector()[:] = RHS[0:dim[0]]
        p.vector()[:] = pp.vector().array()+U.array[dim[0]:dim[0]+dim[1]]
        b.vector()[:] = RHS[dim[0]+dim[1]:dim[0]+dim[1]+dim[2]]
        r.vector()[:] = RHS[dim[0]+dim[1]+dim[2]:]




    print 'u-norm=%g   p-norm=%g  \n b-norm=%g   r-norm=%g' % (epsu,epsp,epsb,epsr), '\n\n\n'
    print 'u-norm=%g   p-norm=%g  \n b-norm=%g   r-norm=%g' % (u_,p_,b_,r_), '\n\n\n'

    return u,p,b,r,epsu+epsp+epsb+epsr




def u_prev(u,p,b,r):
    uOld = np.concatenate((u.vector().array(),p.vector().array(),b.vector().array(),r.vector().array()), axis=0)
    x = IO.arrayToVec(uOld)
    return x
