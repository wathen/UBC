from dolfin import *
import PETScIO as IO
import numpy as np
import scipy.linalg as splin
import scipy
import petsc4py
import sys
import time
petsc4py.init(sys.argv)
import matplotlib.pylab as plt
from petsc4py import PETSc

def StoreMatrix(A,name):
      test ="".join([name,".mat"])
      scipy.io.savemat( test, {name: A},oned_as='row')

def remove_ij(x, i, j):

    # Remove the ith row
    idx = range(x.shape[0])
    idx.remove(i)
    x = x[idx,:]

    # Remove the jth column
    idx = range(x.shape[1])
    idx.remove(j)
    x = x[:,idx]
    # x.eliminate_zeros()
    # A = PETSc.Mat().createAIJ(size=x.shape,csr=(x.indptr, x.indices, x.data))
    return x

def RemoveRowCol(AA,bb,VelPres):
    As = AA.sparray()
    As.eliminate_zeros()
    Adelete = remove_ij(As,VelPres-1,VelPres-1)
    A = PETSc.Mat().createAIJ(size=Adelete.shape,csr=(Adelete.indptr, Adelete.indices, Adelete.data))
    # StoreMatrix(Adelete, "As")
    b = np.delete(bb,VelPres-1,0)
    zeros = 0*b
    bb= IO.arrayToVec(b)
    x = IO.arrayToVec(zeros)
    return A,bb,x


def Errors(x,mesh,FSpaces,ExactSolution,k,dim):
    Vdim = dim[0]
    Pdim = dim[1]
    Mdim = dim[2]
    Rdim = dim[3]
    # k +=2
    VelocityE = VectorFunctionSpace(mesh,"CG",k+4)
    u = interpolate(ExactSolution[0],VelocityE)

    PressureE = FunctionSpace(mesh,"CG",k+3)


    MagneticE = FunctionSpace(mesh,"N1curl",k+4)
    b = interpolate(ExactSolution[2],MagneticE)

    LagrangeE = FunctionSpace(mesh,"CG",k+4)
    r = interpolate(ExactSolution[3],LagrangeE)
    # parameters["form_compiler"]["quadrature_degree"] = 14
    X = IO.vecToArray(x)
    xu = X[0:Vdim]
    ua = Function(FSpaces[0])
    ua.vector()[:] = xu

    pp = X[Vdim:Vdim+Pdim-1]


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


    plot(ua)
    plot(pp)
    plot(ba)
    plot(ra)

    ErrorU = u-ua
    ErrorP = pe-pp
    ErrorB = b-ba
    ErrorR = r-ra

    # errL2u = errornorm(u,ua ,norm_type='L2',degree_rise=4)
    # errH1u = errornorm(u,ua ,norm_type='H10',degree_rise=4)
    # errL2p = errornorm(pe,pp ,norm_type='L2',degree_rise=4)
    # errL2b = errornorm(b,ba ,norm_type='L2',degree_rise=4)
    # errCurlb  =errornorm(b,ba ,norm_type='Hcurl0',degree_rise=4)
    # errL2r = errornorm(r,ra ,norm_type='L2',degree_rise=4)
    # errH1r = errornorm(r,ra ,norm_type='H10',degree_rise=4)
    errL2u= sqrt(abs(assemble(inner(ErrorU, ErrorU)*dx)))
    errH1u= sqrt(abs(assemble(inner(grad(ErrorU), grad(ErrorU))*dx)))
    errL2p= sqrt(abs(assemble(inner(ErrorP, ErrorP)*dx)))
    errL2b= sqrt(abs(assemble(inner(ErrorB, ErrorB)*dx)))
    errCurlb = sqrt(abs(assemble(inner(curl(ErrorB), curl(ErrorB))*dx)))
    errL2r= sqrt(abs(assemble(inner(ErrorR, ErrorR)*dx)))
    errH1r= sqrt(abs(assemble(inner(grad(ErrorR), grad(ErrorR))*dx)))

    return errL2u, errH1u, errL2p, errL2b, errCurlb, errL2r, errH1r




def PicardTolerance(x,u_k,b_k,FSpaces,dim,NormType,iter):
    X = IO.vecToArray(x)
    uu = X[0:dim[0]]
    bb = X[dim[0]+dim[1]-1:dim[0]+dim[1]+dim[2]-1]

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


def PicardToleranceDecouple(x,U,FSpaces,dim,NormType,iter):
    X = IO.vecToArray(x)
    uu = X[0:dim[0]]
    pp = X[dim[0]:dim[0]+dim[1]-1]
    bb = X[dim[0]+dim[1]-1:dim[0]+dim[1]+dim[2]-1]
    rr = X[dim[0]+dim[1]+dim[2]-1:]

    u = Function(FSpaces[0])
    u.vector()[:] = uu
    diffu = u.vector().array()

    p = Function(FSpaces[1])
    n = pp.shape
    p.vector()[:] = np.insert(pp,n,0)
    # ones = Function(FSpaces[1])
    # ones.vector()[:]=(0*ones.vector().array()+1)
    # pp = Function(FSpaces[1])
    # p.vector()[:] = p.vector().array()- assemble(p*dx)/assemble(ones*dx)

    b = Function(FSpaces[2])
    b.vector()[:] = bb
    diffb = b.vector().array()

    r = Function(FSpaces[3])
    print r.vector().array().shape
    print rr.shape
    r.vector()[:] = rr


    if (NormType == '2'):
        epsu = splin.norm(diffu)/sqrt(dim[0])
        epsp = splin.norm(p.vector().array())/sqrt(dim[1])
        epsb = splin.norm(diffb)/sqrt(dim[2])
        epsr = splin.norm(r.vector().array())/sqrt(dim[3])
    elif (NormType == 'inf'):
        epsu = splin.norm(diffu, ord=np.Inf)
        epsp = splin.norm(p.vector().array(),ord=np.inf)
        epsb = splin.norm(diffb, ord=np.Inf)
        epsr = splin.norm(r.vector().array(),ord=np.inf)
    else:
        print "NormType must be 2 or inf"
        quit()


    RHS = IO.vecToArray(U)
    u.vector()[:] = uu + RHS[0:dim[0]]
    p.vector()[:] = p.vector().array() + U.array[dim[0]:dim[0]+dim[1]]
    b.vector()[:] = bb + RHS[dim[0]+dim[1]:dim[0]+dim[1]+dim[2]]
    r.vector()[:] = rr + RHS[dim[0]+dim[1]+dim[2]:]




    print 'iter=%d:\n u-norm=%g   p-norm=%g  \n b-norm=%g   r-norm=%g' % (iter, epsu,epsp,epsb,epsr), '\n\n\n'
    return u,p,b,r,epsu+epsp+epsb+epsr

def u_prev(u,p,b,r):
    uOld = np.concatenate((u.vector().array(),p.vector().array(),b.vector().array(),r.vector().array()), axis=0)
    x = IO.arrayToVec(uOld)
    return x