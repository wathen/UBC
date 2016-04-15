import os, inspect
from dolfin import *
import numpy
import ExactSol
import MatrixOperations as MO
import CheckPetsc4py as CP
import petsc4py
import sys
import HiptmairPrecond
import HiptmairSetup
petsc4py.init(sys.argv)
from petsc4py import PETSc
import MaxwellPrecond
import pandas as pd
import time

m = 10
errL2b =numpy.zeros((m-1,1))
errCurlb =numpy.zeros((m-1,1))
errL2r =numpy.zeros((m-1,1))
errH1r =numpy.zeros((m-1,1))


l2border =  numpy.zeros((m-1,1))
Curlborder =numpy.zeros((m-1,1))
l2rorder =  numpy.zeros((m-1,1))
H1rorder = numpy.zeros((m-1,1))

Vdim = numpy.zeros((m-1,1))
Qdim = numpy.zeros((m-1,1))
Wdim = numpy.zeros((m-1,1))

ItsSave = numpy.zeros((m-1,1))
ItsHipt = numpy.zeros((m-1,1))
ItsCG = numpy.zeros((m-1,1))
DimSave = numpy.zeros((m-1,1))
TimeSave = numpy.zeros((m-1,1))
NN = numpy.zeros((m-1,1))
Curlgrad = numpy.zeros((m-1,1))
Massgrad = numpy.zeros((m-1,1))
Laplgrad = numpy.zeros((m-1,1))
TimeOrder = numpy.zeros((m-1,1))
dim = 2

for xx in xrange(1,m):
    NN[xx-1] = xx
    parameters["form_compiler"]["quadrature_degree"] = -1
    nn = int(2**(NN[xx-1][0]))
    omega = 1
    tic()
    if dim == 2:
        mesh = UnitSquareMesh(nn,nn)
        u0, p0, CurlCurl, gradPres, CurlMass = ExactSol.M2D(2,Show="yes", Mass = omega)
    else:
        mesh = UnitCubeMesh(int(nn),int(nn),int(nn))
        u0, p0, CurlCurl, gradPres, CurlMass = ExactSol.M3D(1,Show="yes", Mass = omega)
    print ("{:40}").format("Mesh setup, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    order = 1
    parameters['reorder_dofs_serial'] = False
    tic()
    Magnetic = FunctionSpace(mesh, "N1curl", order)
    Lagrange = FunctionSpace(mesh, "CG", order)
    W = Magnetic*Lagrange
    print ("{:40}").format("Function space setup, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])
    Vdim[xx-1] = Magnetic.dim()
    Qdim[xx-1] = Lagrange.dim()
    Wdim[xx-1] = W.dim()
    print "\n\nV:  ",Vdim[xx-1],"Q:  ",Qdim[xx-1],"W:  ",Wdim[xx-1],"\n\n"
    parameters['reorder_dofs_serial'] = False

    DimSave[xx-1] = Magnetic.dim()
    print Magnetic.dim()
    parameters['linear_algebra_backend'] = 'uBLAS'

    C, P = HiptmairSetup.HiptmairMatrixSetupBoundary(mesh, Magnetic.dim(), Lagrange.dim(),dim)
    G, P = HiptmairSetup.HiptmairBCsetupBoundary(C,P,mesh)



    def boundary(x, on_boundary):
        return on_boundary

    bcb = DirichletBC(W.sub(0),u0, boundary)
    bcr = DirichletBC(W.sub(1), p0, boundary)
    bcs = [bcb,bcr]

    (v,q) = TestFunctions(W)
    (u,p) = TrialFunctions(W)

    a11 = inner(curl(u),curl(v))*dx
    a12 = inner(v,grad(p))*dx
    a21 = inner(u,grad(q))*dx
    f = CurlCurl + gradPres
    L1  = inner(v, f)*dx
    a = a11+a12+a21
    tic()
    AA,b = assemble_system(a,L1,bcs)
    print ("{:40}").format("System assembled, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    tic()
    A,b = CP.Assemble(AA,b)
    x = b.duplicate()
    print ("{:40}").format("PETSc system assembled, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    tic()
    M = assemble(inner(u,v)*dx)
    bcb.apply(M)
    Prec = M.sparray()[:Magnetic.dim(),:Magnetic.dim()]+  AA.sparray()[:Magnetic.dim(),:Magnetic.dim()]
    del M, AA
    Prec = PETSc.Mat().createAIJ(size=Prec.shape,csr=(Prec.indptr, Prec.indices, Prec.data))
    print ("{:40}").format("Create CurlCurl+shift, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    tic()
    ScalarLaplacian, b1 = assemble_system(inner(grad(p),grad(q))*dx,inner(p0,q)*dx,bcr)
    VectorLaplacian, b2 = assemble_system(inner(grad(p),grad(q))*dx+inner(p,q)*dx,inner(p0,q)*dx,bcr)
    ScalarLaplacian = ScalarLaplacian.sparray()[Magnetic.dim():,Magnetic.dim():]
    VectorLaplacian = VectorLaplacian.sparray()[Magnetic.dim():,Magnetic.dim():]
    del b1, b2
    print ("{:40}").format("Hiptmair Laplacians BC assembled, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])


    tic()
    VectorLaplacian = PETSc.Mat().createAIJ(size=VectorLaplacian.shape,csr=(VectorLaplacian.indptr, VectorLaplacian.indices, VectorLaplacian.data))
    ScalarLaplacian = PETSc.Mat().createAIJ(size=ScalarLaplacian.shape,csr=(ScalarLaplacian.indptr, ScalarLaplacian.indices, ScalarLaplacian.data))
    print ("{:40}").format("PETSc Laplacians assembled, time: "), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])


    ksp = PETSc.KSP().create()
    # ksp.setTolerances(1e-6)
    ksp.setType('minres')
    ksp.setOperators(A)
    # OptDB = PETSc.Options()
    # OptDB['pc_factor_mat_solver_package']  = 'mumps'
    # OptDB["pc_factor_mat_ordering_type"] = "rcm"
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)
    ksp.setFromOptions()

    CGtol = 1e-2
    Hiptmairtol = 1e-3


    tic()
    kspVector, kspScalar, kspCGScalar, diag = HiptmairSetup.HiptmairKSPsetup(VectorLaplacian, ScalarLaplacian, Prec, CGtol)
    del A, VectorLaplacian, ScalarLaplacian
    print ("{:40}").format("Hiptmair Setup time:"), " ==>  ",("{:4f}").format(toc()),  ("{:9}").format("   time: "), ("{:4}").format(time.strftime('%X %x %Z')[0:5])

    reshist = {}
    def monitor(ksp, its, rnorm):
        print rnorm
        reshist[its] = rnorm


    ksp.setMonitor(monitor)
    pc.setPythonContext(MaxwellPrecond.Hiptmair(W, kspScalar, kspCGScalar, kspVector, G, P, Prec,Hiptmairtol))
    scale = b.norm()
    b = b/scale
    start_time = time.time()
    ksp.solve(b, x)
    TimeSave[xx-1] = time.time() - start_time
    x = x*scale
    print ksp.its
    print TimeSave[xx-1]
    ItsSave[xx-1] = len(reshist)
    CGits, Hits, CGtime, HiptmairTime = pc.getPythonContext().ITS()
    ItsHipt[xx-1] =float(Hits)/len(reshist)
    ItsCG[xx-1] =float(CGits)/len(reshist)

    print " \n\n\n\n"
    Ve = FunctionSpace(mesh,"N1curl",2)
    u = interpolate(u0,Ve)
    Qe = FunctionSpace(mesh,"CG",2)
    p = interpolate(p0,Qe)



    X = x.array
    x = X[0:Magnetic.dim()]
    ua = Function(Magnetic)
    ua.vector()[:] = x

    pp = X[Magnetic.dim():]
    pa = Function(Lagrange)

    pa.vector()[:] = pp

    # parameters["form_compiler"]["quadrature_degree"] = 16

    ErrorB = Function(Magnetic)
    ErrorR = Function(Lagrange)

    ErrorB = u-ua
    ErrorR = p-pa


    errL2b[xx-1] = sqrt(abs(assemble(inner(ErrorB, ErrorB)*dx)))
    errCurlb[xx-1] = sqrt(abs(assemble(inner(curl(ErrorB), curl(ErrorB))*dx)))

    errL2r[xx-1] = sqrt(abs(assemble(inner(ErrorR, ErrorR)*dx)))
    errH1r[xx-1] = sqrt(abs(assemble(inner(grad(ErrorR), grad(ErrorR))*dx)))



    if xx == 1:
        a = 1
    else:

        l2border[xx-1] =  numpy.abs(numpy.log2(errL2b[xx-2]/errL2b[xx-1]))
        Curlborder[xx-1] =  numpy.abs(numpy.log2(errCurlb[xx-2]/errCurlb[xx-1]))

        l2rorder[xx-1] =  numpy.abs(numpy.log2(errL2r[xx-2]/errL2r[xx-1]))
        H1rorder[xx-1] =  numpy.abs(numpy.log2(errH1r[xx-2]/errH1r[xx-1]))
        TimeOrder[xx-1] = (TimeSave[xx-1]/TimeSave[xx-2])/(2**dim)

    print errL2b[xx-1]
    print errCurlb[xx-1]

    print errL2r[xx-1]
    print errH1r[xx-1]



# # # print "\n\n   Magnetic convergence"
# # # MagneticTitles = ["Total DoF","B DoF","Soln Time","Iter","B-L2","B-order","B-Curl","Curl-order"]
# # # MagneticValues = np.concatenate((Wdim,Vdim,SolTime,OuterIt,errL2b,l2border,errCurlb,Curlborder),axis=1)
# # # MagneticTable= pd.DataFrame(MagneticValues, columns = MagneticTitles)
# # # pd.set_option('precision',3)
# # # MagneticTable = MO.PandasFormat(MagneticTable,"B-Curl","%2.4e")
# # # MagneticTable = MO.PandasFormat(MagneticTable,'B-L2',"%2.4e")
# # # print MagneticTable

# # # print "\n\n   Lagrange convergence"
# # # LagrangeTitles = ["Total DoF","R DoF","Soln Time","Iter","R-L2","R-order","R-H1","H1-order"]
# # # LagrangeValues = np.concatenate((Wdim,Qdim,SolTime,OuterIt,errL2r,l2rorder,errH1r,H1rorder),axis=1)
# # # LagrangeTable= pd.DataFrame(LagrangeValues, columns = LagrangeTitles)
# # # pd.set_option('precision',3)
# # # LagrangeTable = MO.PandasFormat(LagrangeTable,'R-L2',"%2.4e")
# # # LagrangeTable = MO.PandasFormat(LagrangeTable,'R-H1',"%2.4e")
# # # print LagrangeTable

LatexTitlesB = ["l","B DoF","R DoF","BB-L2","B-order","BB-Curl","Curl-order"]
LatexValuesB = numpy.concatenate((NN,Vdim,Qdim,errL2b,l2border,errCurlb,Curlborder),axis=1)
LatexTableB= pd.DataFrame(LatexValuesB, columns = LatexTitlesB)
pd.set_option('precision',3)
LatexTableB = MO.PandasFormat(LatexTableB,'BB-Curl',"%2.4e")
LatexTableB = MO.PandasFormat(LatexTableB,'BB-L2',"%2.4e")
LatexTableB = MO.PandasFormat(LatexTableB,'Curl-order',"%2.2f")
LatexTableB = MO.PandasFormat(LatexTableB,'B-order',"%2.2f")
print LatexTableB.to_latex()



LatexTitlesR = ["l","B DoF","R DoF","R-L2","R-order","R-H1","H1-order"]
LatexValuesR = numpy.concatenate((NN,Vdim,Qdim,errL2r,l2rorder,errH1r,H1rorder),axis=1)
LatexTableR= pd.DataFrame(LatexValuesR, columns = LatexTitlesR)
pd.set_option('precision',3)
LatexTableR = MO.PandasFormat(LatexTableR,'R-L2',"%2.4e")
LatexTableR = MO.PandasFormat(LatexTableR,'R-H1',"%2.4e")
LatexTableR = MO.PandasFormat(LatexTableR,'R-order',"%2.2f")
LatexTableR = MO.PandasFormat(LatexTableR,'H1-order',"%2.2f")
print LatexTableR.to_latex()


print "\n\n\n"
ItsTitlesB = ["l","DoF","Time","Time Scalability","MINRES Its", "Hiptmair its", "CG its"]
ItsValuesB = numpy.concatenate((NN,Wdim,TimeSave,TimeOrder,ItsSave, ItsHipt, ItsCG),axis=1)
ItsTableB= pd.DataFrame(ItsValuesB, columns = ItsTitlesB)
pd.set_option('precision',5)
ItsTableB = MO.PandasFormat(ItsTableB,'MINRES Its',"%3.0i")
ItsTableB = MO.PandasFormat(ItsTableB,'Hiptmair its',"%3.1f")
ItsTableB = MO.PandasFormat(ItsTableB,'CG its',"%3.1f")
print ItsTableB.to_latex()



