from dolfin import *

import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
Print = PETSc.Sys.Print
# from MatrixOperations import *
import numpy as np
import ExactSol
import CheckPetsc4py as CP


n = 8
mesh = UnitSquareMesh(n,n)

V = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
M = VectorFunctionSpace(mesh, "CG", 2)

W = MixedFunctionSpace([M,V,P])

(b, u, p) = TrialFunctions(W)
(c, v, q) = TestFunctions(W)


u0, p0,b0, r0, Laplacian, Advection, gradPres,CurlCurl, gradR, NS_Couple, M_Couple = ExactSol.MHD2D(6,2)

R = 1
S = 1
Rm = 1

u_k = Function(V)
b_k = Function(M)
a = (1/R)*inner(grad(u),grad(v))*dx + inner((grad(u)*u_k),v)*dx - div(v)*p*dx - div(u)*q*dx + S/Rm*inner(grad(b),grad(c))*dx - S*(u[0]*b_k[1]-u[1]*b_k[0])*curl(c)*dx + S*(v[0]*b_k[1]-v[1]*b_k[0])*curl(b)*dx
#a = (1/R)*inner(grad(u),grad(v))*dx + inner((grad(u)*u_k),v)*dx - div(v)*p*dx - div(u)*q*dx + S/Rm*inner(curl(b),curl(c))*dx + S/Rm*inner(div(b),div(c))*dx - S*(u[0]*b_k[1]-u[1]*b_k[0])*curl(c)*dx + S*(v[0]*b[1]-v[1]*b[0])*curl(b_k)*dx

f = -(1/R)*Laplacian + Advection + S*(NS_Couple - M_Couple) + S/Rm*CurlCurl
L = inner(-(1/R)*Laplacian + Advection + S*NS_Couple,v)*dx + inner(-S*M_Couple + S/Rm*CurlCurl,c)*dx


def boundary(x, on_boundary):
    return on_boundary

bcu = DirichletBC(W.sub(1), u0, boundary)
bcb = DirichletBC(W.sub(0), b0, boundary)
bc = [bcb,bcu]

b_is = PETSc.IS().createGeneral(range(M.dim()))
u_is = PETSc.IS().createGeneral(range(M.dim(),M.dim()+V.dim()))
    
eps = 1.0           # error measure ||u-u_k||
tol = 1.0E-4     # tolerance
iter = 0            # iteration counter
maxiter = 40       # max no of iterations allowed
while eps > tol  and iter < maxiter:
    iter += 1
    A,b = assemble_system(a,L,bc)
    A, b = CP.Assemble(A,b)
    u = b.duplicate()


    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    pc = ksp.getPC()
    ksp.setType('preonly')
    pc.setType('lu')
    OptDB = PETSc.Options()
    OptDB["pc_factor_mat_ordering_type"] = "rcm"
    OptDB["pc_factor_mat_solver_package"] = "mumps"
    ksp.setFromOptions()
    ksp.setOperators(A)
    
    scale = b.norm()
    b = b/scale
    ksp.solve(b,u)
    u = scale*u
    
    b_k1 = u.getSubVector(b_is).array
    u_k1 = u.getSubVector(u_is).array

    b_norm = np.linalg.norm(b_k1-b_k.vector().array())
    u_norm = np.linalg.norm(u_k1-u_k.vector().array())
    

    print "b-norm :", b_norm, "   u-norm :",u_norm
    B = Function(M)
    U = Function(V)
    B.vector()[:] = b_k1
    U.vector()[:] = u_k1
    eps = b_norm + u_norm
    u_k.assign(U)
    b_k.assign(B)


plot(u_k)
plot(b_k)
plot(interpolate(u0,V))
plot(interpolate(b0,M))
interactive()
