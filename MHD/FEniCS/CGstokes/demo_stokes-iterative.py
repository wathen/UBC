"""This demo solves the Stokes equations using an iterative linear solver.
Note that the sign for the pressure has been flipped for symmetry."""

# Copyright (C) 2010 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2010-08-08
# Last changed: 2010-08-08

# Begin demo
import petsc4py
import slepc4py
import sys

petsc4py.init(sys.argv)
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
Print = PETSc.Sys.Print
import PETScIO as IO

from dolfin import *


parameters['linear_algebra_backend'] = 'uBLAS'

# Load mesh
mesh = UnitCubeMesh(16, 16, 16)

# Define function spaces
parameters['reorder_dofs_serial'] = False
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
parameters['reorder_dofs_serial'] = False
W = V * Q

# Boundaries
def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
def left(x, on_boundary): return x[0] < DOLFIN_EPS
def top_bottom(x, on_boundary):
    return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

# No-slip boundary condition for velocity
noslip = Constant((0.0, 0.0, 0.0))
bc0 = DirichletBC(W.sub(0), noslip, top_bottom)

# Inflow boundary condition for velocity
inflow = Expression(("-sin(x[1]*pi)", "0.0", "0.0"))
bc1 = DirichletBC(W.sub(0), inflow, right)

# Boundary condition for pressure at outflow
zero = Constant(0)
bc2 = DirichletBC(W.sub(1), zero, left)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0.0, 0.0, 0.0))
a = inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx
L = inner(f, v)*dx

# Form for use in constructing preconditioner matrix
b = inner(grad(u), grad(v))*dx +  p*q*dx

# Assemble system
AA, bb = assemble_system(a, L, bcs)

# Assemble preconditioner system
PP, btmp = assemble_system(b, L, bcs)
As = AA.sparray()
A = PETSc.Mat().createAIJ(size=As.shape,csr=(As.indptr, As.indices, As.data))

Ps = PP.sparray()
P = PETSc.Mat().createAIJ(size=Ps.shape,csr=(Ps.indptr, Ps.indices, Ps.data))
b = bb.array()
zeros = 0*b
del bb
bb = IO.arrayToVec(b)
x = IO.arrayToVec(zeros)
# Create Krylov solver and AMG preconditioner
ksp = PETSc.KSP().create()
pc = PETSc.PC().create()
ksp.setFromOptions()
ksp.setTolerances(1e-10)
print W.dim()
print 'Solving with:', ksp.setType('minres')
# ksp.setPCSide(2)

pc = ksp.getPC()
pc.setOperators(P)
pc.getType()
ksp.setOperators(A,P)
tic()
ksp.solve(bb, x)
SolTime= toc()
print "time to solve: ",SolTime
iterations =  ksp.its
print "iterations = ", iterations
del PP,Pb,Ps,AA,As


# Associate operator (A) and preconditioner matrix (P)
solver.set_operators(A, P)

# Solve
U = Function(W)
solver.solve(U.vector(), bb)

# Get sub-functions
u, p = U.split()

# Save solution in VTK format
ufile_pvd = File("velocity.pvd")
ufile_pvd << u
pfile_pvd = File("pressure.pvd")
pfile_pvd << p

# Plot solution
plot(u)
plot(p)
interactive()
