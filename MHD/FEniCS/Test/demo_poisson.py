"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

and boundary conditions given by

    u(x, y) = 0        for x = 0 or x = 1
du/dn(x, y) = sin(5*x) for y = 0 or y = 1
"""

# Copyright (C) 2007-2011 Anders Logg
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
# First added:  2007-08-16
# Last changed: 2012-11-12

# Begin demo

from dolfin import *
import scipy.sparse as sps
import scipy.io as save
import scipy
import pdb

# parameters.linear_algebra_backend = "uBLAS"

# Create mesh and define function space:
mesh = UnitSquareMesh(500, 500)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
g = Expression("sin(5*x[0])")
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

print "starting assemble"
A, bb = assemble_system(a, L, bc)
# solver = KrylovSolver("tfqmr", "amg")
# solver.set_operators(A, A)

# Compute solution
u = Function(V)
print "solve"
# solve(A,u.vector(),bb)

set_log_level(PROGRESS)
solver = KrylovSolver("cg")
# solver.parameters["relative_tolerance"] = 1e-10
# solver.parameters["absolute_tolerance"] = 1e-7
solver.solve(A,u.vector(),bb)
set_log_level(PROGRESS)


# problem = VariationalProblem(a, L, bc)
# problem.parameters["linear_solver"] = "gmres"aaa
# problem.parameters["preconditioner"] = "ilu"
# u = problem.solve()


parameters.linear_algebra_backend = "uBLAS"
A, bb = assemble_system(a, L, bc)
print "store matrix"

rows, cols, values = A.data()
# rows1, values1 = bb.data()

Aa = sps.csr_matrix((values, cols, rows))
# b = sps.csr_matrix((values1, cols1, rows1))
print "save matrix"
scipy.io.savemat("Ab.mat", {"A": Aa, "b":bb.data()},oned_as='row')

# scipy.io.savemat("b.mat", {"b":  bb.data()},oned_as='row')
# Save solution in VTK format
# file = File("poisson.pvd")
# file << u

# Plot solution
plot(u, interactive=True)
