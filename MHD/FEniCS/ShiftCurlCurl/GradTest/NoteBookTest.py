# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from dolfin import *

#!/usr/bin/python
import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np

# <codecell>

nn = 8
mesh = RectangleMesh(0, 0, 1, 1, nn, nn,'left')

order  = 2
Magnetic = FunctionSpace(mesh, "N1curl", order)
Lagrange = FunctionSpace(mesh, "CG", order)

# <codecell>

class Constraint:
    """
    Constraint implements a tie between the values at two points p1 and p2.

    Example:

    Create a tie between the values at (0.0, 0.5) and (1.0, 0.5)

    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, 'CG', 2)
    constraint = Constraint(V)
    tie = constraint.vector(Point(0.0, 0.5), Point(1.0, 0.5))

    The constraint equation is given by tie.inner(u.vector()) == 0,
    i.e. all ties span the nullspace of the linear equation.
    """
    def __init__(self, V):
        self.V = V
        self.mesh = V.mesh()
        self.dofmap = V.dofmap()
        self.finite_element = V.element()
    def evaluate_basis(self, p):
#         import numpy as np
        bbt = mesh.bounding_box_tree()
        id = bbt.compute_first_entity_collision(p)
        if id >= mesh.num_cells():
            id = bbt.compute_closest_entity(p)[0]
        c = Cell(self.mesh, id)
        vc = c.get_vertex_coordinates()
        dofs = self.dofmap.cell_dofs(id)
        no_basis_fns = self.finite_element.space_dimension()
        value_dimension = self.finite_element.value_dimension(0)
        print no_basis_fns, value_dimension
        basis = np.zeros((no_basis_fns, value_dimension))
        coords = np.zeros(2)
        coords[0], coords[1] = p.x(), p.y()
        self.finite_element.evaluate_basis_derivatives_all(1,basis, coords, vc, 0)
        u = Function(self.V)
        v = u.vector()
        # fixme: implement mixed spaces
        for k in range(value_dimension):
            for j in range(no_basis_fns):
                l = no_basis_fns*(k-1)+j
                v[dofs[l]] = basis[j][k]
        return v

# <codecell>

C = Constraint(Lagrange)

# <codecell>


print C.evaluate_basis(Point(.1,.1))


