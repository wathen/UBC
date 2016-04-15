from dolfin import *
import numpy as np

mesh = UnitSquareMesh(8,8)
parameters["form_compiler"]["no-evaluate_basis_derivatives"] = False
Q = FunctionSpace(mesh, "CG", 2)

element = Q.dolfin_element()

basis =np.zeros(2*element.space_dimension()*element.value_dimension(0))
coords = np.array((1.0,0.0))
cell = Cell(mesh, 0)
vc = cell.get_vertex_coordinates()


element.evaluate_basis_derivatives_all(1,basis, coords, vc, 0)

print basis
