from dolfin import *
import numpy as np
import pandas as pd
n = 6
Dim = np.zeros((n,1))
ErrorL2 = np.zeros((n,1))
ErrorH1 = np.zeros((n,1))
OrderL2 = np.zeros((n,1))
OrderH1 = np.zeros((n,1))
# parameters['reorder_dofs_serial'] = False

for x in range(1,n+1):
    parameters['form_compiler']['quadrature_degree'] = -1
    mesh = UnitSquareMesh(2**x,2**x)
    V = VectorFunctionSpace(mesh, "CG", 2)

    class u_in(Expression):
        def __init__(self):
            self.p = 1
        def eval_cell(self, values, x, ufc_cell):
            values[0] = x[0]*x[0]*x[0]
            values[1] = x[1]*x[1]*x[1]
        def value_shape(self):
            return (2,)
    class F_in(Expression):
        def __init__(self):
            self.p = 1
        def eval_cell(self, values, x, ufc_cell):
            values[0] = -6*x[0]
            values[1] = -6*x[1]
        def value_shape(self):
            return (2,)

    u0 = u_in()
    F = F_in()

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(F, v)*dx

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u0, boundary)
    u = Function(V)

    solve(a == L, u, bcs=bc,
      solver_parameters={"linear_solver": "lu"},
      form_compiler_parameters={"optimize": True})
    parameters['form_compiler']['quadrature_degree'] = 8

    Vexact = VectorFunctionSpace(mesh, "CG", 4)
    ue = interpolate(u0, Vexact)

    e = ue - u
    Dim[x-1] = V.dim()
    ErrorL2[x-1] = sqrt(abs(assemble(inner(e,e)*dx)))
    ErrorH1[x-1] = sqrt(abs(assemble(inner(grad(e),grad(e))*dx)))

    if (x > 1):
        OrderL2[x-1] = abs(np.log2(ErrorL2[x-1]/ErrorL2[x-2]))
        OrderH1[x-1] = abs(np.log2(ErrorH1[x-1]/ErrorH1[x-2]))

TableTitles = ["DoF","L2-erro","L2-order","H1-error","H1-order"]
TableValues = np.concatenate((Dim,ErrorL2,OrderL2,ErrorH1,OrderH1),axis=1)
Table = pd.DataFrame(TableValues, columns = TableTitles)
pd.set_option('precision',3)
print Table
