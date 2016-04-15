#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = - 2*(x[1]*x[1]-x[1])-2*(x[0]*x[0]-x[0]);
  }
};

// Normal derivative (Neumann boundary condition)


// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary;
  }
};

int main()
{
  // Create mesh and function space
  UnitSquareMesh mesh(32, 32);
  Poisson::FunctionSpace V(mesh);

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Define variational forms
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  Source f;
  L.f = f;

  // Compute solution
  Function u(V);
  solve(a == L, u, bc);

  // Plot solution
  plot(u);
  interactive();

  return 0;
}
