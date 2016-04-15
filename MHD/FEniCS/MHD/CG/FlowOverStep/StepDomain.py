from dolfin import *

def StepMesh(h):

    # Create empty Mesh
    mesh = Mesh()

    # Create list of polygonal domain vertices
    domain_vertices = [Point(0.0, 0.0),
                       Point(0.0, -0.125),
                       Point(0.75, -0.125),
                       Point(0.75, 0.125),
                       Point(-0.25, 0.125),
                       Point(-0.25, 0.0),
                       Point(0.0, 0.0)]

    # Generate mesh and plot
    PolygonalMeshGenerator.generate(mesh, domain_vertices, h);
    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    origin = Point(-0.0, -0.0)
    # plot(mesh)
    for cell in cells(mesh):
        p = cell.midpoint()
          # print p
        if p.distance(origin) < .075:
            cell_markers[cell] = True

    mesh = refine(mesh, cell_markers)

    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)

    for cell in cells(mesh):
        p = cell.midpoint()
          # print p
        if p.distance(origin) < .05:
            cell_markers[cell] = True


    mesh = refine(mesh, cell_markers)

    # plot(mesh)

    class Neumann(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.75)

    class DirichletIn(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], -0.25)


    class DirichletT(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.125)

    class DirichletB(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], -0.125)

    class DirichletTT(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.0)

    class DirichletLL(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0)

    neumann = Neumann()
    dirichletIn = DirichletIn()
    dirichletT = DirichletT()
    dirichletB = DirichletB()
    dirichletTT = DirichletTT()
    dirichletLL = DirichletLL()


    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)
    neumann.mark(boundaries, 1)
    dirichletIn.mark(boundaries, 2)
    dirichletT.mark(boundaries, 3)
    dirichletB.mark(boundaries, 3)
    dirichletTT.mark(boundaries, 3)
    dirichletLL.mark(boundaries, 3)

    return mesh, boundaries
