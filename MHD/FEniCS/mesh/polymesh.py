from dolfin import *
import ipdb
if not has_cgal():
    print "DOLFIN must be compiled with CGAL to run this demo."
    exit(0)

# Create empty Mesh
mesh = Mesh()

domain_vertices = [Point(1.0, -1.0),
                   Point(6.0, -1.0),
                   Point(6.0, 1.0),
                   Point(1.0, 1.0),
                   Point(1.0, 6.0),
                   Point(-1.0, 6.0),
                   Point(-1.0, 1.0),
                   Point(-6.0, 1.0),
                   Point(-6.0, -1.0),
                   Point(-1.0, -1.0),
                   Point(-1.0, -6.0),
                   Point(1.0, -6.0),
                   Point(1.0,-1.0),
                   Point(1.0, -1.0)]

# Generate mesh and plot
PolygonalMeshGenerator.generate(mesh, domain_vertices, 0.5);


cell_markers = CellFunction("bool", mesh)
cell_markers.set_all(False)
origin = Point(0.0, 0.0)

for cell in cells(mesh):
  p = cell.midpoint()
  # print p
  if p.distance(origin) < 2:
      cell_markers[cell] = True


mesh = refine(mesh, cell_markers)


cell_markers = CellFunction("bool", mesh)
cell_markers.set_all(False)
origin = Point(0.0, 0.0)

for cell in cells(mesh):
  p = cell.midpoint()
  # print p
  if p.distance(origin) < 1:
      cell_markers[cell] = True




mesh = refine(mesh, cell_markers)

class noflow1(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-6.0,-1.0)) and near(x[0],1.0)


class noflow4(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (1.0, 6.0)) and near(x[0],1.0)


class noflow5(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-6.0,-1.0)) and near(x[0],-1.0)


class noflow8(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (1.0, 6.0)) and near(x[0],-1.0)


class noflow2(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (1.0,6.0)) and near(x[1],-1.0)


class noflow3(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (1.0,6.0)) and near(x[1],1.0)


class noflow6(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-6.0,-1.0)) and near(x[1],-1.0)


class noflow7(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-6.0,-1.0)) and near(x[1],1.0)


Noflow1 = noflow1()
Noflow2 = noflow2()
Noflow3 = noflow3()
Noflow4 = noflow4()
Noflow5 = noflow5()
Noflow6 = noflow6()
Noflow7 = noflow7()
Noflow8 = noflow8()


class inflow1(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (-1.0,1.0)) and near(x[1],-6.0)

class inflow2(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (-1.0,1.0)) and near(x[1],6.0)

class outflow1(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-1.0,1.0)) and near(x[0],6.0)

class outflow2(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-1.0,1.0)) and near(x[0],-6.0)

boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)

Noflow1.mark(boundaries, 1)
Noflow2.mark(boundaries, 1)
Noflow3.mark(boundaries, 1)
Noflow4.mark(boundaries, 1)
Noflow5.mark(boundaries, 1)
Noflow6.mark(boundaries, 1)
Noflow7.mark(boundaries, 1)
Noflow8.mark(boundaries, 1)

Inflow1 = inflow1()
Inflow2 = inflow2()

Outlow1 = outflow1()
Outlow2 = outflow2()

Inflow1.mark(boundaries, 2)
Inflow2.mark(boundaries, 2)

Outlow1.mark(boundaries, 3)
Outlow2.mark(boundaries, 3)

# plot(boundaries, title="S_t1 subdomains",interactive=True)

# file = File("CrossMesh.xml")
# file << mesh

plot(mesh, interactive=True)

