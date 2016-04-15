from dolfin import *
import ipdb

# Create empty Mesh
# n = 32;
# mesh = RectangleMesh(-1, -1, 1, 1, n, n,'crossed')

# cell_markers = CellFunction("bool", mesh)
# cell_markers.set_all(False)
# origin = Point(0.0, 0.0)

# for cell in cells(mesh):
#   p = cell.midpoint()
#   # print p
#   if p.distance(origin) < 2:
#       cell_markers[cell] = True


# mesh = refine(mesh, cell_markers)


# cell_markers = CellFunction("bool", mesh)
# cell_markers.set_all(False)
# origin = Point(0.5, 0.5)

# for cell in cells(mesh):
#   p = cell.midpoint()
#   # print p
#   if p.distance(origin) < .5:
#       cell_markers[cell] = True

# origin = Point(0.5, -0.5)

# for cell in cells(mesh):
#   p = cell.midpoint()
#   # print p
#   if p.distance(origin) < .35:
#       cell_markers[cell] = True

# origin = Point(-0.5, 0.5)

# for cell in cells(mesh):
#   p = cell.midpoint()
#   # print p
#   if p.distance(origin) < .15:
#       cell_markers[cell] = True


# origin = Point(-0.5, -0.5)

# for cell in cells(mesh):
#   p = cell.midpoint()
#   # print p
#   if p.distance(origin) < .2:
#       cell_markers[cell] = True


# mesh = refine(mesh, cell_markers)


# # plot(boundaries, title="S_t1 subdomains",interactive=True)

# # file = File("CrossMesh.xml")
# # file << mesh

# plot(mesh, interactive=True)


domain_vertices = [Point(0.0, 0.0),
                 Point(10.0, 0.0),
                 Point(10.0, 2.0),
                 Point(8.0, 2.0),
                 Point(7.5, 1.0),
                 Point(2.5, 1.0),
                 Point(2.0, 4.0),
                 Point(0.0, 4.0),
                 Point(0.0, 0.0)]

# Create empty Mesh
mesh = Mesh()
PolygonalMeshGenerator.generate(mesh, domain_vertices, 0.1);
plot(mesh, interactive=True)