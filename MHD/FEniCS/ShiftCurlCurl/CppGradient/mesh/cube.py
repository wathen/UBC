from dolfin import *

mesh = Mesh()
domain_vertices = [Point(0.0, 0.0,0.0),
                              Point(4.0, 0.0, 0.0),
                              Point(4.0, 4.0, 0.0),
                              Point(0.0, 4.0),
                              Point(0.0, 0.0)]
PolygonalMeshGenerator.generate(mesh, domain_vertices, .75);

box = Box(0, 0, 0, 1, 1, 1)
info("\nCompact output of 3D geometry:")
info(box)
info("\nVerbose output of 3D geometry:")
info(box, True)
mesh3d = Mesh(box, 4)
plot(mesh3d)
interactive()
