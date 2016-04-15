from dolfin import Mesh, Point, PolygonalMeshGenerator, plot, interactive

def UnstructuredRectangle(xa,ya,xb,yb, h):
    mesh = Mesh()
    domain_vertices = [Point(xa,ya),
                       Point(xa,yb),
                       Point(xb,yb),
                       Point(xb,ya),
                       Point(xa,ya)]

    PolygonalMeshGenerator.generate(mesh, domain_vertices, h)
    return mesh

def Lshape(x1,x2,x3,y1,y2,y3, h):
    """
        (x1,y3)       (x2,y3)


                          (x2,y2)         (x3,y2)


        (x1,y1)                           (x3,y1)
    """

    mesh = Mesh()
    domain_vertices = [Point(x1,y1),
                       Point(x1,y3),
                       Point(x2,y3),
                       Point(x2,y2),
                       Point(x3,y2),
                       Point(x3,y1),
                       Point(x1,y1)]

    PolygonalMeshGenerator.generate(mesh, domain_vertices, h)
    return mesh



# mesh = Lshape(0,1,2, 0,1,2, .2)
# plot(mesh)
# interactive()
