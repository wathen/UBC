from dolfin import *

def CavityMesh2d(n):

    # Create empty Mesh
    mesh = RectangleMesh(-1, -1, 1, 1, n, n,'left')


    class DirichletL(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], -1.0)

    class DirichletR(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0],1.0)

    class DirichletB(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1],-1.0)

    class DirichletT(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1],1.0)


    dirichletL = DirichletL()
    dirichletR = DirichletR()
    dirichletB = DirichletB()

    dirichletT = DirichletT()


    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)

    dirichletL.mark(boundaries, 1)
    dirichletR.mark(boundaries, 1)
    dirichletB.mark(boundaries, 1)
    dirichletT.mark(boundaries, 2)

    return mesh, boundaries
