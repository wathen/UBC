from dolfin import *
from FIAT import *
import numpy

def tangential_edge_integral(mesh, f, quad_degree):
    """
    This function uses quadrature to compute the average of
    of the dot product  f.t  on each edge of the given mesh.
    Here f is a given vector valued function and t is the vector
    pointing from the first to the second vertex of the edge.
    The returned value is a numpy array edge_int where
        edge_int[edge.index()] = avg of f.t on that edge
    """
    # Reference interval
    L = reference_element.DefaultLine()
    # Coordinates for the reference interval
    x = numpy.array(L.get_vertices())
    # Make quadrature rule on the line
    Q = make_quadrature(L, quad_degree)
    # Quadrature points
    qp = Q.get_points()
    # Quadrature weights
    qw = Q.get_weights()
    # Get mesh vertex coordinates
    coor = mesh.coordinates()
    # Create an array to save the result
    edge_int = numpy.zeros(mesh.num_edges())
    # Integrate over all the edges
    for e in edges(mesh):
        # Extract end points of the edge
        end_pts = e.entities(0)
        y = numpy.vstack([coor[end_pts[0]], coor[end_pts[1]]])
        # Compute the tangent.  Following the FEniCS convention this
        # is the vector pointing from the first to the end point
        # of the edge.
        t = y[1] - y[0]
        # Make the affine map from the reference line
        (A, b) = reference_element.make_affine_mapping(x, y)
        # Quadrature points on the physical edge
        phys_qp = map(lambda x: A.dot(x)+b, qp)
        # Evaluate f.t at quadrature points
        vals = map(lambda x: t.dot(f(x)), phys_qp)
        # Approximate average of f.t on the edge using the quadrature rule
        edge_int[e.index()] = qw.dot(vals)/2.0
    return edge_int

def build_edge2dof_map(V):
    """
    This function takes a N1Curl(1) space and return an integer valued array edge2dof.
    This array has the number of edges as its length. In particular
        edge2dof[i] = j
    means that dof #i, that is u.vector()[i], is associated to edge #j.
    """
    # Extract the cell to edge map (given an cell index, it returns the indices of its edges)
    cell2edges = V.mesh().topology()(3, 1)
    # Extract the cell dofmap (given a cell index, it returns the dof numbers)
    cell2dofs = V.dofmap().cell_dofs
    # Array to save the result
    edge2dof = numpy.zeros(mesh.num_edges(), dtype="int")
    # Iterate over cells, associating the edges to the dofs for that cell
    for c in range(mesh.num_cells()):
        # get the global edge numbers for this cell
        c_edges = cell2edges(c)
        # get the global dof numbers for this cell
        c_dofs = cell2dofs(c)
        # associate the edge numbers to the corresponding dof numbers
        edge2dof[c_dofs] = c_edges
    # This algorithm might not look fast as it does quite some redundant work. In actual
    # runs, for most meshes, this is not the most time consuming step and does not take
    # more than a milisecond.
    return edge2dof

def n1curl_1_canonical_projection(f, mesh, quad_degree):
    # Initialize the mesh
    mesh.init()
    # Compute the average edge integrals
    edge_int = tangential_edge_integral(mesh, f, quad_degree)
    # Create the return value
    V = FunctionSpace(mesh, "N1curl", 1)
    u = Function(V)
    # Permute the average edge integrals to match the order of the dofs.
    u.vector()[:] = edge_int.take(build_edge2dof_map(V))
    return u

f = Expression(("1.0 - x[1] + 2*x[2]", "3.0 + x[0] + 3*x[2]", "2.0 - 2*x[0] - 3*x[1]"))
domain = Box(0., 0., 0., 1., 1., 1.)
n = 8; mesh = BoxMesh(0., 0., 0., 1., 1., 1., n,n,n)
quad_degree = 2
V = FunctionSpace(mesh, "N1curl", 1)
u = n1curl_1_canonical_projection(f, mesh, quad_degree)
v = interpolate(f, V)

print(assemble(dot(u-v,u-v)*dx))