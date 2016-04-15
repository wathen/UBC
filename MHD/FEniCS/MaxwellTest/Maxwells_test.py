from dolfin import *
from numpy import *
import scipy as Sci
import scipy.linalg
from math import pi,sin,cos,sqrt
import scipy.sparse as sps
import scipy.io as save
import scipy
import pdb
from matplotlib.pyplot import *
import pandas as pd


parameters["std_out_all_processes"] = False;
j = 1
n = 2
m = 11

time = zeros((m-1,1))
N = zeros((m-1,1))
error = zeros((m-1,1))
NumCells = zeros((m-1,1))

for j in xrange(1,m):
    # j = j+1
    n =2*n
    N[j-1,0] = n
    print n
    mesh = UnitSquareMesh(n,n)

    NumCells[j-1,0] = mesh.num_cells()
    # mesh = Mesh("lshape.xml.gz")
    V = FunctionSpace(mesh, "N1curl", 1)
    # File("mesh.xml") << mesh


    # Define basis and bilinear form
    u = TrialFunction(V)
    v = TestFunction(V)

    c = 1



    # bc = inner(cross(N,u),v)*ds
    #F = Assemble(rhs)
    # print "starting assemble"
    tic()
    V = FunctionSpace(mesh, "N1curl", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f=Expression(("2+x[1]*(1-x[1])","2+x[0]*(1-x[0])"))
    # f= Expression(("(8*pow(pi,2)+1)*sin(2*pi*x[1])*cos(2*pi*x[0])","-(8*pow(pi,2)+1)*sin(2*pi*x[0])*cos(2*pi*x[1])"))
    ue = Expression(("x[1]*(1-x[1])","x[0]*(1-x[0])"))

    u0 = Expression(('0','0'))
    def u0_boundary(x, on_boundary):
        # N = FacetNormal(mesh)
        return on_boundary
    bc = DirichletBC(V, u0, u0_boundary)
    a = dolfin.inner(curl(u),curl(v))*dx+c*dolfin.inner(u,v)*dx
    b = dolfin.inner(f,v)*dx
    A, bb = assemble_system(a, b, bc)
    time[j-1,0] = toc()
    # solver = KrylovSolver("tfqmr", "amg")
    # solver.set_operators(A, A)
    print bb
    # Compute solution
    u = Function(V)
    # print "solve"
    # solve(A,u.vector(),bb)

    # set_log_level(PROGRESS)
    # solver = KrylovSolver("cg","amg")
    # solver.parameters["relative_tolerance"] = 1e-10
    # solver.parameters["absolute_tolerance"] = 1e-7
    # solver.solve(A,u.vector(),bb)
    # set_log_level(PROGRESS)

    # A, bb = assemble_system(a, b)

    # parameters.linear_algebra_backend = "uBLAS"
    # A, bb = assemble_system(a, b)
    # print "store matrix"

    # rows, cols, values = A.data()
    # rows1, values1 = bb.data()

    # Aa = sps.csr_matrix((values, cols, rows))
    # b = sps.csr_matrix((values1, cols1, rows1))
    # print "save matrix"
    # scipy.io.savemat("Ab.mat", {"A": Aa, "b":bb.data()},oned_as='row')

    # file = File("maxwells.xml")
    # file << u
    #

    # err = ue - u
    # L2normerr = sqrt(assemble(dolfin.inner(err,err)*dx))
    # print n,L2normerr
    # error[j-1,0] = L2normerr
    parameters.linear_algebra_backend = "PETSc"
    # plot(u, interactive=True)

print "\n"
A = zeros((m-1,3))
for x in xrange(1,m):
    print x-1,NumCells[x-1,0],time[x-1,0]
    A[x-1,0] = N[x-1,0]
    A[x-1,1] = NumCells[x-1,0]
    # A[x-1,2] = time[x-1,0]
    A[x-1,2] = error[x-1,0]

print A
loglog(NumCells,error)
# loglog(NumCells,time)
A = pd.DataFrame(A)
print A.to_latex()
print A
show()