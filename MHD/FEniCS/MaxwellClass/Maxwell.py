from dolfin import *
from numpy import *
import scipy as Sci
#import scipy.linalg
from math import pi,sin,cos,sqrt
import scipy.sparse as sps
import scipy.io as save
import scipy
import pdb


class Maxwell(object):
    """docstring for Maxwell"""
    def __init__(self, n):
        self.n = n
        self.mesh = self.MeshGenerator(n)

    def MeshGenerator(self, n):
        mesh = UnitSquareMesh(n,n)
        return mesh

    def CreateTrialTestFuncs(self, mesh):
        V = FunctionSpace(mesh, "N1curl", 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        return V,u,v

    def AssembleSystem(self, V,u,v,f,u0,c,BackEnd):
        parameters.linear_algebra_backend = BackEnd
        def u0_boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(V, u0, u0_boundary)
        a = dolfin.inner(curl(u),curl(v))*dx+c*dolfin.inner(u,v)*dx
        b = dolfin.inner(f,v)*dx
        A, bb = assemble_system(a, b, bc)
        return A,bb

    def SolveSystem(self, A,b,V,Solver,precond,absol,relsol,prog):
        if prog == 1:
            set_log_level(PROGRESS)

        u = Function(V)
        solver = KrylovSolver(Solver,precond)
        solver.parameters["relative_tolerance"] = absol
        solver.parameters["absolute_tolerance"] = relsol
        solver.solve(A,u.vector(),b)
        if prog == 1:
            set_log_level(PROGRESS)

        return u

    def StoreSystem(self, A,b):
        rows, cols, values = A.data()
        Aa = sps.csr_matrix((values, cols, rows))
        scipy.io.savemat("Ab.mat", {"A": Aa,"b": b.data()},oned_as='row')

    def SaveEpertaMatrix(self,A,name):
        from PyTrilinos import EpetraExt
        from numpy import array
        import scipy.sparse as sps
        import scipy.io
        test ="".join([name,".txt"])
        EpetraExt.RowMatrixToMatlabFile(test,A)
        data = loadtxt(test)
        col,row,values = data[:,0]-1,data[:,1]-1,data[:,2]
        Asparse = sps.csr_matrix((values, (row, col)))
        testmat ="".join([name,".mat"])
        scipy.io.savemat( testmat, {name: Asparse},oned_as='row')

    def Error(self, u,ue):
        err = ue - u
        L2normerr = sqrt(assemble(dolfin.inner(err,err)*dx))
        return L2normerr


    def LabelStucture(self, Values,Label):
        for x,y in zip(Values,Label):
            d.setdefault(y, []).append(x)
        return d

