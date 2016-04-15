import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import HiptmairSetup
from dolfin import tic, toc
class BaseMyPC(object):
    def setup(self, pc):
        pass
    def reset(self, pc):
        pass
    def apply(self, pc, x, y):
        raise NotImplementedError
    def applyT(self, pc, x, y):
        self.apply(pc, x, y)
    def applyS(self, pc, x, y):
        self.apply(pc, x, y)
    def applySL(self, pc, x, y):
        self.applyS(pc, x, y)
    def applySR(self, pc, x, y):
        self.applyS(pc, x, y)
    def applyRich(self, pc, x, y, w, tols):
        self.apply(pc, x, y)

class Direct(BaseMyPC):

    def __init__(self, W):
        self.W = W
        self.u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
        self.p_is = PETSc.IS().createGeneral(range(W.sub(0).dim(),W.sub(0).dim()+W.sub(1).dim()))
    def create(self, pc):
        OptDB = PETSc.Options()
        OptDB["pc_factor_mat_ordering_type"] = "rcm"
#        OptDB['pc_factor_mat_solver_package']  = "pastix"
        
        kspCurlCurl = PETSc.KSP()
        kspCurlCurl.create(comm=PETSc.COMM_WORLD)
        pcCurlCurl = kspCurlCurl.getPC()
        kspCurlCurl.setType('preonly')
        pcCurlCurl.setType('lu')
        kspCurlCurl.setFromOptions()
        self.kspCurlCurl = kspCurlCurl

        kspL = PETSc.KSP()
        kspL.create(comm=PETSc.COMM_WORLD)
        pcL = kspCurlCurl.getPC()
        kspL.setType('preonly')
        pcL.setType('lu')
        kspL.setFromOptions()
        self.kspL = kspL
        self.kspL.setFromOptions()


    def setUp(self, pc):
        A, P = pc.getOperators()
        CurlCurl = P.getSubMatrix(self.u_is,self.u_is)
        Laplace = P.getSubMatrix(self.p_is,self.p_is)

        self.kspCurlCurl.setOperators(CurlCurl)
        self.kspL.setOperators(Laplace)


    def apply(self, pc, x, y):
        # self.kspCurlCurl.setOperators(self.B)
        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()

        self.kspCurlCurl.solve(x1, y1)
        self.kspL.solve(x2, y2)

        # print y1.array

        y.array = (np.concatenate([y1.array, y2.array]))














class Approx(BaseMyPC):

    def __init__(self, W):
        self.W = W
        self.u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
        self.p_is = PETSc.IS().createGeneral(range(W.sub(0).dim(),W.sub(0).dim()+W.sub(1).dim()))
    def create(self, pc):
        self.diag = None
        kspCurlCurl = PETSc.KSP()
        kspCurlCurl.create(comm=PETSc.COMM_WORLD)
        pcCurlCurl = kspCurlCurl.getPC()
        kspCurlCurl.setType('cg')
        pcCurlCurl.setType('hypre')
        # pc.setFactorSolverPackage("umfpack")
        kspCurlCurl.rtol = 1e-5
        kspCurlCurl.max_it=1000000
        OptDB = PETSc.Options()

        # OptDB["pc_factor_mat_ordering_type"] = "rcm"
        # OptDB["pc_factor_levels"] = 2
        # OptDB["pc_hypre_boomeramg_grid_sweeps_all"] = 3
        # OptDB[""]
        # OptDB["pc_hypre_boomeramg_interp_type"] = "FF1"
        # OptDB["pc_hypre_boomeramg_coarsen_type"] = "PMIS"
        # kspCurlCurl.max_it = 1
        kspCurlCurl.setFromOptions()
        self.kspCurlCurl = kspCurlCurl
        # print kspCurlCurl.view()

        kspLapl = PETSc.KSP()
        kspLapl.create(comm=PETSc.COMM_WORLD)
        pcLapl = kspLapl.getPC()
        kspLapl.setType('richardson')
        pcLapl.setType('hypre')
        # pc.setFactorSolverPackage("umfpack")
        kspLapl.max_it = 1
        kspLapl.setFromOptions()
        self.kspLapl = kspLapl
        # print kspLapl.view()


    def setUp(self, pc):
        A, P, flag = pc.getOperators()
        CurlCurl = P.getSubMatrix(self.u_is,self.u_is)
        Laplace = P.getSubMatrix(self.p_is,self.p_is)

        self.kspCurlCurl.setOperators(CurlCurl)
        self.kspLapl.setOperators(Laplace)


    def apply(self, pc, x, y):
        # self.kspCurlCurl.setOperators(self.B)
        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()


        self.kspCurlCurl.solve(x1, y1)
        print self.kspCurlCurl.its
        self.kspLapl.solve(x2, y2)

        y.array = (np.concatenate([y1.array, y2.array]))








class Hiptmair(BaseMyPC):

    def __init__(self, W, kspScalar, kspCGScalar, kspVector, G, P, A, Hiptmairtol):
        self.W = W
        self.kspScalar = kspScalar
        self.kspCGScalar = kspCGScalar
        self.kspVector = kspVector
        self.P = P
        self.G = G
        self.A = A
        self.u_is = PETSc.IS().createGeneral(range(W.sub(0).dim()))
        self.p_is = PETSc.IS().createGeneral(range(W.sub(0).dim(),W.sub(0).dim()+W.sub(1).dim()))
        self.HiptmairIts = 0
        self.CGits = 0
        self.tol = Hiptmairtol

    # def create(self, pc):
    #     # print "Create"


    # def setUp(self, pc):

        # print "setup"


    def apply(self, pc, x, y):
        # self.kspCurlCurl.setOperators(self.B)
        x1 = x.getSubVector(self.u_is)
        y1 = x1.duplicate()
        x2 = x.getSubVector(self.p_is)
        y2 = x2.duplicate()

        # tic()
        y1, its, self.HiptmairTime = HiptmairSetup.HiptmairApply(self.A, x1, self.kspScalar, self.kspVector, self.G, self.P, self.tol)
        # print "Hiptmair time: ", toc()
        self.HiptmairIts += its
        tic()
        self.kspCGScalar.solve(x2, y2)
        self.CGtime = toc()
        # print "Laplacian, its ", self.kspCGScalar.its, "  time ",  self.CGtime

        # print "CG time: ", toc()
        # print "Laplacian inner iterations: ", self.kspCGScalar.its
        y.array = (np.concatenate([y1.array, y2.array]))
        self.CGits += self.kspCGScalar.its

    def ITS(self):
        return self.CGits, self.HiptmairIts , self.CGtime, self.HiptmairTime
