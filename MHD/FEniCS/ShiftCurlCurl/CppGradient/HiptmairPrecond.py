import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

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

    def __init__(self, G, Gt, P, Pt, VectorLaplacian, ScalarLaplacian):
        self.G = G
        self.Gt = Gt
        self.P = P
        self.Pt = Pt
        self.VectorLaplacian = VectorLaplacian
        self.ScalarLaplacian = ScalarLaplacian

    def create(self, pc):
        OptDB = PETSc.Options()
        # OptDB['pc_factor_shift_amount'] = 1
        OptDB['pc_hypre_type'] = 'boomeramg'
        OptDB['pc_hypre_boomeramg_strong_threshold']  = 0.5

        # OptDB['pc_factor_mat_ordering_type'] = 'amd'
        # OptDB['pc_factor_mat_solver_package']  = 'mumps'

        self.diag = None
        kspVector = PETSc.KSP()
        kspVector.create(comm=PETSc.COMM_WORLD)
        pcVector = kspVector.getPC()
        kspVector.setType('preonly')
        pcVector.setType('hypre')
        kspVector.setFromOptions()
        # kspVector.max_it = 10
        OptDB = PETSc.Options()
        # OptDB['pc_factor_shift_amount'] = 1

        kspVector.setFromOptions()
        self.kspVector = kspVector

        kspScalar = PETSc.KSP()
        kspScalar.create(comm=PETSc.COMM_WORLD)
        pcScalar = kspScalar.getPC()
        kspScalar.setType('preonly')
        pcScalar.setType('hypre')
        kspScalar.setFromOptions()
        # kspScalar.max_it = 10
        kspScalar.setFromOptions()
        self.kspScalar = kspScalar
        # print kspM.view()


    def setUp(self, pc):
        A, P, flag = pc.getOperators()
        self.kspVector.setOperators(self.VectorLaplacian,self.VectorLaplacian)
        self.kspScalar.setOperators(self.ScalarLaplacian,self.ScalarLaplacian)
        self.diag = A.getDiagonal()
        self.diag.reciprocal()

    def apply(self, pc, x, y):
        x1 = (self.Pt*x).duplicate()
        x2 = (self.Gt*x).duplicate()
        x3 = x.duplicate()
        self.kspVector.solve(self.Pt*x, x1)
        self.kspScalar.solve(self.Gt*x, x2)
        x3.pointwiseMult(self.diag, x)

        y.array = (x3.array+(self.P*x1).array+(self.G*x2).array)














class GS(BaseMyPC):

    def __init__(self, G, Gt, P, Pt, VectorLaplacian, ScalarLaplacian, Lower, Upper):
        self.G = G
        self.Gt = Gt
        self.P = P
        self.Pt = Pt
        self.VectorLaplacian = VectorLaplacian
        self.ScalarLaplacian = ScalarLaplacian
        self.Upper = Upper
        self.Lower = Lower

    def create(self, pc):
        OptDB = PETSc.Options()
        # OptDB['pc_factor_shift_amount'] = 1
        # OptDB['pc_hypre_type'] = 'boomeramg'
        # OptDB['pc_hypre_boomeramg_strong_threshold']  = 0.5
        # OptDB['pc_hypre_boomeramg_grid_sweeps_all']  = 3
        OptDB['pc_factor_mat_ordering_type'] = 'amd'
        OptDB['pc_factor_mat_solver_package'] = 'mumps'

        kspVector = PETSc.KSP()
        kspVector.create(comm=PETSc.COMM_WORLD)
        pcVector = kspVector.getPC()
        kspVector.setType('preonly')
        pcVector.setType('lu')
        kspVector.setFromOptions()
        # kspVector.max_it = 1
        OptDB = PETSc.Options()
        kspVector.setFromOptions()
        self.kspVector = kspVector

        kspScalar = PETSc.KSP()
        kspScalar.create(comm=PETSc.COMM_WORLD)
        pcScalar = kspScalar.getPC()
        kspScalar.setType('preonly')
        pcScalar.setType('lu')
        kspScalar.setFromOptions()
        # kspScalar.max_it = 1
        kspScalar.setFromOptions()
        self.kspScalar = kspScalar





        kspLower = PETSc.KSP()
        kspLower.create(comm=PETSc.COMM_WORLD)
        pcScalar = kspLower.getPC()
        kspLower.setType('preonly')
        pcScalar.setType('lu')
        self.kspLower = kspLower


        kspUpper = PETSc.KSP()
        kspUpper.create(comm=PETSc.COMM_WORLD)
        pcScalar = kspUpper.getPC()
        kspUpper.setType('preonly')
        pcScalar.setType('lu')
        self.kspUpper = kspUpper



    def setUp(self, pc):
        A, P, flag = pc.getOperators()
        self.kspVector.setOperators(self.VectorLaplacian,self.VectorLaplacian)
        self.kspScalar.setOperators(self.ScalarLaplacian,self.ScalarLaplacian)
        self.kspLower.setOperators(self.Lower,self.Lower)
        self.kspUpper.setOperators(self.Upper,self.Upper)
        self.A = P
        self.diag = A.getDiagonal()
        self.diag.scale(3.0/2)

        self.diag.reciprocal()

    def apply(self, pc, x, y):
        xhat = self.Pt.getVecLeft()
        self.Pt.mult(x,xhat)
        yp =self.Pt.getVecRight()
        yhat =self.Pt.getVecLeft()
        self.kspVector.solve(xhat, yhat)
        self.P.mult(yhat,yp)


        xhat = self.Gt.getVecLeft()
        self.Gt.mult(x,xhat)
        yg =self.Gt.getVecRight()
        yhat =self.Gt.getVecLeft()
        self.kspScalar.solve(xhat, yhat)
        self.G.mult(yhat,yg)

        xsave = x.duplicate()
        for i in range(1,2):
            # xx1 = x.duplicate()
            # xx2 = x.duplicate()
            # xx3 = x.duplicate()
            xx = x.duplicate()
            # # xx5 = x.duplicate()

            # self.kspLower.solve(x, xx1)
            # self.kspUpper.solve(x, xx2)

            # self.kspLower.solve(self.A*xx2, xx3)

            xx.pointwiseMult(self.diag, x)
            r = x-self.A*(xx)
            x = r
            xsave += xx

        y.array = (xx.array+yg.array+yp.array)
        # print y.array

















class GSvector(BaseMyPC):

    def __init__(self, G, P, VectorLaplacian, ScalarLaplacian, Lower, Upper):
        self.G = G
        self.P = P
        self.VectorLaplacian = VectorLaplacian
        self.ScalarLaplacian = ScalarLaplacian
        self.Upper = Upper
        self.Lower = Lower

    def create(self, pc):
        OptDB = PETSc.Options()
        OptDB['pc_factor_shift_amount'] = 1
        OptDB['pc_hypre_type'] = 'boomeramg'
        OptDB['pc_hypre_boomeramg_strong_threshold']  = 0.5
        OptDB['pc_hypre_boomeramg_grid_sweeps_all']  = 1
        # OptDB['pc_factor_mat_ordering_type'] = 'amd'
        # OptDB['pc_factor_mat_solver_package'] = 'mumps'


        kspVector = PETSc.KSP()
        kspVector.create(comm=PETSc.COMM_WORLD)
        pcVector = kspVector.getPC()
        kspVector.setType('preonly')
        pcVector.setType('hypre')
        kspVector.max_it = 1
        kspVector.setFromOptions()
        self.kspVector = kspVector

        kspScalar = PETSc.KSP()
        kspScalar.create(comm=PETSc.COMM_WORLD)
        pcScalar = kspScalar.getPC()
        kspScalar.setType('preonly')
        pcScalar.setType('hypre')
        kspScalar.setFromOptions()
        kspScalar.max_it = 1
        self.kspScalar = kspScalar



        kspLower = PETSc.KSP()
        kspLower.create(comm=PETSc.COMM_WORLD)
        pcScalar = kspLower.getPC()
        kspLower.setType('preonly')
        pcScalar.setType('lu')
        self.kspLower = kspLower

        kspUpper = PETSc.KSP()
        kspUpper.create(comm=PETSc.COMM_WORLD)
        pcScalar = kspUpper.getPC()
        kspUpper.setType('preonly')
        pcScalar.setType('lu')
        self.kspUpper = kspUpper

    def setUp(self, pc):
        A, P, flag = pc.getOperators()
        self.kspVector.setOperators(self.VectorLaplacian,self.VectorLaplacian)
        self.kspScalar.setOperators(self.ScalarLaplacian,self.ScalarLaplacian)
        self.kspLower.setOperators(self.Lower,self.Lower)
        self.kspUpper.setOperators(self.Upper,self.Upper)
        self.A = A
        self.diag = A.getDiagonal()
        # self.diag.scale(3.0/2)

        self.diag.reciprocal()

    def apply(self, pc, x, y):
        ysave = x.duplicate()


        xhat = self.P[0].getVecRight()
        self.P[0].multTranspose(x,xhat)
        yp1 =self.P[0].getVecLeft()
        yhat =self.P[0].getVecRight()
        self.kspVector.solve(xhat, yhat)
        self.P[0].mult(yhat,yp1)
        xhat.destroy()
        yhat.destroy()
        # y.axpy(1.0,yp1)


        xhat = self.P[1].getVecRight()
        self.P[1].multTranspose(x,xhat)
        yp2 =self.P[1].getVecLeft()
        yhat =self.P[1].getVecRight()
        self.kspVector.solve(xhat, yhat)
        self.P[1].mult(yhat,yp2)
        xhat.destroy()
        yhat.destroy()
        # y.axpy(1.0,yp2)

        if len(self.P) == 3:
            xhat = self.P[2].getVecRight()
            self.P[2].multTranspose(x,xhat)
            yp3 =self.P[2].getVecLeft()
            yhat =self.P[2].getVecRight()
            self.kspVector.solve(xhat, yhat)
            self.P[2].mult(yhat,yp3)
            # y.axpy(1.0,yp3)


        xhat = self.G.getVecRight()
        self.G.multTranspose(x,xhat)
        yg =self.G.getVecLeft()
        yhat =self.G.getVecRight()
        self.kspScalar.solve(xhat, yhat)
        self.G.mult(yhat,yg)
        xhat.destroy()
        yhat.destroy()
        # y.aypx(1.0,yg)
        xsave = x.duplicate()
        for i in range(1,2):
            # xx1 = x.duplicate()
            # xx2 = x.duplicate()
            # xx3 = x.duplicate()
            xx = x.duplicate()
            # # xx5 = x.duplicate()

            # self.kspLower.solve(x, xx1)
            # self.kspUpper.solve(x, xx2)

            # self.kspLower.solve(self.A*xx2, xx3)

            xx.pointwiseMult(self.diag, x)
            x = x-self.A*(xx)
            # x = r
            xsave += xx
        # y.aypx(1.0,x3)

        if len(self.P) == 2:
            ysave.array = (xsave.array+yp1.array+yp2.array+yg.array)
        else:
            ysave.array = (xsave.array+yp1.array+yp2.array+yp3.array+yg.array)
        # r = x.duplicate()
        # b = x.duplicate()
        # self.A.mult(ysave,r)
        # b.array = x.array
        # x.array = b.array - r.array
        y.array = ysave.array



            # x.aypx(-1,r)
            # # print r.array
            # x.array = r.array


        # y.array = x3.array
        # print y.array
        # Vector = (self.P[0]*xPx).array +(self.P[1]*xPy).array
        # if len(self.P) == 2:
        #     y.array = ysave
        # else:
        #      y.array = ysave
