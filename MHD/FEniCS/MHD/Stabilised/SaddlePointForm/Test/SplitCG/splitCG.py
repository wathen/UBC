class MyKSP(object):

    def create(self, ksp):
        self.work = []

    def destroy(self, ksp):
        for v in self.work:
            v.destroy()

    def setUp(self, ksp):
        self.work[:] = ksp.getWorkVecs(right=2, left=None)

    def reset(self, ksp):
        for v in self.work:
            v.destroy()
        del self.work[:]

    def loop(self, ksp, r):
        its = ksp.getIterationNumber()
        rnorm = r.norm()
        ksp.setResidualNorm(rnorm)
        ksp.logConvergenceHistory(rnorm)
        ksp.monitor(its, rnorm)
        reason = ksp.callConvergenceTest(its, rnorm)
        if not reason:
            ksp.setIterationNumber(its+1)
        else:
            ksp.setConvergedReason(reason)
        return reason

class MyCG(MyKSP):


    def __init(self, C, Ct, F):
        self.C = C
        self.Ct = Ct
        self.F = F

    def setUp(self, ksp):
        super(MyCG, self).setUp(ksp)
        d = self.work[0].duplicate()
        q = d.duplicate()
        self.work += [d, q]

    def solve(self, ksp, b, x):
        A, B, flag = ksp.getOperators()
        P = ksp.getPC()
        r, z, d, q = self.work
        #
        A.mult(x, r)
        r.aypx(-1, b)
        r.copy(d)
        delta_0 = r.dot(r)
        delta = delta_0
        while not self.loop(ksp, r):
            A.mult(d, q)
            alpha = delta / d.dot(q)
            x.axpy(+alpha, d)
            r.axpy(-alpha, q)
            delta_old = delta
            delta = r.dot(r)
            beta = delta / delta_old
            d.aypx(beta, r)

class splitCG(MyKSP):


#    def __init(self, A, B):
#        self.A = A
#        self.B = B

    def Params(self,A,B):
        self.A = A
        self.B = B


    def setUp(self, ksp):
        super(splitCG, self).setUp(ksp)
        d = self.work[0].duplicate()
        q = d.duplicate()
        self.work += [d, q]

    def solve(self, ksp, b, x):
        #A, B, flag = ksp.getOerators()
        P = ksp.getPC()
        r, z, d, q = self.work
        
        r1 = r.duplicate() 
        r2 = r.duplicate() 
        self.A.mult(x, r1)
        self.B.mult(x, r2)
        r.aypx(1,r1)
        r.aypx(1,r2)
        
        r.aypx(-1, b)
        r.copy(d)
        delta_0 = r.dot(r)
        delta = delta_0
        while not self.loop(ksp, r):
            
            q1 = q.duplicate()
            q2 = q.duplicate()
            A.mult(d, q1)
            B.mult(d, q2)
            q.aypx(1, q1)
            q.aypx(1, q2)

            alpha = delta / d.dot(q)
            x.axpy(+alpha, d)
            r.axpy(-alpha, q)
            delta_old = delta
            delta = r.dot(r)
            beta = delta / delta_old
            d.aypx(beta, r)




class SplitMulti:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def mult(self, A, x, y):

        x1 = x.duplicate()
        x2 = x.duplicate()
        self.A.mult(x, x1)
        self.B.mult(x, x2)
        y.array = (x1.array+x2.array)
        #y.aypx(1, x1)
        #y.aypx(1, x2)
