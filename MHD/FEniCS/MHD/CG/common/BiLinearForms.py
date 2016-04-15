from dolfin import *


class Forms(object):
    """docstring for Forms"""
    def __init__(self, mesh, W,F_M,F_NS, u_k,b_k,params,options={}):
        assert type(options) is dict, 'options must be a dictionary object'

        self.mesh = mesh
        self.W = W
        self.F_M= F_M
        self.F_NS= F_NS
        self.u_k= u_k
        self.b_k= b_k
        self.params= params
        self.options= options


        def printW(self, W):
            print W

def MHD2D(mesh, W,F_M,F_NS, u_k,b_k,params,split):

    (u, p, b, r) = TrialFunctions(W)
    (v, q, c,s ) = TestFunctions(W)

    if (split == "Linear"):
        "'Maxwell Setup'"
        a11 = params[1]*params[0]*inner(curl(b),curl(c))*dx
        a12 = inner(c,grad(r))*dx
        a21 = inner(b,grad(s))*dx
        Lmaxwell  = inner(c, F_M)*dx
        maxwell = a11+a12+a21


        "'NS Setup'"
        n = FacetNormal(mesh)
        a11 = params[2]*inner(grad(v), grad(u))*dx +inner((grad(u)*u_k),v)*dx+(1/2)*div(u_k)*inner(u,v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds
        a12 = -div(v)*p*dx
        a21 = -div(u)*q*dx
        Lns  = inner(v, F_NS)*dx
        ns = a11+a12+a21


        "'Coupling term Setup'"
        CoupleTerm = params[0]*inner(v[0]*b_k[1]-v[1]*b_k[0],curl(b))*dx - params[0]*inner(u[0]*b_k[1]-u[1]*b_k[0],curl(c))*dx


        return ns,maxwell,CoupleTerm,Lmaxwell,Lns

    elif (split == NoneLinear):

        "' Linear Setup'"
        m11 = params[1]*params[0]*inner(curl(b),curl(c))*dx
        m12 = inner(c,grad(r))*dx
        m21 = inner(b,grad(s))*dx
        Lmaxwell  = inner(c, F_M)*dx
        maxwell = m11+m12+m21

        ns11 = params[2]*inner(grad(v), grad(u))*dx
        ns12 = -div(v)*p*dx
        ns21 = -div(u)*q*dx
        Lns  = inner(v, F_NS)*dx
        ns = ns11+ns12+ns21

        linear = ns+maxwell
        RHS = Lns+Lmaxwell

        "' None-Linear Setup'"
        n = FacetNormal(mesh)
        Nlinear = params[0]*inner(v[0]*b_k[1]-v[1]*b_k[0],curl(b))*dx - params[0]*inner(u[0]*b_k[1]-u[1]*b_k[0],curl(c))*dx +inner((grad(u)*u_k),v)*dx+(1/2)*div(u_k)*inner(u,v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds

        return linear, Nlinear, RHS




def MHD3D(mesh, W,F_M,F_NS, u_k,b_k,params):

    (u, p, b, r) = TrialFunctions(W)
    (v, q, c,s ) = TestFunctions(W)
    "'Maxwell Setup'"
    a11 = params[1]*params[0]*inner(curl(b),curl(c))*dx
    a12 = inner(c,grad(r))*dx
    a21 = inner(b,grad(s))*dx
    Lmaxwell  = inner(c, F_M)*dx
    maxwell = a11+a12+a21


    "'NS Setup'"
    n = FacetNormal(mesh)
    a11 = params[2]*inner(grad(v), grad(u))*dx +inner((grad(u)*u_k),v)*dx+(1/2)*div(u_k)*inner(u,v)*dx- (1/2)*inner(u_k,n)*inner(u,v)*ds
    a12 = -div(v)*p*dx
    a21 = -div(u)*q*dx
    Lns  = inner(v, F_NS)*dx
    ns = a11+a12+a21


    "'Coupling term Setup'"
    CoupleTerm = params[0]*inner(cross(v,b_k),curl(b))*dx - params[0]*inner(cross(u,b_k), b,curl(c))*dx


    return ns,maxwell,CoupleTerm,Lmaxwell,Lns
