

from dolfin import *

def MHD2D(mesh, W,F_M,F_NS, u_k,b_k,params,IterType="Full",FS = "CG", SaddlePoint = "No", Stokes = "No", Split = "No"):
    if str(W.__class__).find('list') == -1:
        Split = 'No'
    else:
        Split = 'Yes'
    dim = F_M.shape()[0]
    if Split == "No":
        if SaddlePoint == "Yes":
            (u, b, p, r) = TrialFunctions(W)
            (v, c, q, s) = TestFunctions(W)
        elif SaddlePoint == "number2":
            (b, u, p, r) = TrialFunctions(W)
            (c, v, q, s) = TestFunctions(W)
        else:
            (u, p, b, r) = TrialFunctions(W)
            (v, q, c, s) = TestFunctions(W)
    else:
        if SaddlePoint == "Yes":
            u  = TrialFunction(W[0])
            b = TrialFunction(W[1])
            p = TrialFunction(W[2])
            r = TrialFunction(W[3])
            v = TestFunction(W[0])
            c = TestFunction(W[1])
            q = TestFunction(W[2])
            s = TestFunction(W[3])
        elif SaddlePoint == "number2":
            u  = TrialFunction(W[1])
            b = TrialFunction(W[0])
            p = TrialFunction(W[2])
            r = TrialFunction(W[3])
            v = TestFunction(W[1])
            c = TestFunction(W[0])
            q = TestFunction(W[2])
            s = TestFunction(W[3])
        else:
            u  = TrialFunction(W[0])
            b = TrialFunction(W[2])
            p = TrialFunction(W[1])
            r = TrialFunction(W[3])
            v = TestFunction(W[0])
            c = TestFunction(W[2])
            q = TestFunction(W[1])
            s = TestFunction(W[3])

    "'Maxwell Setup'"
    if params[0] == 0:
        m11 = params[1]*inner(curl(b),curl(c))*dx
    else:
        m11 = params[1]*params[0]*inner(curl(b),curl(c))*dx

    m22 = inner(r,s)*dx
    m21 = inner(c,grad(r))*dx
    m12 = inner(b,grad(s))*dx
    Lmaxwell  = inner(c, F_M)*dx

    "'NS Setup'"
    n = FacetNormal(mesh)


    if IterType == "CD":
        a11 = params[2]*inner(grad(v), grad(u))*dx
    else:
        if Stokes == "No":
            a11 = params[2]*inner(grad(v), grad(u))*dx(mesh)+ inner((grad(u)*u_k),v)*dx(mesh) +(1./2)*div(u_k)*inner(u,v)*dx(mesh) - (1./2)*inner(u_k,n)*inner(u,v)*ds(mesh)
        else:
            a11 = params[2]*inner(grad(v),grad(u))*dx

    a12 = -div(v)*p*dx
    a21 = -div(u)*q*dx
    Lns  = inner(v, F_NS)*dx


    "'Coupling term Setup'"
    if IterType == "Full":
        if dim == 2:
            CoupleT = params[0]*(v[0]*b_k[1]-v[1]*b_k[0])*curl(b)*dx
            Couple = -params[0]*(u[0]*b_k[1]-u[1]*b_k[0])*curl(c)*dx
        elif dim == 3:
            CoupleT = params[0]*inner(cross(v,b_k),curl(b))*dx
            Couple = -params[0]*inner(cross(u,b_k),curl(c))*dx
    else:
        Couple = 0
        CoupleT = 0



    if Split == "No":
        maxwell = m11+m12+m21
        if FS == "CG":
            if W.sub(0).__str__().find("Bubble") == -1 and W.sub(0).__str__().find("CG1") != -1:
                h = CellSize(mesh)
                beta  = 0.2
                delta = beta*h*h
                a22 = delta*inner(grad(p),grad(q))*dx
                ns = a11+a12+a21-a22
                Lns  = inner(v + delta*grad(q), F_NS)*dx
            else:
                ns = a11+a12+a21

        else:
            h = CellSize(mesh)
            h_avg =avg(h)
            a22 = 0.1*h_avg*jump(p)*jump(q)*dS
            ns = a11+a12+a21 - a22
        CoupleTerm = Couple+CoupleT
    else:
        maxwell = [m11,m12,m21,m22]
        if FS == "CG":
            if W[0].__str__().find("Bubble") == -1 and W[0].__str__().find("CG1") != -1:
                h = CellSize(mesh)
                beta  = 0.2
                delta = beta*h*h
                a22 = delta*inner(grad(p),grad(q))*dx
                ns = [a11,a12,a21 ,-a22]
            else:
                ns = [a11,a12,a21,None]

        else:
            h = CellSize(mesh)
            h_avg =avg(h)
            a22 = 0.1*h_avg*jump(p)*jump(q)*dS
            ns = [a11,a12,a21 ,-a22]
        CoupleTerm = [Couple,CoupleT]
    return ns,maxwell,CoupleTerm,Lmaxwell,Lns





def PicardRHS(mesh, W, u_k,p_k,b_k,r_k,params, FS = "CG",SaddlePoint = "No", Stokes = "No"):
     # (u, p, b, r) = TrialFunctions(W)

    if str(W.__class__).find('list') == -1:
        Split = 'No'
    else:
        Split = 'Yes'
    if Split == "No":
        if SaddlePoint == "Yes":
            (u, b, p, r) = TrialFunctions(W)
            (v, c, q, s) = TestFunctions(W)
        elif SaddlePoint == "number2":
            (b, u, p, r) = TrialFunctions(W)
            (c, v, q, s) = TestFunctions(W)
        else:
            (u, p, b, r) = TrialFunctions(W)
            (v, q, c, s) = TestFunctions(W)
    else:
        if SaddlePoint == "Yes":
            v = TestFunction(W[0])
            c = TestFunction(W[1])
            q = TestFunction(W[2])
            s = TestFunction(W[3])
        elif SaddlePoint == "number2":
            v = TestFunction(W[1])
            c = TestFunction(W[0])
            q = TestFunction(W[2])
            s = TestFunction(W[3])
        else:
            v = TestFunction(W[0])
            c = TestFunction(W[2])
            q = TestFunction(W[1])
            s = TestFunction(W[3])


    dim = mesh.geometry().dim()
    "'Maxwell Setup'"
    if params[0] == 0:
        m11 = params[1]*inner(curl(b_k),curl(c))*dx
    else:
        m11 = params[1]*params[0]*inner(curl(b_k),curl(c))*dx

    m21 = inner(c,grad(r_k))*dx
    m12 = inner(b_k,grad(s))*dx
    maxwell = m11+m12+m21

    "'NS Setup'"
    n = FacetNormal(mesh)
    if Stokes == "No":
        a11 = params[2]*inner(grad(v), grad(u_k))*dx(mesh) + inner((grad(u_k)*u_k),v)*dx(mesh) +(1./2)*div(u_k)*inner(u_k,v)*dx(mesh) - (1./2)*inner(u_k,n)*inner(u_k,v)*ds(mesh)
    else:
        a11 = params[2]*inner(grad(v), grad(u_k))*dx
    a12 = -div(v)*p_k*dx
    a21 = -div(u_k)*q*dx
    a22 = 0
    if Split == "No":
        if FS == "CG":
            if W.sub(0).__str__().find("Bubble") == -1 and W.sub(0).__str__().find("CG1") != -1:
                h = CellSize(mesh)
                beta  = 0.2
                delta = beta*h*h
                a22 = delta*inner(grad(p_k),grad(q))*dx
                ns = a11+a12+a21-a22
            else:
                ns = a11+a12+a21
        else:
            h = CellSize(mesh)
            h_avg =avg(h)
            a22 = 0.1*h_avg*jump(p_k)*jump(q)*dS
            ns = a11+a12+a21 - a22
    else:
        if FS == "CG":
            if W[0].__str__().find("Bubble") == -1 and W.sub(0).__str__().find("CG1") != -1:
                h = CellSize(mesh)
                beta  = 0.2
                delta = beta*h*h
                a22 = delta*inner(grad(p_k),grad(q))*dx
                ns = a11+a12+a21-a22
            else:
                ns = a11+a12+a21
        else:
            h = CellSize(mesh)
            h_avg =avg(h)
            a22 = 0.1*h_avg*jump(p_k)*jump(q)*dS
            ns = a11+a12+a21 - a22
    "'Coupling term Setup'"
    if dim == 2:
        CoupleT = params[0]*(v[0]*b_k[1]-v[1]*b_k[0])*curl(b_k)*dx
        Couple = -params[0]*(u_k[0]*b_k[1]-u_k[1]*b_k[0])*curl(c)*dx
    elif dim == 3:
        CoupleT = params[0]*inner(cross(v,b_k),curl(b_k))*dx
        Couple = -params[0]*inner(cross(u_k,b_k),curl(c))*dx



    if Split == "No":
        CoupleTerm = Couple+CoupleT
        RHSform = ns+maxwell+CoupleTerm
        return RHSform
    else:
        RHSform = [a11+CoupleT+a12,a21-a22,Couple+m11+m21,m12]
    return RHSform






def MHDmatvec(mesh, W, F_M, F_NS,u_k,b_k,u,b,p,r, params,IterType="Full",FS = "CG", SaddlePoint = "No", Stokes = "No", Split = "No"):
    if str(W.__class__).find('list') == -1:
        Split = 'No'
    else:
        Split = 'Yes'
    dim = F_M.shape()[0]
    if Split == "No":
        if SaddlePoint == "Yes":
            (v, c, q, s) = TestFunctions(W)
        elif SaddlePoint == "number2":
            (c, v, q, s) = TestFunctions(W)
        else:
            (v, q, c, s) = TestFunctions(W)
    else:
        if SaddlePoint == "Yes":
            v = TestFunction(W[0])
            c = TestFunction(W[1])
            q = TestFunction(W[2])
            s = TestFunction(W[3])
        elif SaddlePoint == "number2":
            v = TestFunction(W[1])
            c = TestFunction(W[0])
            q = TestFunction(W[2])
            s = TestFunction(W[3])
        else:
            v = TestFunction(W[0])
            c = TestFunction(W[2])
            q = TestFunction(W[1])
            s = TestFunction(W[3])

    "'Maxwell Setup'"
    if params[0] == 0:
        m11 = params[1]*inner(curl(b),curl(c))*dx
    else:
        m11 = params[1]*params[0]*inner(curl(b),curl(c))*dx

    m22 = inner(r,s)*dx
    m21 = inner(c,grad(r))*dx
    m12 = inner(b,grad(s))*dx
    Lmaxwell  = inner(c, F_M)*dx

    "'NS Setup'"
    n = FacetNormal(mesh)


    if IterType == "CD":
        a11 = params[2]*inner(grad(v), grad(u))*dx
    else:
        if Stokes == "No":
            a11 = params[2]*inner(grad(v), grad(u))*dx(mesh)+ inner((grad(u)*u_k),v)*dx(mesh) +(1./2)*div(u_k)*inner(u,v)*dx(mesh) - (1./2)*inner(u_k,n)*inner(u,v)*ds(mesh)
        else:
            a11 = params[2]*inner(grad(v),grad(u))*dx

    a12 = -div(v)*p*dx
    a21 = -div(u)*q*dx
    Lns  = inner(v, F_NS)*dx


    "'Coupling term Setup'"
    if IterType == "Full":
        if dim == 2:
            CoupleT = params[0]*(v[0]*b_k[1]-v[1]*b_k[0])*curl(b)*dx
            Couple = -params[0]*(u[0]*b_k[1]-u[1]*b_k[0])*curl(c)*dx
        elif dim == 3:
            CoupleT = params[0]*inner(cross(v,b_k),curl(b))*dx
            Couple = -params[0]*inner(cross(u,b_k),curl(c))*dx
    else:
        Couple = 0
        CoupleT = 0



    if Split == "No":
        maxwell = m11+m12+m21
        if FS == "CG":
            if W.sub(0).__str__().find("Bubble") == -1 and W.sub(0).__str__().find("CG1") != -1:
                h = CellSize(mesh)
                beta  = 0.2
                delta = beta*h*h
                a22 = delta*inner(grad(p),grad(q))*dx
                ns = a11+a12+a21-a22
                Lns  = inner(v + delta*grad(q), F_NS)*dx
            else:
                ns = a11+a12+a21

        else:
            h = CellSize(mesh)
            h_avg =avg(h)
            a22 = 0.1*h_avg*jump(p)*jump(q)*dS
            ns = a11+a12+a21 - a22
        CoupleTerm = Couple+CoupleT
    else:
        maxwell = [m11,m12,m21,m22]
        if FS == "CG":
            if W[0].__str__().find("Bubble") == -1 and W[0].__str__().find("CG1") != -1:
                h = CellSize(mesh)
                beta  = 0.2
                delta = beta*h*h
                a22 = delta*inner(grad(p),grad(q))*dx
                ns = [a11,a12,a21 ,-a22]
            else:
                ns = [a11,a12,a21,None]

        else:
            h = CellSize(mesh)
            h_avg =avg(h)
            a22 = 0.1*h_avg*jump(p)*jump(q)*dS
            ns = [a11,a12,a21 ,-a22]
        CoupleTerm = [Couple,CoupleT]
    return ns+maxwell+CoupleTerm, Lmaxwell, Lns, a12, CoupleT


