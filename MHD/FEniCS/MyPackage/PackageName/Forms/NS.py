from dolfin import *

def NS(W, f, nu, u_k, p_k = None, Stab = 'No'):

    if str(W.__class__).find('dict') == -1:
        Split = 'No'
        mesh = W.mesh()
    else:
        Split = 'Yes'
        mesh = W['Velocity'].mesh()

    if Split == "No":
        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)
    else:
        u  = TrialFunction(W['Velocity'])
        v = TestFunction(W['Velocity'])
        p = TrialFunction(W['Pressure'])
        q = TestFunction(W['Pressure'])

    n = FacetNormal(mesh)
    Laplacian = nu*inner(grad(u),grad(v))*dx
    Advection = inner((grad(u)*u_k),v)*dx(mesh) +(1./2)*div(u_k)*inner(u,v)*dx(mesh) - (1./2)*inner(u_k,n)*inner(u,v)*ds(mesh)

    Grad = -div(u)*q*dx
    GradT = -div(v)*p*dx

    if Stab == 'No':
        if Split == 'Yes':
            S = None
            Srhs = None

        else:
            S = 0
            Srhs = 0

    elif Stab == 'P1P1':
        h = CellSize(mesh)
        beta = 0.2
        delta = beta*h*h
        S = -delta*inner(grad(p),grad(q))*dx
        Srhs = -delta*inner(grad(p_k),grad(q))*dx

    elif Stab == 'P1Q0':
        h = CellSize(mesh)
        beta = 0.2
        delta = beta*h*h
        S = -delta*(jump(p)*jump(q))*dS
        Srhs = -delta*(jump(p_k)*jump(q))*dS

    if Split == 'Yes':
        if Stab == 'Yes':
            A = {'ConvecDiff': Laplacian + Advection, 'Grad': Grad, 'Stab': S}

        else:
            A = {'Laplacian': Laplacian, 'Grad': Grad}

        if p_k == None:
            b = {'Velocity': inner(v,f)*dx}

        else:
            b = {'Velocity': inner(v,f)*dx - nu*inner(grad(u_k),grad(v))*dx - inner((grad(u_k)*u_k),v)*dx(mesh) +(1./2)*div(u_k)*inner(u_k,v)*dx(mesh) - (1./2)*inner(u_k,n)*inner(u_k,v)*ds(mesh) + div(v)*p_k*dx, 'Pressure': div(u_k)*q*dx - Srhs}

    else:
        A = Laplacian + Advection + Grad + GradT + S

        if p_k == None:
            b = inner(v,f)*dx

        else:
            Arhs =  nu*inner(grad(u_k),grad(v))*dx + inner((grad(u_k)*u_k),v)*dx(mesh) +(1./2)*div(u_k)*inner(u_k,v)*dx(mesh) - (1./2)*inner(u_k,n)*inner(u_k,v)*ds(mesh)
            Gradrhs = -div(u_k)*q*dx
            GradTrhs = -div(v)*p_k*dx
            b = inner(v,f)*dx - Arhs - Gradrhs - GradTrhs - Srhs

    return A, b
