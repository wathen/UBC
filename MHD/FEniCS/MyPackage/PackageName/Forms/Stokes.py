from dolfin import *

def Stokes(W, f, nu, Stab = 'No'):

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

    Laplacian = nu*inner(grad(u),grad(v))*dx
    Grad = -div(u)*q*dx
    GradT = -div(v)*p*dx

    if Stab == 'No':
        if Split == 'Yes':
            S = None
        else:
            S = 0
    elif Stab == 'P1P1':
        h = CellSize(mesh)
        beta = 0.2
        delta = beta*h*h
        S = -delta*inner(grad(p),grad(q))*dx
    elif Stab == 'P1Q0':
        h = CellSize(mesh)
        beta = 0.2
        delta = beta*h*h
        S = -delta*(jump(p)*jump(q))*dS

    if Split == 'Yes':
        if Stab == 'Yes':
            A = {'Laplacian': Laplacian, 'Grad': Grad, 'Stab': S}
        else:
            A = {'Laplacian': Laplacian, 'Grad': Grad}
        b = {'Velocity': inner(v,f)*dx}
    else:
        A = Laplacian + Grad + GradT + S
        b = inner(v,f)*dx

    return A, b
