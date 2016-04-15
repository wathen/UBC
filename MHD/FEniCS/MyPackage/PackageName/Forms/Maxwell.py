from dolfin import *

def Maxwell(W, f, nu_m):

    if str(W.__class__).find('dict') == -1:
        Split = 'No'
    else:
        Split = 'Yes'

    dim = f.shape()[0]
    if Split == "No":
        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)
    else:
        u  = TrialFunction(W['Magnetic'])
        v = TestFunction(W['Magnetic'])
        p = TrialFunction(W['Multiplier'])
        q = TestFunction(W['Multiplier'])

    CurlCurl = nu_m*inner(curl(u), curl(v))*dx
    Div = inner(u,grad(q))*dx
    DivT = inner(v,grad(p))*dx

    if Split == 'Yes':
        S = Constant('0.0')*inner(p,q)*dx
    else:
        S = 0

    if Split == 'Yes':
        A = {'CurlCurl': CurlCurl, 'Div': Div, 'Stab': S}
        b = {'Magnetic': inner(f,v)*dx, 'Multiplier': Constant('0.0')*q*dx}
    else:
        A = CurlCurl + Div + DivT + S
        b = inner(f,v)*dx

    return A, b



