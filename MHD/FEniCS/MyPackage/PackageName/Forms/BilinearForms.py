# from dolfin import *

# def Stokes(W, f, nu, Stab = 'No'):

#     if str(W.__class__).find('list') == -1:
#         Split = 'No'
#         mesh = W.mesh()
#     else:
#         Split = 'Yes'
#         mesh = W[0].mesh()

#     dim = f.shape()[0]
#     if Split == "No":
#         (u, p) = TrialFunctions(W)
#         (v, q) = TestFunctions(W)
#     else:
#         u  = TrialFunction(W[0])
#         v = TestFunction(W[0])
#         p = TrialFunction(W[1])
#         q = TestFunction(W[1])

#     Laplacian = nu*inner(grad(u),grad(v))*dx
#     Grad = -div(u)*p*dx
#     GradT = -div(v)*q*dx

#     if Stab == 'No':
#         if Split == 'Yes':
#             S = None
#         else:
#             S = 0
#     elif Stab == 'P1P1':
#         h = CellSize(mesh)
#         beta = 0.2
#         delta = 0.2*h*h
#         S = delta*inner(grad(p),grad(q))*dx
#     elif Stab == 'P1Q0':
#         h = CellSize(mesh)
#         beta = 0.2
#         delta = 0.2*h*h
#         S = delta*(jump(p),jump(q))*dx

#     if Split == 'Yes':
#         A = [Laplacian, GradT, S]
#         b = inner(f,v)*dx
#     else:
#         A = Laplacian + Grad + GradT - S
#         b = inner(f,v)*dx

#     return A, b

# def NS(W, f, nu, u_k, Stab = 'No'):

#     if str(W.__class__).find('list') == -1:
#         Split = 'No'
#         mesh = W.mesh()
#     else:
#         Split = 'Yes'
#         mesh = W[0].mesh()

#     dim = f.shape()[0]
#     if Split == "No":
#         (u, p) = TrialFunctions(W)
#         (v, q) = TestFunctions(W)
#     else:
#         u  = TrialFunction(W[0])
#         v = TestFunction(W[0])
#         p = TrialFunction(W[1])
#         q = TestFunction(W[1])

#     n = FacetNormal(mesh)
#     Laplacian = nu*inner(grad(u),grad(v))*dx
#     Advection = inner((grad(u)*u_k),v)*dx(mesh) +(1/2)*div(u_k)*inner(u,v)*dx(mesh) - (1/2)*inner(u_k,n)*inner(u,v)*ds(mesh)
#     Grad = -div(u)*p*dx
#     GradT = -div(v)*q*dx

#     if Stab == 'No':
#         if Split == 'Yes':
#             S = None
#         else:
#             S = 0
#     elif Stab == 'P1P1':
#         h = CellSize(mesh)
#         beta = 0.2
#         delta = 0.2*h*h
#         S = delta*inner(grad(p),grad(q))*dx
#     elif Stab == 'P1Q0':
#         h = CellSize(mesh)
#         beta = 0.2
#         delta = 0.2*h*h
#         S = delta*(jump(p),jump(q))*dx

#     if Split == 'Yes':
#         A = [Laplacian, Advection, GradT, S]
#         b = inner(f,v)*dx
#     else:
#         A = Laplacian + Advection + Grad + GradT - S
#         b = inner(f,v)*dx

#     return A, b


# def Maxwell():
#     return 0

def MHD():
    return 0
