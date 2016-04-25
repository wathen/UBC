import HartmanChannel
kappa = 1.0
Mu_m = float(1e4)
MU = 1.0
params = [kappa,Mu_m,MU]
mesh, boundaries, domains = HartmanChannel.Domain(2)
u0, b0, p0, r0, F_S, F_M = HartmanChannel.ExactSol(mesh, params)
