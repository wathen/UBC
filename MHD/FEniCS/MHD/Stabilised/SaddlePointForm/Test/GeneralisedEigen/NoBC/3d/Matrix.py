from sympy import *

u = 3
b = 4
p = 1
r = 2
F = MatrixSymbol('F',u,u)
M = MatrixSymbol('M',b,b)
L = MatrixSymbol('L',r,r)
D = MatrixSymbol('D',r,b)
C = MatrixSymbol('C',b,u)
A = BlockMatrix([[F, C.T], [-C, M+D.T*Inverse(L)*D]])
P = BlockMatrix([[F+C.T*Inverse(M+D.T*Inverse(L)*D)*C,C.T],[0*C,M+D.T*Inverse(L)*D]])


print block_collapse(A*Inverse(P))
print block_collapse(Inverse(P)*A)
