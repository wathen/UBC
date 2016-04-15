from sympy import *

x = symbols('x[0]')
y = symbols('x[1]')

u = sin(y)*exp(x+y)+cos(y)*exp(x+y)
v = -sin(y)*exp(x+y)
p = pow(x,3)*sin(y)+exp(x+y)

L1 = diff(u,x,x)+diff(u,y,y)
L2 = diff(v,x,x)+diff(v,y,y)
print "Laplacian 1:    ",L1
print "\n"
print "Laplacian 2:    ",L2
print "\n"
A1 = u*diff(u,x)+v*diff(u,y)
A2 = u*diff(v,x)+v*diff(v,y)

print "Advection 1:    ",A1
print "\n"
print "Advection 2:    ",A2

print "\n"
P1 = diff(p,x)
P2 = diff(p,y)
print "Pressure 1:    ",P1
print "\n"
print "Pressure 2:    ",P2




 # (exp(x + y)*sin(y) + exp(x + y)*cos(y))**2 - 2*exp(2*x + 2*y)*sin(y)*cos(y)


