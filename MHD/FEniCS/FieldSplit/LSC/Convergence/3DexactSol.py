from sympy import *

x = symbols('x[0]')
y = symbols('x[1]')
z = symbols('x[2]')
uu = sin(x)*exp(x+y+z)
uv = sin(y)*exp(x+y+z)
uw = sin(z)*exp(x+y+z)
uu = (x**4+y**5+z**6)
uv = (x**4+y**5+z**6)
uw =(x**4+y**5+z**6)
u = diff(uw,y)-diff(uv,z)
v = diff(uu,z)-diff(uw,x)
w = diff(uv,x)-diff(uu,y)
p = sin(y)+exp(x+y+z)
# u = y**1*z**2
# v = x**1*z**2
w = diff(u,x)
# p = x*y
print "u:    ",u
print "\n"
print "v:    ",v
print "\n"
print "w:    ",w
print "\n"
print "p:    ",p
print "\n"
L1 = diff(u,x,x)+diff(u,y,y) + diff(u,z,z)
L2 = diff(v,x,x)+diff(v,y,y) + diff(v,z,z)
L3 = diff(w,x,x)+diff(w,y,y) + diff(w,z,z)
print "Laplacian 1:    ",L1
print "\n"
print "Laplacian 2:    ",L2
print "\n"
print "Laplacian 3:    ",L3
print "\n"
A1 = u*diff(u,x)+v*diff(u,y)+w*diff(u,z)
A2 = u*diff(v,x)+v*diff(v,y)+w*diff(v,z)
A3 = u*diff(w,x)+v*diff(w,y)+w*diff(w,z)

print "Advection 1:    ",A1
print "\n"
print "Advection 2:    ",A2
print "\n"
print "Advection 3:    ",A3
print "\n"
P1 = diff(p,x)
P2 = diff(p,y)
P3 = diff(p,z)
print "Pressure 1:    ",P1
print "\n"
print "Pressure 2:    ",P2
print "\n"
print "Pressure 3:    ",P3

from dolfin import Expression
u0 = Expression((ccode(u),ccode(v),ccode(w)))

Laplacian = Expression((ccode(L1),ccode(L2),ccode(L3)))
Advection = Expression((ccode(A1),ccode(A2),ccode(A3)))
gradPres = Expression((ccode(P1),ccode(P2),ccode(P3)))



 # (exp(x + y)*sin(y) + exp(x + y)*cos(y))**2 - 2*exp(2*x + 2*y)*sin(y)*cos(y)


