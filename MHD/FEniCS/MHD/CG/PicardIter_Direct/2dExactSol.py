from sympy import *

x = symbols('x[0]')
y = symbols('x[1]')

u = sin(y)*exp(x+y)+cos(y)*exp(x+y)
v = -sin(y)*exp(x+y)
p = pow(x,3)*sin(y)+exp(x+y)
print "u:    ",u
print "v:    ",v
print "p:    ",p
print "\n"

L1 = diff(u,x,x)+diff(u,y,y)
L2 = diff(v,x,x)+diff(v,y,y)
print "Laplacian 1:    ",L1
print "Laplacian 2:    ",L2
print "\n"
A1 = u*diff(u,x)+v*diff(u,y)
A2 = u*diff(v,x)+v*diff(v,y)

print "Advection 1:    ",A1
print "Advection 2:    ",A2

print "\n"
P1 = diff(p,x)
P2 = diff(p,y)
print "Pressure 1:    ",P1
print "Pressure 2:    ",P2
print "\n"

x = symbols('x[0]')
y = symbols('x[1]')
uu = cos(x)*exp(x+y)
b = diff(uu,y)
d = -diff(uu,x)
r = sin(2*pi*y)*sin(2*pi*x)
print "b:    ",b
print "d:    ",d
print "r:    ",r
print "\n"
L1 = diff(d,x,y) - diff(b,y,y)
L2 = diff(b,x,y) - diff(d,x,x)
print "Curl 1:    ",L1
print "Curl 2:    ",L2
print "\n"


print "\n"
R1 = diff(r,x)
R2 = diff(r,y)
print "Multi 1:    ",R1
print "Multi 2:    ",R2


print "\n\n"
print "NS couple"

NS1 = -d*(diff(d,x)-diff(b,y))
NS2 = b*(diff(d,x)-diff(b,y))
print "\n"
print "NS_couple 1:    ",NS1
print "\n"
print "NS_couple 2:    ",NS2

print "\n\n"
print "Maxwell couple"

M1 = diff(u*d-v*b,y)
M2 = -diff(u*d-v*b,x)
print "\n"
print "Maxwell_couple 1:    ",M1

print "\n"
print "Maxwell_couple 2:    ",M2