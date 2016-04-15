from sympy import *

x = symbols('x[0]')
y = symbols('x[1]')
uu = cos(x)*exp(x+y)
u = diff(uu,y)
v = -diff(uu,x)
p = sin(2*pi*y)*sin(2*pi*x)
print "u:    ",u
print "\n"
print "v:    ",v
print "\n"
print "p:    ",p
print "\n"
L1 = diff(v,x,y) - diff(u,y,y)
L2 = diff(u,x,y) - diff(v,x,x)
print "Curl 1:    ",L1
print "\n"
print "Curl 2:    ",L2
print "\n"


print "\n"
P1 = diff(p,x)
P2 = diff(p,y)
print "Pressure 1:    ",P1
print "\n"
print "Pressure 2:    ",P2




 # (exp(x + y)*sin(y) + exp(x + y)*cos(y))**2 - 2*exp(2*x + 2*y)*sin(y)*cos(y)


