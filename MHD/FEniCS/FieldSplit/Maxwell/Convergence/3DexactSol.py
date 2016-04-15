from sympy import *

x = symbols('x[0]')
y = symbols('x[1]')
z = symbols('x[2]')
uu = sin(x)*exp(x+y+z)
uv = sin(y)*exp(x+y+z)
uw = sin(z)*exp(x+y+z)
u = diff(uw,y)-diff(uv,z)
v = diff(uu,z)-diff(uw,x)
w = diff(uv,x)-diff(uu,y)
p = sin(2*pi*y)*sin(2*pi*x)*sin(2*pi*z)
print "u:    ",u
print "\n"
print "v:    ",v
print "\n"
print "w:    ",w
print "\n"
print "p:    ",p
print "\n"
L1 = diff(v,x,y) - diff(u,y,y) - diff(u,z,z) +diff(w,x,z)
L2 = diff(w,y,z) - diff(v,z,z) - diff(v,x,x) +diff(u,x,y)
L3 = diff(u,x,z) - diff(w,x,x) - diff(w,y,y) +diff(v,y,z)
print "Curl 1:    ",L1
print "\n"
print "Curl 2:    ",L2
print "\n"
print "Curl 3:    ",L3
print "\n"


print "\n"
P1 = diff(p,x)
P2 = diff(p,y)
P3 = diff(p,z)
print "Pressure 1:    ",P1
print "\n"
print "Pressure 2:    ",P2
print "\n"
print "Pressure 3:    ",P3


print diff(u,x)+diff(v,y)+diff(w,z)

 # (exp(x + y)*sin(y) + exp(x + y)*cos(y))**2 - 2*exp(2*x + 2*y)*sin(y)*cos(y)


