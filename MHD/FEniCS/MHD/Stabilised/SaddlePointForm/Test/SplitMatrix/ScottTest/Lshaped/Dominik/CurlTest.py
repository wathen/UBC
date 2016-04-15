import sympy as sy

x = sy.symbols('x')
y = sy.symbols('y')
rho = sy.sqrt(x**2 + y**2)
phi = sy.atan2(y,x)

f = rho**(2./3)*sy.sin((2./3)*phi)
b = sy.diff(f,x)
d = sy.diff(f,y)
curlB = sy.diff(d, x) - sy.diff(b, y)

Error = 0
for j in range(21):
    for i in range(41):
        if abs(curlB.subs(x,float(i)/20-1).subs(y,float(j)/20).evalf()) > 1e-10:
            print "curl(b) != 0 for x = ", float(i)/100, ' and y = ', float(j)/100
            Error = 1

if Error != 1:
    print "Curl(b) in upper region = 0 for all (x,y)"
else:
    print "Curl(b) in upper region != 0 for all (x,y)"


curlBlower = curlB.xreplace({sy.atan2(y, x):(sy.atan2(y, x)+2*sy.pi)})
Error = 0

for j in range(21):
    for i in range(41):
        if abs(curlBlower.subs(x,float(i)/20-1).subs(y,float(j)/20-1).evalf()) > 1e-10:
            print "curl(b) != 0 for x = ", float(i)/100, ' and y = ', float(j)/100
            Error = 1

if Error != 1:
    print "Curl(b) in lower region = 0 for all (x,y)"
else:
    print "Curl(b) in lower region != 0 for all (x,y)"
