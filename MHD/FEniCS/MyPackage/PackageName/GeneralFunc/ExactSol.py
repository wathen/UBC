from sympy import *
from dolfin import Expression
import numpy as np
import PrintFuncs

def Print2D(u,v,p,opt):
    if opt == "NS":
        print "  u = (",str(u).replace('x[0]','x').replace('x[1]','y'),",",str(v).replace('x[0]','x').replace('x[1]','y'),")\n"
        print "  p = (",str(p).replace('x[0]','x').replace('x[1]','y'),")\n"
    if opt == "M":
        print "  b = (",str(u).replace('x[0]','x').replace('x[1]','y'),",",str(v).replace('x[0]','x').replace('x[1]','y'),")\n"
        print "  r = (",str(p).replace('x[0]','x').replace('x[1]','y'),")\n"

def Print3D(u,v,w,p,opt):
    if opt == "NS":
        print "  u = (",str(u).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),",",str(v).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),",",str(w).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),")\n"
        print "  p = (",str(p).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),")\n"
    if opt == "M":
        print "  b = (",str(u).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),",",str(v).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),",",str(w).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),")\n"
        print "  r = (",str(p).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),")\n"


"""
================================
                                 NS exact
================================
"""

"""
----------------------2D----------------------
"""


def Fluid2D(case,Show="no",type = 'no'):
    x = symbols('x[0]')
    y = symbols('x[1]')

    PrintFuncs.PrintStr("NS Exact Solution:",3,"-")
    if Show == "yes":
        case = 1

    if case == 1:
        u = sin(y)*exp(x)
        v = cos(y)*exp(x)
        p = sin(x)*cos(y)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print2D(u,v,p,"NS")
    if case == 2:
        u = pow(y,3)
        v = pow(x,3)
        p = pow(x,2)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print2D(u,v,p,"NS")
    if case == 3:
        u = sin(y)*exp(x+y)+cos(y)*exp(x+y)
        v = -sin(y)*exp(x+y)
        p = pow(x,3)*sin(y)+exp(x+y)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print2D(u,v,p,"NS")
    if case == 4:
        uu = y*x*exp(x+y)
        u = diff(uu,y)
        v = -diff(uu,x)
        p = sin(x)*exp(y)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print2D(u,v,p,"NS")
    if case == 5:
        # uu = y*x*exp(x+y)
        u = y**2
        v = x**2
        p = x
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print2D(u,v,p,"NS")
    if case == 6:
        u = y*(y-1)*x*(x-1)
        v = y*(y-1)*x*(x-1)
        p = y*(y-1)*x*(x-1)
    if case == 7:
        l = 0.54448373678246
        omega = (3./2)*np.pi
        # rho = sqrt(x*x+y*y)

        phi = symbols('phi')
        rho = symbols('rho')
        psi = sin((1+l)*phi)*cos(l*omega)/(1+l) - cos((1+l)*phi) -sin((1-l)*phi)*cos(l*omega)/(1-l) + cos((1-l)*phi)
        psi_prime = diff(psi, phi)
        psi_3prime = diff(psi, phi, phi, phi)

        u = rho**l*((1+l)*sin(phi)*psi + cos(phi)*psi_prime)
        v = rho**l*(-(1+l)*cos(phi)*psi + sin(phi)*psi_prime)
        p = -rho**(l-1)*((1+l)**2*psi_prime + psi_3prime)/(1-l)

        u = u.subs(phi, atan2(y,x))
        v = v.subs(phi, atan2(y,x))
        p = p.subs(phi, atan2(y,x))

        u = u.subs(rho, sqrt(x*x + y*y))
        v = v.subs(rho, sqrt(x*x + y*y))
        p = p.subs(rho, sqrt(x*x + y*y))

    L1 = diff(u,x,x)+diff(u,y,y)
    L2 = diff(v,x,x)+diff(v,y,y)

    A1 = u*diff(u,x)+v*diff(u,y)
    A2 = u*diff(v,x)+v*diff(v,y)

    P1 = diff(p,x)
    P2 = diff(p,y)


    u0 = Expression((ccode(u),ccode(v)))
    p0 = Expression(ccode(p))

    Laplacian = Expression((ccode(L1),ccode(L2)))
    Advection = Expression((ccode(A1),ccode(A2)))
    gradPres = Expression((ccode(P1),ccode(P2)))
    if Show == "no":
        Print2D(u,v,p,"NS")
    if type == "MHD":
        return u, v, p, u0, p0, Laplacian, Advection, gradPres
    else:
        return u0, p0, Laplacian, Advection, gradPres


"""
----------------------3D----------------------
"""

def Fluid3D(case,Show="no",type="no"):
    x = symbols('x[0]')
    y = symbols('x[1]')
    z = symbols('x[2]')

    PrintFuncs.PrintStr("NS Exact Solution:",3,"-")
    if Show == "yes":
        case = 1

    if case == 1:
        uu = sin(x)*exp(x+y+z)
        uv = sin(y)*exp(x+y+z)
        uw = sin(z)*exp(x+y+z)
        u = diff(uw,y)-diff(uv,z)
        v = diff(uu,z)-diff(uw,x)
        w = diff(uv,x)-diff(uu,y)
        p = sin(y)+exp(x+y+z)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print3D(u,v,w,p,"NS")
    if case == 2:
        uu = (x**4+y**5+z**6)
        uv = (x**4+y**5+z**6)
        uw =(x**4+y**5+z**6)
        u = diff(uw,y)-diff(uv,z)
        v = diff(uu,z)-diff(uw,x)
        w = diff(uv,x)-diff(uu,y)
        p = x*y*z
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print3D(u,v,w,p,"NS")
    if case == 3:
        u = y**3*z**3
        v = x**3*z**3
        w = y**3*x**3
        p = x**2*y**2*z**2
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print3D(u,v,w,p,"NS")
    if case == 4:
        u = y**2*z**2
        v = x**2*z**2
        w = y**2*x**2
        p = x*y*z
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print3D(u,v,w,p,"NS")
    if case == 5:
        u = y*z
        v = x*z
        w = y*x
        p = x+y+z
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print3D(u,v,w,p,"NS")
    if case == 6:
        u = y*z**2
        v = x*z**2
        w = diff(u,x)
        p = x*y
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print3D(u,v,w,p,"NS")
    if case == 7:
        uu = x*y*z*exp(x+y+z)
        uv = x*y*z*exp(x+y+z)
        uw = x*y*z*exp(x+y+z)
        u = diff(uw,y)-diff(uv,z)
        v = diff(uu,z)-diff(uw,x)
        w = diff(uv,x)-diff(uu,y)
        p = sin(y)*exp(x+y+z)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print3D(u,v,w,p,"NS")
    if case == 8:
        u = sin(y)*z**2
        v = x*z**2
        w = diff(u,x)
        p = sin(x)*y
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print3D(u,v,w,p,"NS")
    if u:
        f = 1
    else:
        print "No case selected"
        return

    L1 = diff(u,x,x)+diff(u,y,y) + diff(u,z,z)
    L2 = diff(v,x,x)+diff(v,y,y) + diff(v,z,z)
    L3 = diff(w,x,x)+diff(w,y,y) + diff(w,z,z)

    A1 = u*diff(u,x)+v*diff(u,y)+w*diff(u,z)
    A2 = u*diff(v,x)+v*diff(v,y)+w*diff(v,z)
    A3 = u*diff(w,x)+v*diff(w,y)+w*diff(w,z)

    P1 = diff(p,x)
    P2 = diff(p,y)
    P3 = diff(p,z)

    u0 = Expression((ccode(u),ccode(v),ccode(w)))
    p0 = Expression(ccode(p))
    Laplacian = Expression((ccode(L1),ccode(L2),ccode(L3)))
    Advection = Expression((ccode(A1),ccode(A2),ccode(A3)))
    gradPres = Expression((ccode(P1),ccode(P2),ccode(P3)))

    if Show == 'no':
        Print3D(u,v,w,p,"NS")
    if type == "MHD":
        return u,v,w,p,u0, p0, Laplacian, Advection, gradPres
    else:
        return u0, p0, Laplacian, Advection, gradPres




"""
================================
                               Maxwell exact
================================
"""

"""
----------------------2D----------------------
"""

def Mag2D(case,Show="no",type="no", Mass = 0):
    x = symbols('x[0]')
    y = symbols('x[1]')
    if Show == "yes":
        case = 1


    PrintFuncs.PrintStr("Maxwell Exact Solution:",3,"-")
    if case == 1:
        uu = cos(x)*exp(x+y)
        u = diff(uu,y)
        v = -diff(uu,x)
        p = x*sin(2*pi*y)*sin(2*pi*x)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print2D(u,v,p,"M")
    if case == 2:
        u = y*y*(y-1)
        v = x*x*(x-1)
        p = y*(y-1)*x*(x-1)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print2D(u,v,p,"M")
    if case == 3:
        uu = x*y*cos(x)
        u = diff(uu,y)
        v = -diff(uu,x)
        p = x*sin(2*pi*y)*sin(2*pi*x)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print2D(u,v,p,"M")
    if case == 4:
        uu = x*cos(x)
        u = diff(uu,y)
        v = diff(uu,y)
        p = diff(uu,y)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print2D(u,v,p,"M")
    if case == 5:
        l = 0.54448373678246
        z = symbols('z')
        omega = (3./2)*np.pi
        rho = sqrt(x*x+y*y)
        phi = atan2(y,x)
        u = diff(rho**(2./3)*sin((2./3)*phi),x)
        v = diff(rho**(2./3)*sin((2./3)*phi),y)
        p = diff(rho,z)

    if u:
        f = 1
    else:
        print "No case selected"
        return

    L1 = diff(v,x,y) - diff(u,y,y)
    L2 = diff(u,x,y) - diff(v,x,x)

    P1 = diff(p,x)
    P2 = diff(p,y)

    u0 = Expression((ccode(u),ccode(v)))
    p0 = Expression(ccode(p).replace('M_PI','pi'))

    CurlCurl = Expression((ccode(L1),ccode(L2)))
    gradPres = Expression((ccode(P1).replace('M_PI','pi'),ccode(P2).replace('M_PI','pi')))
    CurlMass = Expression((ccode(Mass*u+L1),ccode(Mass*v+L2)))

    if Show == 'no':
        print "  b = (",str(u).replace('x[0]','x').replace('x[1]','y'),",",str(v).replace('x[0]','x').replace('x[1]','y'),")\n"
        print "  p = (",str(p).replace('x[0]','x').replace('x[1]','y'),")\n"
    if type == "MHD":
        if Mass == 0:
            return u,v,p,u0, p0, CurlCurl, gradPres
        else:
            return u,v,p,u0, p0, CurlCurl, gradPres, CurlMass
    else:
        if Mass == 0:
            return u0, p0, CurlCurl, gradPres
        else:
            return u0, p0, CurlCurl, gradPres, CurlMass


"""
----------------------3D----------------------
"""

def Mag3D(case,Show="no",type="no",Mass=0):
    x = symbols('x[0]')
    y = symbols('x[1]')
    z = symbols('x[2]')

    PrintFuncs.PrintStr("Maxwell Exact Solution:",3,"-")
    if case == 1:
        uu = sin(x)*exp(x+y+z)
        uv = sin(y)*exp(x+y+z)
        uw = sin(z)*exp(x+y+z)
        u = diff(uw,y)-diff(uv,z)
        v = diff(uu,z)-diff(uw,x)
        w = diff(uv,x)-diff(uu,y)
        p = sin(2*pi*y)*sin(2*pi*x)*sin(2*pi*z)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print3D(u,v,w,p,"M")
    if case == 2:
        uu = cos(x+y)*exp(x+y+z)
        uv = y*exp(x+y+z)
        uw = z*exp(x+y+z)
        u = diff(uw,y)-diff(uv,z)
        v = diff(uu,z)-diff(uw,x)
        w = diff(uv,x)-diff(uu,y)
        p = sin(2*pi*y)*sin(2*pi*x)*sin(2*pi*z)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print3D(u,v,w,p,"M")
    if case == 3:
        u = y*y*(1-y)*x*x*(1-x)
        v = x*x*(1-x)*z*z*(1-z)
        w = x*x*(1-x)*y*y*(1-y)
        p = y*(y-1)*x*(x-1)*x*(x-1)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print3D(u,v,w,p,"M")
    if case == 4:
        uu = exp(x+y+z)
        uv = y*exp(x+y+z)
        uw = z*exp(x+y+z)
        u = diff(uw,y)-diff(uv,z)
        v = diff(uu,z)-diff(uw,x)
        w = diff(uv,x)-diff(uu,y)
        p = sin(2*pi*y)*sin(2*pi*x)*sin(2*pi*z)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print3D(u,v,w,p,"M")
    if case == 5:
        uu = y*exp(x)
        uv = y*exp(y+z)
        uw = exp(x+z)
        u = diff(uw,y)-diff(uv,z)
        v = diff(uu,z)-diff(uw,x)
        w = diff(uv,x)-diff(uu,y)
        p = sin(2*pi*y)*sin(2*pi*x)*sin(2*pi*z)
        if Show == "yes":
            case +=1
            print "Case ",case-1,":\n"
            Print3D(u,v,w,p,"M")

    if u:
        f = 1
    else:
        print "No case selected"
        return

    L1 = diff(v,x,y) - diff(u,y,y) - diff(u,z,z) +diff(w,x,z)
    L2 = diff(w,y,z) - diff(v,z,z) - diff(v,x,x) +diff(u,x,y)
    L3 = diff(u,x,z) - diff(w,x,x) - diff(w,y,y) +diff(v,y,z)
    P1 = diff(p,x)
    P2 = diff(p,y)
    P3 = diff(p,z)


    u0 = Expression((ccode(u),ccode(v),ccode(w)))
    p0 = Expression(ccode(p).replace('M_PI','pi'))
    CurlCurl = Expression((ccode(L1),ccode(L2),ccode(L3)))
    gradPres = Expression((ccode(P1).replace('M_PI','pi'),ccode(P2).replace('M_PI','pi'),ccode(P3).replace('M_PI','pi')))
    CurlMass = Expression((ccode(Mass*u+L1),ccode(Mass*v+L2),ccode(Mass*w+L3)))


    if Show == 'no':
        PrintFuncs.PrintStr("Maxwell Exact Solution:",3,"-")
        print "  b = (",str(u).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),",",str(v).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),",",str(w).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),")\n"
        print "  p = (",str(p).replace('x[0]','x').replace('x[1]','y').replace('x[2]','z'),")\n"

    if type == "MHD":
        if Mass == 0:
            return u,v,w,p,u0, p0, CurlCurl, gradPres
        else:
            return u,v,w,p,u0, p0, CurlCurl, gradPres, CurlMass
    else:
        if Mass == 0:
            return u0, p0, CurlCurl, gradPres
        else:
            return u0, p0, CurlCurl, gradPres, CurlMass





"""
================================
                                MHD exact
================================
"""

"""
----------------------2D----------------------
"""


def MHD2D(NScase,Mcase,Show="no"):

    PrintFuncs.PrintStr("MHD 2D Exact Solution:",5,">","\n","\n")

    x = symbols('x[0]')
    y = symbols('x[1]')
    u, v, p, u0, p0, Laplacian, Advection, gradPres = Fluid2D(NScase,Show,"MHD")
    b,d,r,b0, r0, CurlCurl, gradR = Mag2D(Mcase,Show,"MHD")

    NS1 = -d*(diff(d,x)-diff(b,y))
    NS2 = b*(diff(d,x)-diff(b,y))

    M1 = diff(u*d-v*b,y)
    M2 = -diff(u*d-v*b,x)
    NS_Couple = Expression((ccode(NS1),ccode(NS2)))
    M_Couple = Expression((ccode(M1),ccode(M2)))

    return u0, p0,b0, r0, Laplacian, Advection, gradPres,CurlCurl, gradR, NS_Couple, M_Couple


"""
----------------------3D----------------------
"""


def MHD3D(NScase,Mcase,Show="no"):
    PrintFuncs.PrintStr("MHD 3D Exact Solution:",5,">","\n","\n")

    x = symbols('x[0]')
    y = symbols('x[1]')
    z = symbols('x[2]')
    u, v,w, p, u0, p0, Laplacian, Advection, gradPres = Fluid3D(NScase,Show,"MHD")
    b,d,e,r,b0, r0, CurlCurl, gradR = Mag3D(Mcase,Show,"MHD")


    f = u*e-d*w
    g = b*w-u*e
    h = u*d-v*d
    NS1 = diff(h,y)-diff(g,z)
    NS2 = diff(f,z)-diff(h,x)
    NS3 = diff(g,x)-diff(f,y)


    m = diff(e,y)-diff(d,z)
    n = diff(b,z)-diff(e,x)
    p = diff(d,x)-diff(b,y)
    M1 = n*e - d*p
    M2 = b*p - m*e
    M3 = m*d - n*b
    NS_Couple = Expression((ccode(NS1),ccode(NS2),ccode(NS3)))
    M_Couple = Expression((ccode(M1),ccode(M2),ccode(M3)))

    return u0, p0,b0, r0, Laplacian, Advection, gradPres,CurlCurl, gradR, NS_Couple, M_Couple

