from dolfin import *
from numpy import *

import scipy as Sci
#import scipy.linalg
from math import pi,sin,cos,sqrt
import scipy.sparse as sps
import scipy.io as save
import scipy
import ipdb
import os

class BDMelements(object):

    def Laplacian(self,mesh,u,v,alpha,gamma):

        n = FacetNormal(mesh)
        h = CellSize(mesh)
        h_avg =avg(h)

        a = inner(grad(v), grad(u))*dx \
           - inner(avg(grad(v)), outer(u('+'),n('+'))+outer(u('-'),n('-')))*dS \
           - inner(outer(v('+'),n('+'))+outer(v('-'),n('-')), avg(grad(u)))*dS \
           + alpha/h_avg*inner(outer(v('+'),n('+'))+outer(v('-'),n('-')),outer(u('+'),n('+'))+outer(u('-'),n('-')))*dS \
           + gamma/h*inner(v,u)*ds

        return a


    def Stokes(self,mesh,u,v,p,q,alpha,gamma):
        A = Laplacian(self,mesh,u,v,alpha,gamma)
