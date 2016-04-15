import petsc4py
import sys
petsc4py.init(sys.argv)
import petsc4py.PETSc as PETSc
from dolfin import *

import ipdb
import Forms
import Preconditioners
import Assemble
import Solvers
import Errors
import PETScFunc
import GeneralFunc
