from dolfin import *
from PackageName.GeneralFunc import common
from PackageName.PETScFunc import PETScMatOps
from numpy import concatenate, zeros, ones, zeros, append, array
from scipy.sparse import spdiags, bmat



def Boundary(Space,BoundaryMarkers):
    key = BoundaryMarkers.keys()
    BC = zeros(0)
    for i in range(len(key)):
        BC = append(BC,int(str(key[i])))
    Boundary = ones(Space.dim())
    Boundary[array(BC).astype('int')] = 0
    BCmarkers = spdiags(Boundary,0,Space.dim(),Space.dim())

    Boundary = zeros(Space.dim())
    Boundary[array(BC).astype('int')] = 1
    BC = spdiags(Boundary,0,Space.dim(),Space.dim())

    return BCmarkers, BC

def RHSAssemble(FS, L, u, eq):

    def boundary(x, on_boundary):
        return on_boundary

    if str(FS.__class__).find('dict') != -1:
        if eq == 'Fluid':
            bcu = DirichletBC(FS['Velocity'], u['u0'], boundary)
            bu = assemble(L['Velocity'])
            bcu.apply(bu)
            if L.has_key('Pressure'):
                bp = assemble(L['Pressure'])
                b = concatenate([bu.array(),bp.array()])
            else:
                b = concatenate([bu.array(),zeros(FS['Pressure'].dim())])
            VelocityMarkers = Boundary(FS['Velocity'],bcu.get_boundary_values())
            markers = {'Velocity': VelocityMarkers}

        elif eq == 'Maxwell':
            bcb = DirichletBC(FS['Magnetic'], u['b0'], boundary)
            bcr = DirichletBC(FS['Multiplier'], u['r0'], boundary)
            bb = assemble(L['Magnetic'])
            br = assemble(L['Multiplier'])
            bcb.apply(bb)
            bcr.apply(br)
            markers = {'Magnetic': Boundary(FS['Magnetic'],bcb.get_boundary_values()), 'Multiplier': Boundary(FS['Multiplier'],bcr.get_boundary_values())}
            b = concatenate([bb.array(),br.array()])

        elif eq == 'MHD':
            bcu = DirichletBC(FS['Velocity'], u['u0'], boundary)
            bcb = DirichletBC(FS['Magnetic'], u['b0'], boundary)
            bcr = DirichletBC(FS['Multiplier'], u['r0'], boundary)
            bu = assemble(L)
            bb = assemble(L)
            br = assemble(L)
            if L.has_key('Pressure'):
                bu = assemble(L['Pressure'])
            bcu.apply(bu)
            bcb.apply(bb)
            bcr.apply(br)
            markers = {'Velocity': bcu.markers(), 'Magnetic': bcb.markers(), 'Multiplier': bcr.markers()}

    else:
        if eq == 'Fluid':
            bcu = DirichletBC(FS.sub(0), u['u0'], boundary)
            b = assemble(L)
            bcu.apply(b)

        elif eq == 'Maxwell':
            bcb = DirichletBC(FS.sub(0), u['b0'], boundary)
            bcr = DirichletBC(FS.sub(1), u['r0'], boundary)
            b = assemble(L)
            bcb.apply(b)
            bcr.apply(b)

        elif eq == 'MHD':
            bcu = DirichletBC(FS.sub(0), u['u0'], boundary)
            bcb = DirichletBC(FS.sub(0), u['b0'], boundary)
            bcr = DirichletBC(FS.sub(0), u['r0'], boundary)
            bu = assemble(L)
            bb = assemble(L)
            br = assemble(L)
            bcu.apply(bu)
            bcb.apply(bb)
            bcr.apply(br)

        markers = 0.0
    return b, markers

def FluidAssemble(FS, A, L, u, BC, eq, Aout = {}, iter = 1):
    def boundary(x, on_boundary):
        return on_boundary
    if str(FS.__class__).find('dict') != -1:
        if eq == 'Stokes':
            Laplacian = assemble(A['Laplacian'])
            bcu = DirichletBC(FS['Velocity'], u['u0'], boundary)
            # bcu.apply(Laplacian)
            Laplacian = BC['Velocity'][0]*Laplacian.sparray()*BC['Velocity'][0]+BC['Velocity'][1]

        else:
            ConvecDiff = assemble(A['ConvecDiff'])
            bc.apply(ConvecDiff)
            ConvecDiff = ConvecDiff.sparray()

        if iter < 2:
            B = assemble(A['Grad'])
            B = B.sparray()*BC['Velocity'][0]

        if eq == 'Stokes':
            if A.has_key('Stab'):
                Stab = assemble(A['Stab'])
                Aout['A'] =  Laplacian.sparray()
                Aout['B'] = B
                Aout['C'] = Stab
            else:
                Aout['A'] =  Laplacian
                Aout['B'] = B
        else:
            if iter > 1:
                Aout['A'] = ConvecDiff
            else:
                Aout['A'] = ConvecDiff
                Aout['B'] = B
                if A.has_key['C']:
                    Aout['C'] = Stab
    else:
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(FS.sub(0), u['u0'], boundary)
        Aout, b = assemble_system(A, L, bc)


        aaas

    return Aout


def MagAssemble(FS, A, L, u, BC, Aout = {}):

    def boundary(x, on_boundary):
        return on_boundary

    if str(FS.__class__).find('dict') != -1:
        CurlCurl = assemble(A['CurlCurl'])
        CurlCurl = BC['Magnetic'][0]*CurlCurl.sparray()*BC['Magnetic'][0]+BC['Magnetic'][1]
        bcr = DirichletBC(FS['Multiplier'], u['r0'], boundary)

        B = assemble(A['Div'])
        B = BC['Multiplier'][0]*B.sparray()*BC['Magnetic'][0]
        Stab = BC['Multiplier'][1]
        Aout['A'] = CurlCurl
        Aout['B'] = B
        Aout['C'] = Stab

    else:
        def boundary(x, on_boundary):
            return on_boundary
        bcb = DirichletBC(FS.sub(0), u['b0'], boundary)
        bcr = DirichletBC(FS.sub(1), u['r0'], boundary)
        bc = [bcb, bcr]
        Aout, b = assemble_system(A, L, bc)
        print Aout.sparray().todense()
        common.StoreMatrix(Aout.sparray(),'A')
        print b.array()
        sss

    return Aout

def PETScAssemble(FS, A, b, opt = 'Matrix'):

    if str(FS.__class__).find('dict') != -1:
        if opt == 'Matrix':
            if A.has_key('C') == True:
                Aout = PETScMatOps.Scipy2PETSc(bmat([[A['A'],A['B'].T],
                    [A['B'],A['C']]]))
                common.StoreMatrix(bmat([[A['A'],A['B'].T],
                    [A['B'],A['C']]]),'A')

            else:
                Aout = PETScMatOps.Scipy2PETSc(bmat([[A['A'],A['B'].T],
                    [A['B'],None]]))
        else:
            for key, value in A.iteritems():
                A[key] = PETScMatOps.Scipy2PETSc(value)
            Aout = PETSc.Mat().createPython([FS.keys()[0] + FS.keys()[1], FS.keys()[0] + FS.keys()[1]])
            Aout.setType('python')
            p = MHDmulti.MHDmat(W,A)
            Aout.setPythonContext(p)
        b = PETScMatOps.arrayToVec(b)

    else:
        Aout, b = PETScMatOps.Assemble(A,b)

    print b.array
    ssss
    return Aout, b




