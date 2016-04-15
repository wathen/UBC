from __future__ import division
import dolfin
from block_mat import block_mat
from block_vec import block_vec
import numpy

class block_bc(list):
    """This class applies Dirichlet BCs to a block matrix. It is not a block operator itself."""
    def apply(self, A=None, b=None, symmetric=True, save_A=False, signs=None):
        # Clean up self, and check arguments
        for i in range(len(self)):
            if self[i] is None:
                self[i] = []
            if not hasattr(self[i], '__iter__'):
                self[i] = [self[i]]

        if A is None and b is None:
            raise TypeError('too few arguments to apply')
        if b is None and isinstance(A, block_vec):
            A,b = None,A
        if (A and not isinstance(A, block_mat)) or (b and not isinstance(b, block_vec)):
            raise TypeError('arguments of wrong type')
        if A is None:
            return self.apply_rhs(b, symmetric)
        if save_A and not symmetric:
            raise TypeError('no point in saving A if symmetric=False')
        if symmetric and dolfin.MPI.num_processes() > 1:
            raise RuntimeError('symmetric BCs not supported in parallel (yet)')

        # Find signs of matrices. Robust for definite matrices. For indefinite matrices, the
        # sign doesn't matter. For semi-definite matrices, we may get wrong answer.
        if signs:
            self.signs = signs
        if not hasattr(self, 'signs'):
            self.compute_signs(A, b)

        if save_A:
            # Save a copy of the matrices, for subsequent RHS modification.
            self.A_unmod = A.copy()

        self.apply_matvec(A, b, symmetric)

    def compute_signs(self, AA, bb):
        self.signs = [None]*len(self)
        bb.allocate(AA, dim=0)
        for i in range(len(self)):
            if not self[i]:
                # No BC on this block, sign doesn't matter
                continue
            if numpy.isscalar(AA[i,i]):
                xAx = AA[i,i]
            else:
                # Do not use a constant vector, as that may be in the null space
                # before boundary conditions are applied
                x = AA[i,i].create_vec(dim=1)
                ran = numpy.random.random(x.local_size())
                x.set_local(ran)
                Ax = AA[i,i]*x
                xAx = x.inner(Ax)
            if xAx == 0:
                from dolfin import warning
                warning("block_bc: zero or semi-definite block (%d,%d), using sign +1"%(i,i))
            self.signs[i] = -1 if xAx < 0 else 1
        dolfin.info('Calculated signs of diagonal blocks:' + str(self.signs))

    def apply_matvec(self, A, b, symmetric):
        b.allocate(A, dim=0)
        for i in range(len(self)):
            for bc in self[i]:
                for j in range(len(self)):
                    if i==j:
                        if numpy.isscalar(A[i,i]):
                            # Convert to a diagonal matrix, so that the individual rows can be modified
                            import block.algebraic
                            A[i,i] = block.algebraic.active_backend().create_identity(b[i], val=A[i,i])
                        if symmetric:
                            bc.zero_columns(A[i,i], b[i], self.signs[i])
                        else:
                            bc.apply(A[i,i], b[i])
                    else:
                        if numpy.isscalar(A[i,j]):
                            if A[i,j] != 0:
                                dolfin.error("can't modify block (%d,%d) for BC, expected a GenericMatrix" % (i,j))
                            continue
                        bc.zero(A[i,j])
                        if symmetric:
                            bc.zero_columns(A[j,i], b[j])

    def apply_rhs(self, b, symmetric):
        # First, collect a vector containing all non-zero BCs. These are required for
        # symmetric modification, and for changing sign
        b_mod = b.copy()
        b_mod.zero()
        for i in range(len(self)):
            for bc in self[i]:
                if symmetric or self.signs[i] == -1:
                    bc.apply(b_mod[i])

        if symmetric:
            if not hasattr(self, 'A_unmod'):
                raise RuntimeError('for symmetric modification, apply() must have been called with save_A=True')

            # The non-zeroes of b_mod are now exactly the x values (assuming the
            # matrix diagonal is in fact 1). We can thus create the necessary modifications
            # to b by just multiplying with the un-symmetricised original matrix.
            b -= self.A_unmod * b_mod

        # Apply the actual BC dofs to b. (This must be done after the symmetric
        # correction above, since the correction might also change the BC dofs.)
        for i in range(len(self)):
            for bc in self[i]:
                # Must call bc.apply even if sign==1, to set zero BCs
                bc.apply(b[i])
                if self.signs[i] == -1:
                    # Then we only need to consider the nonzseroes...
                    B = b_mod[i].array()
                    idx = numpy.where(B != 0)[0]
                    if len(idx):
                        b[i][idx] = -B[idx]
