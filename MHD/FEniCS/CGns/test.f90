program test

	implicit  none

	integer(8) :: i
	real(4) :: x, y, z
	open (unit = 1, file = "A_epetra.txt")
	do i = 1,276937
		read(1,*)x, y, z
!		write(*,'(3f11.3)')x, y, z
	enddo
!	open(unit=1, file='A_epetra.txt')
!	read(1,*)i, x, y, z
!	write(*,'(i16, 3f11.3)')i, x, y, z
!	end

end program

      subroutine ReadTXT(x,y,z)
      double precision u(n,m)
      double precision dx,dy,dx2,dy2,dnr_inv,tmp,diff
      integer n,m,i,j
cf2py intent(in) :: dx,dy
cf2py intent(in,out) :: u
cf2py intent(out) :: error
cf2py intent(hide) :: n,m
      dx2 = dx*dx
      dy2 = dy*dy
      dnr_inv = 0.5d0 / (dx2+dy2)
      error = 0d0
      do 200,j=2,m-1
         do 100,i=2,n-1
            tmp = u(i,j)
            u(i,j) = ((u(i-1,j) + u(i+1,j))*dy2+
     &           (u(i,j-1) + u(i,j+1))*dx2)*dnr_inv
            diff = u(i,j) - tmp
            error = error + diff*diff
 100     continue
 200  continue
      error = sqrt(error)
      end