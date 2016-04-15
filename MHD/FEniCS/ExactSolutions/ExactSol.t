[?1034h
   >>>>>>>>>>>>>>>>>>>>>>>>>>
     MHD 2D Exact Solution:
   >>>>>>>>>>>>>>>>>>>>>>>>>>

 ----------------------
   NS Exact Solution:
 ----------------------
Case  1 :

  u = ( exp(x)*sin(y) , exp(x)*cos(y) )

  p = ( sin(x)*cos(y) )

Case  2 :

  u = ( y**3 , x**3 )

  p = ( x**2 )

Case  3 :

  u = ( exp(x + y)*sin(y) + exp(x + y)*cos(y) , -exp(x + y)*sin(y) )

  p = ( x**3*sin(y) + exp(x + y) )

Case  4 :

  u = ( x*y*exp(x + y) + x*exp(x + y) , -x*y*exp(x + y) - y*exp(x + y) )

  p = ( exp(y)*sin(x) )

Case  5 :

  u = ( y**2 , x**2 )

  p = ( x )

 ---------------------------
   Maxwell Exact Solution:
 ---------------------------
Case  1 :

  b = ( exp(x + y)*cos(x) , exp(x + y)*sin(x) - exp(x + y)*cos(x) )

  r = ( x*sin(2*pi*x)*sin(2*pi*y) )

Case  2 :

  b = ( y**2*(y - 1) , x**2*(x - 1) )

  r = ( x*y*(x - 1)*(y - 1) )

Case  3 :

  b = ( x*cos(x) , x*y*sin(x) - y*cos(x) )

  r = ( x*sin(2*pi*x)*sin(2*pi*y) )

Case  4 :

  b = ( 0 , 0 )

  r = ( 0 )








   >>>>>>>>>>>>>>>>>>>>>>>>>>
     MHD 3D Exact Solution:
   >>>>>>>>>>>>>>>>>>>>>>>>>>

 ----------------------
   NS Exact Solution:
 ----------------------
Case  1 :

  u = ( -exp(x + y + z)*sin(y) + exp(x + y + z)*sin(z) , exp(x + y + z)*sin(x) - exp(x + y + z)*sin(z) , -exp(x + y + z)*sin(x) + exp(x + y + z)*sin(y) )

  p = ( exp(x + y + z) + sin(y) )

Case  2 :

  u = ( 5*y**4 - 6*z**5 , -4*x**3 + 6*z**5 , 4*x**3 - 5*y**4 )

  p = ( x*y*z )

Case  3 :

  u = ( y**3*z**3 , x**3*z**3 , x**3*y**3 )

  p = ( x**2*y**2*z**2 )

Case  4 :

  u = ( y**2*z**2 , x**2*z**2 , x**2*y**2 )

  p = ( x*y*z )

Case  5 :

  u = ( y*z , x*z , x*y )

  p = ( x + y + z )

Case  6 :

  u = ( y*z**2 , x*z**2 , 0 )

  p = ( x*y )

Case  7 :

  u = ( -x*y*exp(x + y + z) + x*z*exp(x + y + z) , x*y*exp(x + y + z) - y*z*exp(x + y + z) , -x*z*exp(x + y + z) + y*z*exp(x + y + z) )

  p = ( exp(x + y + z)*sin(y) )

Case  8 :

  u = ( z**2*sin(y) , x*z**2 , 0 )

  p = ( y*sin(x) )

 ---------------------------
   Maxwell Exact Solution:
 ---------------------------
Case  1 :

  b = ( -exp(x + y + z)*sin(y) + exp(x + y + z)*sin(z) , exp(x + y + z)*sin(x) - exp(x + y + z)*sin(z) , -exp(x + y + z)*sin(x) + exp(x + y + z)*sin(y) )

  r = ( sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z) )

Case  2 :

  b = ( -y*exp(x + y + z) + z*exp(x + y + z) , -z*exp(x + y + z) + exp(x + y + z)*cos(x + y) , y*exp(x + y + z) + exp(x + y + z)*sin(x + y) - exp(x + y + z)*cos(x + y) )

  r = ( sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z) )

Case  3 :

  b = ( x**2*y**2*(-x + 1)*(-y + 1) , x**2*z**2*(-x + 1)*(-z + 1) , x**2*y**2*(-x + 1)*(-y + 1) )

  r = ( x**2*y*(x - 1)**2*(y - 1) )

Case  4 :

  b = ( -y*exp(x + y + z) + z*exp(x + y + z) , -z*exp(x + y + z) + exp(x + y + z) , y*exp(x + y + z) - exp(x + y + z) )

  r = ( sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z) )

Case  5 :

  b = ( -y*exp(y + z) , -exp(x + z) , -exp(x) )

  r = ( sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z) )

