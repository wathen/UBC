import numpy as np
import scipy as sp
import scipy.sparse as sps

def MyMinres(A,b,uInt,tol,maxit):
    N = A.shape[0]

    v1 = np.zeros((n,1))
    w1 = np.zeros((n,1))
    w2 = np.zeros((n,1))
    v1 = b - np.dot(A,uInt)
    #precond solve
    #z1 = M\v1
    z1 = v1


    gamma = np.sqrt(np.inner(z1,v1))

    eta = gamma;
    s0 = 0; s1 = 0;
    c0= 1; c1 = 1;

    for i in xrange(1,2):
        z1 = z1/gamma
        print z1
        Az = np.dot(A,z1.T)
        delta = np.inner(Az,z1.T)

        if i == 1:
            v2 = Az - (delta/gamma)*v1.T
        else:
            v2 = Az - (delta/gamma)*v1.T -(gamma/gammaOld)*v0

        #second precond solve
        #z2 = M\v2;
        z2 = v2

        gammaNew = np.sqrt(np.dot(z2,v2.T))
        print gammaNew
        alpha0 = c1*delta - c0*s1*gamma
        alpha1 = np.sqrt(alpha0**2 + gammaNew**2)
        alpha2 = s1*delta + c0*c1*gamma
        alpha3 = s0*gamma
        s0 = s1; c0 = c1;
        c1 = alpha0/alpha1; s1 = gammaNew/alpha1;


n = 10;
Adiag = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
A = np.mat(np.diag(Adiag))
b = np.mat([11.,12.,13.,412.,5.,3.,56.,1.,5.,6.])

print b



tol = 1e-6
uInt = np.arange(n)
maxit = 100
M = np.eye((n))

MyMinres(A,b,uInt,tol,maxit)


#  [U,iter] = mgminresWorking(AA,b,uInt,tol,maxit,N,M)

# [n,~] = size(AA);
# v1 = sparse(n,1);
# w1 = sparse(n,1);
# w2 = sparse(n,1);
# nn = N;

# v1 = b - AA*uInt;
# z1 = M\v1;
# gamma = sqrt(z1'*v1);
# eta = gamma;
# s0 = 0; s1 = 0;
# c0= 1; c1 = 1;

# for i = 1:maxit

#     z1 = z1/gamma;
#     Az = AA*z1;
#     delta = (Az'*z1);
#     if i == 1
#         v2 = Az - (delta/gamma)*v1;
#     else
#         v2 = Az - (delta/gamma)*v1 - ...
#             (gamma/gammaOld)*v0;
#     end
#     z2 = M\v2;
# %     z2 = z2(:);
    # gammaNew = sqrt(z2'*v2);
    # alpha0 = c1*delta - c0*s1*gamma;
    # alpha1 = sqrt(alpha0^2 + gammaNew^2);
    # alpha2 = s1*delta + c0*c1*gamma;
    # alpha3 = s0*gamma;
#     s0 = s1; c0 = c1;
#     c1 = alpha0/alpha1; s1 = gammaNew/alpha1;
#     if i == 1
#         w2 = (z1 - alpha2*w1)/alpha1;
#         u = uInt + c1*eta*w2;
#     else
#         w2 = (z1 - alpha3*w0 - alpha2*w1)/alpha1;
#         u = u + c1*eta*w2;
#     end
#     eta = -s1*eta;
#     if norm(AA*u-b)/norm(b) <= tol
#         break
#     end
#     gammaOld = gamma;
#     gamma = gammaNew;
#     z1 = z2;
#     w0 = w1;
#     w1 = w2;
#     v0 = v1;
#     v1 = v2;
# end
# iter = i;