syms x y
syms u_sym b_sym p_sym r_sym w_sym d_sym
syms nu num G Ha kappa p0
syms uy by py

kappa = nu*num*Ha^2;

uy = G/(nu*Ha*tanh(Ha))*(1-cosh(y*Ha)/cosh(Ha));
by = G/kappa *(sinh(y*Ha)/sinh(Ha) -y);
py = -kappa/2*by^2+p0;

u_sym = [uy; 0];
b_sym = [by; 1];
p_sym = -G*x+py;
r_sym = 0*x;

w_sym = [G/(nu*Ha*tanh(Ha))*(1-cosh(y*Ha)/cosh(Ha)); 0];
d_sym = [G/kappa*(sinh(y*Ha)/sinh(Ha)-y); 1];

lapu_sym = simplify([diff(diff(u_sym(1),'x'),'x')+diff(diff(u_sym(1),'y'),'y'); ...
            diff(diff(u_sym(2),'x'),'x')+diff(diff(u_sym(2),'y'),'y')]);

gradu_sym = simplify([diff(u_sym(1),'x'), diff(u_sym(1),'y'); ...
             diff(u_sym(2),'x'), diff(u_sym(2),'y')]);

divu_sym = simplify(gradu_sym(1,1) + gradu_sym(2,2));

curlb_sym = simplify([0; 0; diff(b_sym(2),'x')-diff(b_sym(1),'y')]);

curlcurlb_sym = simplify([diff(curlb_sym(3),'y'); -diff(curlb_sym(3),'x')]);

gradp_sym = simplify([diff(p_sym,'x'); diff(p_sym,'y')]);

gradr_sym = simplify([diff(r_sym,'x'); diff(r_sym,'y')]);

%curl(uxd)
uxd = simplify(ourcross(u_sym,d_sym));
curl_uxd_sym = simplify([ diff(uxd(3),'y')-diff(uxd(2),'z'); ...
                -diff(uxd(3),'x')+diff(uxd(1),'z'); ...
                 diff(uxd(2),'x')-diff(uxd(1),'y')]);

%(curlb)xd
curlbxd = simplify(ourcross(curlb_sym, [d_sym;0]));             
f_sym = simplify(-nu*lapu_sym + gradu_sym * w_sym + gradp_sym - kappa*curlbxd(1:2)) 

g_sym = simplify(kappa*num*curlcurlb_sym + gradr_sym - kappa*curl_uxd_sym(1:2))