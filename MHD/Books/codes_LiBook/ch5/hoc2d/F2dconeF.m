% Calculate the RHS function

function rhsF=F2dconeF(x,y,t)
ID=4;
if ID==1    % rotating cone model
   rt=(2+sin(pi*t))/4;
   st=(2+cos(pi*t))/4;
   tmp=exp(-80*((x-rt)^2+(y-st)^2));
   ut=0.8*tmp*40*pi*((x-rt)*cos(pi*t)-(y-st)*sin(pi*t));
   uxx=0.8*tmp*(160^2*(x-rt)^2-160);
   uyy=0.8*tmp*(160^2*(y-st)^2-160);
   rhsF=ut-uxx-uyy;
elseif ID==2   % periodic function model
   rhsF=0;
elseif ID==4
    rhsF=(-1+16*pi*pi)*exp(-t)*(sin(4*pi*x)+sin(4*pi*y));
end
