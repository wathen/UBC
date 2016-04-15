Downloading/unpacking scipy from https://pypi.python.org/packages/source/s/scipy/scipy-0.13.2.tar.gz#md5=fcd110802b0bf3505ba567cf831566e1
  Running setup.py egg_info for package scipy
    
    warning: no previously-included files matching '*_subr_*.f' found under directory 'scipy/linalg/src/id_dist/src'
    no previously-included directories found matching 'scipy/special/tests/data/boost'
    no previously-included directories found matching 'scipy/special/tests/data/gsl'
    no previously-included directories found matching 'doc/build'
    no previously-included directories found matching 'doc/source/generated'
    no previously-included directories found matching '*/__pycache__'
    warning: no previously-included files matching '*~' found anywhere in distribution
    warning: no previously-included files matching '*.bak' found anywhere in distribution
    warning: no previously-included files matching '*.swp' found anywhere in distribution
    warning: no previously-included files matching '*.pyo' found anywhere in distribution
Installing collected packages: scipy
  Found existing installation: scipy 0.12.0
    Uninstalling scipy:
      Successfully uninstalled scipy
  Running setup.py install for scipy
    blas_opt_info:
    blas_mkl_info:
      libraries mkl,vml,guide not found in ['/usr/local/lib', '/usr/lib', '/usr/lib/x86_64-linux-gnu']
      NOT AVAILABLE
    
    openblas_info:
      libraries openblas not found in ['/usr/local/lib', '/usr/lib', '/usr/lib/x86_64-linux-gnu']
      NOT AVAILABLE
    
    atlas_blas_threads_info:
    Setting PTATLAS=ATLAS
      libraries ptf77blas,ptcblas,atlas not found in ['/usr/local/lib', '/usr/lib/atlas-base', '/usr/lib', '/usr/lib/x86_64-linux-gnu']
      NOT AVAILABLE
    
    atlas_blas_info:
    customize Gnu95FCompiler
    Found executable /usr/bin/gfortran
    customize Gnu95FCompiler
    customize Gnu95FCompiler using config
    compiling '_configtest.c':
    
    /* This file is generated from numpy/distutils/system_info.py */
    void ATL_buildinfo(void);
    int main(void) {
      ATL_buildinfo();
      return 0;
    }
    
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-c'
    x86_64-linux-gnu-gcc: _configtest.c
    x86_64-linux-gnu-gcc -pthread _configtest.o -L/usr/lib/atlas-base -lf77blas -lcblas -latlas -o _configtest
    ATLAS version 3.10.1 built by buildd on Sat Jul 27 19:04:50 UTC 2013:
       UNAME    : Linux roseapple 3.2.0-37-generic #58-Ubuntu SMP Thu Jan 24 15:28:10 UTC 2013 x86_64 x86_64 x86_64 GNU/Linux
       INSTFLG  : -1 0 -a 1 -l 1
       ARCHDEFS : -DATL_OS_Linux -DATL_ARCH_x86SSE2 -DATL_CPUMHZ=1596 -DATL_SSE2 -DATL_SSE1 -DATL_USE64BITS -DATL_GAS_x8664
       F2CDEFS  : -DAdd_ -DF77_INTEGER=int -DStringSunStyle
       CACHEEDGE: 1048576
       F77      : /usr/bin/x86_64-linux-gnu-gfortran-4.8, version GNU Fortran (Ubuntu/Linaro 4.8.1-8ubuntu1) 4.8.1
       F77FLAGS : -fomit-frame-pointer -mfpmath=sse -O2 -msse2 -fPIC -m64
       SMC      : /usr/bin/c99-gcc, version gcc (Ubuntu/Linaro 4.8.1-8ubuntu1) 4.8.1
       SMCFLAGS : -fomit-frame-pointer -mfpmath=sse -O2 -msse2 -fPIC -m64
       SKC      : /usr/bin/c99-gcc, version gcc (Ubuntu/Linaro 4.8.1-8ubuntu1) 4.8.1
       SKCFLAGS : -fomit-frame-pointer -mfpmath=sse -O2 -msse2 -fPIC -m64
    success!
    removing: _configtest.c _configtest.o _configtest
      FOUND:
        libraries = ['f77blas', 'cblas', 'atlas']
        library_dirs = ['/usr/lib/atlas-base']
        language = c
        define_macros = [('ATLAS_INFO', '"\\"3.10.1\\""')]
        include_dirs = ['/usr/include/atlas']
    
      FOUND:
        libraries = ['f77blas', 'cblas', 'atlas']
        library_dirs = ['/usr/lib/atlas-base']
        language = c
        define_macros = [('ATLAS_INFO', '"\\"3.10.1\\""')]
        include_dirs = ['/usr/include/atlas']
    
    Running from scipy source directory.
    lapack_opt_info:
    lapack_mkl_info:
    mkl_info:
      libraries mkl,vml,guide not found in ['/usr/local/lib', '/usr/lib', '/usr/lib/x86_64-linux-gnu']
      NOT AVAILABLE
    
      NOT AVAILABLE
    
    atlas_threads_info:
    Setting PTATLAS=ATLAS
      libraries ptf77blas,ptcblas,atlas not found in /usr/local/lib
      libraries lapack_atlas not found in /usr/local/lib
      libraries ptf77blas,ptcblas,atlas not found in /usr/lib/atlas-base
      libraries ptf77blas,ptcblas,atlas not found in /usr/lib
      libraries ptf77blas,ptcblas,atlas not found in /usr/lib/x86_64-linux-gnu
      libraries lapack_atlas not found in /usr/lib/x86_64-linux-gnu
    numpy.distutils.system_info.atlas_threads_info
      NOT AVAILABLE
    
    atlas_info:
      libraries f77blas,cblas,atlas not found in /usr/local/lib
      libraries lapack_atlas not found in /usr/local/lib
    numpy.distutils.system_info.atlas_info
      FOUND:
        libraries = ['lapack', 'f77blas', 'cblas', 'atlas']
        library_dirs = ['/usr/lib/atlas-base/atlas', '/usr/lib/atlas-base']
        language = f77
        define_macros = [('ATLAS_INFO', '"\\"3.10.1\\""')]
        include_dirs = ['/usr/include/atlas']
    
      FOUND:
        libraries = ['lapack', 'f77blas', 'cblas', 'atlas']
        library_dirs = ['/usr/lib/atlas-base/atlas', '/usr/lib/atlas-base']
        language = f77
        define_macros = [('ATLAS_INFO', '"\\"3.10.1\\""')]
        include_dirs = ['/usr/include/atlas']
    
    ATLAS version: 3.10.1
    ATLAS version: 3.10.1
    ATLAS version: 3.10.1
    Splitting linalg.interpolative Fortran source files
    umfpack_info:
    amd_info:
      FOUND:
        libraries = ['amd']
        library_dirs = ['/usr/lib']
        swig_opts = ['-I/usr/include/suitesparse']
        define_macros = [('SCIPY_AMD_H', None)]
        include_dirs = ['/usr/include/suitesparse']
    
      FOUND:
        libraries = ['umfpack', 'amd']
        library_dirs = ['/usr/lib']
        swig_opts = ['-I/usr/include/suitesparse', '-I/usr/include/suitesparse']
        define_macros = [('SCIPY_UMFPACK_H', None), ('SCIPY_AMD_H', None)]
        include_dirs = ['/usr/include/suitesparse']
    
    unifing config_cc, config, build_clib, build_ext, build commands --compiler options
    unifing config_fc, config, build_clib, build_ext, build commands --fcompiler options
    build_src
    building py_modules sources
    building library "dfftpack" sources
    building library "fftpack" sources
    building library "linpack_lite" sources
    building library "mach" sources
    building library "quadpack" sources
    building library "odepack" sources
    building library "dop" sources
    building library "fitpack" sources
    building library "odrpack" sources
    building library "minpack" sources
    building library "rootfind" sources
    building library "superlu_src" sources
    building library "arpack_scipy" sources
    building library "sc_c_misc" sources
    building library "sc_cephes" sources
    building library "sc_mach" sources
    building library "sc_amos" sources
    building library "sc_cdf" sources
    building library "sc_specfun" sources
    building library "statlib" sources
    building extension "scipy.cluster._vq" sources
    building extension "scipy.cluster._hierarchy_wrap" sources
    building extension "scipy.fftpack._fftpack" sources
    conv_template:> build/src.linux-x86_64-2.7/scipy/fftpack/src/dct.c
    conv_template:> build/src.linux-x86_64-2.7/scipy/fftpack/src/dst.c
    f2py options: []
    f2py: scipy/fftpack/fftpack.pyf
    Reading fortran codes...
    	Reading file 'scipy/fftpack/fftpack.pyf' (format:free)
    Line #86 in scipy/fftpack/fftpack.pyf:"       /* Single precision version */"
    	crackline:2: No pattern for line
    Post-processing...
    	Block: _fftpack
    			Block: zfft
    			Block: drfft
    			Block: zrfft
    			Block: zfftnd
    			Block: destroy_zfft_cache
    			Block: destroy_zfftnd_cache
    			Block: destroy_drfft_cache
    			Block: cfft
    			Block: rfft
    			Block: crfft
    			Block: cfftnd
    			Block: destroy_cfft_cache
    			Block: destroy_cfftnd_cache
    			Block: destroy_rfft_cache
    			Block: ddct1
    			Block: ddct2
    			Block: ddct3
    			Block: dct1
    			Block: dct2
    			Block: dct3
    			Block: destroy_ddct2_cache
    			Block: destroy_ddct1_cache
    			Block: destroy_dct2_cache
    			Block: destroy_dct1_cache
    			Block: ddst1
    			Block: ddst2
    			Block: ddst3
    			Block: dst1
    			Block: dst2
    			Block: dst3
    			Block: destroy_ddst2_cache
    			Block: destroy_ddst1_cache
    			Block: destroy_dst2_cache
    			Block: destroy_dst1_cache
    Post-processing (stage 2)...
    Building modules...
    	Building module "_fftpack"...
    		Constructing wrapper function "zfft"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = zfft(x,[n,direction,normalize,overwrite_x])
    		Constructing wrapper function "drfft"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = drfft(x,[n,direction,normalize,overwrite_x])
    		Constructing wrapper function "zrfft"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = zrfft(x,[n,direction,normalize,overwrite_x])
    		Constructing wrapper function "zfftnd"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = zfftnd(x,[s,direction,normalize,overwrite_x])
    		Constructing wrapper function "destroy_zfft_cache"...
    		  destroy_zfft_cache()
    		Constructing wrapper function "destroy_zfftnd_cache"...
    		  destroy_zfftnd_cache()
    		Constructing wrapper function "destroy_drfft_cache"...
    		  destroy_drfft_cache()
    		Constructing wrapper function "cfft"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = cfft(x,[n,direction,normalize,overwrite_x])
    		Constructing wrapper function "rfft"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = rfft(x,[n,direction,normalize,overwrite_x])
    		Constructing wrapper function "crfft"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = crfft(x,[n,direction,normalize,overwrite_x])
    		Constructing wrapper function "cfftnd"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = cfftnd(x,[s,direction,normalize,overwrite_x])
    		Constructing wrapper function "destroy_cfft_cache"...
    		  destroy_cfft_cache()
    		Constructing wrapper function "destroy_cfftnd_cache"...
    		  destroy_cfftnd_cache()
    		Constructing wrapper function "destroy_rfft_cache"...
    		  destroy_rfft_cache()
    		Constructing wrapper function "ddct1"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = ddct1(x,[n,normalize,overwrite_x])
    		Constructing wrapper function "ddct2"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = ddct2(x,[n,normalize,overwrite_x])
    		Constructing wrapper function "ddct3"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = ddct3(x,[n,normalize,overwrite_x])
    		Constructing wrapper function "dct1"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = dct1(x,[n,normalize,overwrite_x])
    		Constructing wrapper function "dct2"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = dct2(x,[n,normalize,overwrite_x])
    		Constructing wrapper function "dct3"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = dct3(x,[n,normalize,overwrite_x])
    		Constructing wrapper function "destroy_ddct2_cache"...
    		  destroy_ddct2_cache()
    		Constructing wrapper function "destroy_ddct1_cache"...
    		  destroy_ddct1_cache()
    		Constructing wrapper function "destroy_dct2_cache"...
    		  destroy_dct2_cache()
    		Constructing wrapper function "destroy_dct1_cache"...
    		  destroy_dct1_cache()
    		Constructing wrapper function "ddst1"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = ddst1(x,[n,normalize,overwrite_x])
    		Constructing wrapper function "ddst2"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = ddst2(x,[n,normalize,overwrite_x])
    		Constructing wrapper function "ddst3"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = ddst3(x,[n,normalize,overwrite_x])
    		Constructing wrapper function "dst1"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = dst1(x,[n,normalize,overwrite_x])
    		Constructing wrapper function "dst2"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = dst2(x,[n,normalize,overwrite_x])
    		Constructing wrapper function "dst3"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = dst3(x,[n,normalize,overwrite_x])
    		Constructing wrapper function "destroy_ddst2_cache"...
    		  destroy_ddst2_cache()
    		Constructing wrapper function "destroy_ddst1_cache"...
    		  destroy_ddst1_cache()
    		Constructing wrapper function "destroy_dst2_cache"...
    		  destroy_dst2_cache()
    		Constructing wrapper function "destroy_dst1_cache"...
    		  destroy_dst1_cache()
    	Wrote C/API module "_fftpack" to file "build/src.linux-x86_64-2.7/scipy/fftpack/_fftpackmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.fftpack.convolve" sources
    f2py options: []
    f2py: scipy/fftpack/convolve.pyf
    Reading fortran codes...
    	Reading file 'scipy/fftpack/convolve.pyf' (format:free)
    Post-processing...
    	Block: convolve__user__routines
    			Block: kernel_func
    	Block: convolve
    			Block: init_convolution_kernel
    In: scipy/fftpack/convolve.pyf:convolve:unknown_interface:init_convolution_kernel
    get_useparameters: no module convolve__user__routines info used by init_convolution_kernel
    			Block: destroy_convolve_cache
    			Block: convolve
    			Block: convolve_z
    Post-processing (stage 2)...
    Building modules...
    	Constructing call-back function "cb_kernel_func_in_convolve__user__routines"
    	  def kernel_func(k): return kernel_func
    	Building module "convolve"...
    		Constructing wrapper function "init_convolution_kernel"...
    		  omega = init_convolution_kernel(n,kernel_func,[d,zero_nyquist,kernel_func_extra_args])
    		Constructing wrapper function "destroy_convolve_cache"...
    		  destroy_convolve_cache()
    		Constructing wrapper function "convolve"...
    		  y = convolve(x,omega,[swap_real_imag,overwrite_x])
    		Constructing wrapper function "convolve_z"...
    		  y = convolve_z(x,omega_real,omega_imag,[overwrite_x])
    	Wrote C/API module "convolve" to file "build/src.linux-x86_64-2.7/scipy/fftpack/convolvemodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.integrate._quadpack" sources
    building extension "scipy.integrate._odepack" sources
    building extension "scipy.integrate.vode" sources
    f2py options: []
    f2py: scipy/integrate/vode.pyf
    Reading fortran codes...
    	Reading file 'scipy/integrate/vode.pyf' (format:free)
    Post-processing...
    	Block: dvode__user__routines
    		Block: dvode_user_interface
    			Block: f
    			Block: jac
    	Block: zvode__user__routines
    		Block: zvode_user_interface
    			Block: f
    			Block: jac
    	Block: vode
    			Block: dvode
    In: scipy/integrate/vode.pyf:vode:unknown_interface:dvode
    get_useparameters: no module dvode__user__routines info used by dvode
    			Block: zvode
    In: scipy/integrate/vode.pyf:vode:unknown_interface:zvode
    get_useparameters: no module zvode__user__routines info used by zvode
    Post-processing (stage 2)...
    Building modules...
    	Constructing call-back function "cb_f_in_dvode__user__routines"
    	  def f(t,y): return ydot
    	Constructing call-back function "cb_jac_in_dvode__user__routines"
    	  def jac(t,y): return jac
    	Constructing call-back function "cb_f_in_zvode__user__routines"
    	  def f(t,y): return ydot
    	Constructing call-back function "cb_jac_in_zvode__user__routines"
    	  def jac(t,y): return jac
    	Building module "vode"...
    		Constructing wrapper function "dvode"...
    warning: callstatement is defined without callprotoargument
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y,t,istate = dvode(f,jac,y,t,tout,rtol,atol,itask,istate,rwork,iwork,mf,[f_extra_args,jac_extra_args,overwrite_y])
    		Constructing wrapper function "zvode"...
    warning: callstatement is defined without callprotoargument
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y,t,istate = zvode(f,jac,y,t,tout,rtol,atol,itask,istate,zwork,rwork,iwork,mf,[f_extra_args,jac_extra_args,overwrite_y])
    	Wrote C/API module "vode" to file "build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.integrate.lsoda" sources
    f2py options: []
    f2py: scipy/integrate/lsoda.pyf
    Reading fortran codes...
    	Reading file 'scipy/integrate/lsoda.pyf' (format:free)
    Post-processing...
    	Block: lsoda__user__routines
    		Block: lsoda_user_interface
    			Block: f
    			Block: jac
    	Block: lsoda
    			Block: lsoda
    In: scipy/integrate/lsoda.pyf:lsoda:unknown_interface:lsoda
    get_useparameters: no module lsoda__user__routines info used by lsoda
    Post-processing (stage 2)...
    Building modules...
    	Constructing call-back function "cb_f_in_lsoda__user__routines"
    	  def f(t,y): return ydot
    	Constructing call-back function "cb_jac_in_lsoda__user__routines"
    	  def jac(t,y): return jac
    	Building module "lsoda"...
    		Constructing wrapper function "lsoda"...
    warning: callstatement is defined without callprotoargument
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y,t,istate = lsoda(f,y,t,tout,rtol,atol,itask,istate,rwork,iwork,jac,jt,[f_extra_args,overwrite_y,jac_extra_args])
    	Wrote C/API module "lsoda" to file "build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.integrate._dop" sources
    f2py options: []
    f2py: scipy/integrate/dop.pyf
    Reading fortran codes...
    	Reading file 'scipy/integrate/dop.pyf' (format:free)
    Post-processing...
    	Block: __user__routines
    			Block: fcn
    			Block: solout
    	Block: _dop
    			Block: dopri5
    In: scipy/integrate/dop.pyf:_dop:unknown_interface:dopri5
    get_useparameters: no module __user__routines info used by dopri5
    			Block: dop853
    In: scipy/integrate/dop.pyf:_dop:unknown_interface:dop853
    get_useparameters: no module __user__routines info used by dop853
    Post-processing (stage 2)...
    Building modules...
    	Constructing call-back function "cb_fcn_in___user__routines"
    	  def fcn(x,y): return f
    	Constructing call-back function "cb_solout_in___user__routines"
    	  def solout(nr,xold,x,y,con,icomp,[nd]): return irtn
    	Building module "_dop"...
    		Constructing wrapper function "dopri5"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y,iwork,idid = dopri5(fcn,x,y,xend,rtol,atol,solout,iout,work,iwork,[fcn_extra_args,overwrite_y,solout_extra_args])
    		Constructing wrapper function "dop853"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y,iwork,idid = dop853(fcn,x,y,xend,rtol,atol,solout,iout,work,iwork,[fcn_extra_args,overwrite_y,solout_extra_args])
    	Wrote C/API module "_dop" to file "build/src.linux-x86_64-2.7/scipy/integrate/_dopmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.interpolate.interpnd" sources
    building extension "scipy.interpolate._fitpack" sources
    building extension "scipy.interpolate.dfitpack" sources
    f2py options: []
    f2py: scipy/interpolate/src/fitpack.pyf
    Reading fortran codes...
    	Reading file 'scipy/interpolate/src/fitpack.pyf' (format:free)
    Post-processing...
    	Block: dfitpack
    			Block: splev
    			Block: splder
    			Block: splint
    			Block: sproot
    			Block: spalde
    			Block: curfit
    			Block: percur
    			Block: parcur
    			Block: fpcurf0
    			Block: fpcurf1
    			Block: fpcurfm1
    			Block: bispev
    			Block: bispeu
    			Block: surfit_smth
    			Block: surfit_lsq
    			Block: spherfit_smth
    			Block: spherfit_lsq
    			Block: regrid_smth
    			Block: regrid_smth_spher
    			Block: dblint
    Post-processing (stage 2)...
    Building modules...
    	Building module "dfitpack"...
    		Constructing wrapper function "splev"...
    		  y = splev(t,c,k,x,[e])
    		Constructing wrapper function "splder"...
    		  y = splder(t,c,k,x,[nu,e])
    		Creating wrapper for Fortran function "splint"("splint")...
    		Constructing wrapper function "splint"...
    		  splint = splint(t,c,k,a,b)
    		Constructing wrapper function "sproot"...
    		  zero,m,ier = sproot(t,c,[mest])
    		Constructing wrapper function "spalde"...
    		  d,ier = spalde(t,c,k,x)
    		Constructing wrapper function "curfit"...
    		  n,c,fp,ier = curfit(iopt,x,y,w,t,wrk,iwrk,[xb,xe,k,s])
    		Constructing wrapper function "percur"...
    		  n,c,fp,ier = percur(iopt,x,y,w,t,wrk,iwrk,[k,s])
    		Constructing wrapper function "parcur"...
    		  n,c,fp,ier = parcur(iopt,ipar,idim,u,x,w,ub,ue,t,wrk,iwrk,[k,s])
    		Constructing wrapper function "fpcurf0"...
    		  x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier = fpcurf0(x,y,k,[w,xb,xe,s,nest])
    		Constructing wrapper function "fpcurf1"...
    		  x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier = fpcurf1(x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier,[overwrite_x,overwrite_y,overwrite_w,overwrite_t,overwrite_c,overwrite_fpint,overwrite_nrdata])
    		Constructing wrapper function "fpcurfm1"...
    		  x,y,w,xb,xe,k,s,n,t,c,fp,fpint,nrdata,ier = fpcurfm1(x,y,k,t,[w,xb,xe,overwrite_t])
    		Constructing wrapper function "bispev"...
    		  z,ier = bispev(tx,ty,c,kx,ky,x,y)
    		Constructing wrapper function "bispeu"...
    		  z,ier = bispeu(tx,ty,c,kx,ky,x,y)
    		Constructing wrapper function "surfit_smth"...
    		  nx,tx,ny,ty,c,fp,wrk1,ier = surfit_smth(x,y,z,[w,xb,xe,yb,ye,kx,ky,s,nxest,nyest,eps,lwrk2])
    		Constructing wrapper function "surfit_lsq"...
    		  tx,ty,c,fp,ier = surfit_lsq(x,y,z,tx,ty,[w,xb,xe,yb,ye,kx,ky,eps,lwrk2,overwrite_tx,overwrite_ty])
    		Constructing wrapper function "spherfit_smth"...
    		  nt,tt,np,tp,c,fp,ier = spherfit_smth(teta,phi,r,[w,s,eps])
    		Constructing wrapper function "spherfit_lsq"...
    		  tt,tp,c,fp,ier = spherfit_lsq(teta,phi,r,tt,tp,[w,eps,overwrite_tt,overwrite_tp])
    		Constructing wrapper function "regrid_smth"...
    		  nx,tx,ny,ty,c,fp,ier = regrid_smth(x,y,z,[xb,xe,yb,ye,kx,ky,s])
    		Constructing wrapper function "regrid_smth_spher"...
    		  nu,tu,nv,tv,c,fp,ier = regrid_smth_spher(iopt,ider,u,v,r,[r0,r1,s])
    		Creating wrapper for Fortran function "dblint"("dblint")...
    		Constructing wrapper function "dblint"...
    		  dblint = dblint(tx,ty,c,kx,ky,xb,xe,yb,ye)
    	Wrote C/API module "dfitpack" to file "build/src.linux-x86_64-2.7/scipy/interpolate/src/dfitpackmodule.c"
    	Fortran 77 wrappers are saved to "build/src.linux-x86_64-2.7/scipy/interpolate/src/dfitpack-f2pywrappers.f"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
      adding 'build/src.linux-x86_64-2.7/scipy/interpolate/src/dfitpack-f2pywrappers.f' to sources.
    building extension "scipy.interpolate._interpolate" sources
    building extension "scipy.io.matlab.streams" sources
    building extension "scipy.io.matlab.mio_utils" sources
    building extension "scipy.io.matlab.mio5_utils" sources
    building extension "scipy.lib.blas.fblas" sources
    from_template:> build/src.linux-x86_64-2.7/scipy/lib/blas/fblas.pyf
    Including file scipy/lib/blas/fblas_l1.pyf.src
    Including file scipy/lib/blas/fblas_l2.pyf.src
    Including file scipy/lib/blas/fblas_l3.pyf.src
    Mismatch in number of replacements (base <prefix=s,d,c,z>) for <__l1=->. Ignoring.
    f2py options: []
    f2py: build/src.linux-x86_64-2.7/scipy/lib/blas/fblas.pyf
    Reading fortran codes...
    	Reading file 'build/src.linux-x86_64-2.7/scipy/lib/blas/fblas.pyf' (format:free)
    Post-processing...
    	Block: fblas
    			Block: srotg
    			Block: drotg
    			Block: crotg
    			Block: zrotg
    			Block: srotmg
    			Block: drotmg
    			Block: srot
    			Block: drot
    			Block: csrot
    			Block: zdrot
    			Block: srotm
    			Block: drotm
    			Block: sswap
    			Block: dswap
    			Block: cswap
    			Block: zswap
    			Block: sscal
    			Block: dscal
    			Block: cscal
    			Block: zscal
    			Block: csscal
    			Block: zdscal
    			Block: scopy
    			Block: dcopy
    			Block: ccopy
    			Block: zcopy
    			Block: saxpy
    			Block: daxpy
    			Block: caxpy
    			Block: zaxpy
    			Block: sdot
    			Block: ddot
    			Block: cdotu
    			Block: zdotu
    			Block: cdotc
    			Block: zdotc
    			Block: snrm2
    			Block: scnrm2
    			Block: dnrm2
    			Block: dznrm2
    			Block: sasum
    			Block: scasum
    			Block: dasum
    			Block: dzasum
    			Block: isamax
    			Block: idamax
    			Block: icamax
    			Block: izamax
    			Block: sgemv
    			Block: dgemv
    			Block: cgemv
    			Block: zgemv
    			Block: ssymv
    			Block: dsymv
    			Block: chemv
    			Block: zhemv
    			Block: strmv
    			Block: dtrmv
    			Block: ctrmv
    			Block: ztrmv
    			Block: sger
    			Block: dger
    			Block: cgeru
    			Block: zgeru
    			Block: cgerc
    			Block: zgerc
    			Block: sgemm
    			Block: dgemm
    			Block: cgemm
    			Block: zgemm
    Post-processing (stage 2)...
    Building modules...
    	Building module "fblas"...
    		Constructing wrapper function "srotg"...
    		  c,s = srotg(a,b)
    		Constructing wrapper function "drotg"...
    		  c,s = drotg(a,b)
    		Constructing wrapper function "crotg"...
    		  c,s = crotg(a,b)
    		Constructing wrapper function "zrotg"...
    		  c,s = zrotg(a,b)
    		Constructing wrapper function "srotmg"...
    		  param = srotmg(d1,d2,x1,y1)
    		Constructing wrapper function "drotmg"...
    		  param = drotmg(d1,d2,x1,y1)
    		Constructing wrapper function "srot"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = srot(x,y,c,s,[n,offx,incx,offy,incy,overwrite_x,overwrite_y])
    		Constructing wrapper function "drot"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = drot(x,y,c,s,[n,offx,incx,offy,incy,overwrite_x,overwrite_y])
    		Constructing wrapper function "csrot"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = csrot(x,y,c,s,[n,offx,incx,offy,incy,overwrite_x,overwrite_y])
    		Constructing wrapper function "zdrot"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = zdrot(x,y,c,s,[n,offx,incx,offy,incy,overwrite_x,overwrite_y])
    		Constructing wrapper function "srotm"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = srotm(x,y,param,[n,offx,incx,offy,incy,overwrite_x,overwrite_y])
    		Constructing wrapper function "drotm"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = drotm(x,y,param,[n,offx,incx,offy,incy,overwrite_x,overwrite_y])
    		Constructing wrapper function "sswap"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = sswap(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "dswap"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = dswap(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "cswap"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = cswap(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "zswap"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = zswap(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "sscal"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = sscal(a,x,[n,offx,incx])
    		Constructing wrapper function "dscal"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = dscal(a,x,[n,offx,incx])
    		Constructing wrapper function "cscal"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = cscal(a,x,[n,offx,incx])
    		Constructing wrapper function "zscal"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = zscal(a,x,[n,offx,incx])
    		Constructing wrapper function "csscal"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = csscal(a,x,[n,offx,incx,overwrite_x])
    		Constructing wrapper function "zdscal"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = zdscal(a,x,[n,offx,incx,overwrite_x])
    		Constructing wrapper function "scopy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = scopy(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "dcopy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = dcopy(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "ccopy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = ccopy(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "zcopy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = zcopy(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "saxpy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  z = saxpy(x,y,[n,a,offx,incx,offy,incy])
    		Constructing wrapper function "daxpy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  z = daxpy(x,y,[n,a,offx,incx,offy,incy])
    		Constructing wrapper function "caxpy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  z = caxpy(x,y,[n,a,offx,incx,offy,incy])
    		Constructing wrapper function "zaxpy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  z = zaxpy(x,y,[n,a,offx,incx,offy,incy])
    		Creating wrapper for Fortran function "sdot"("wsdot")...
    		Constructing wrapper function "sdot"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  xy = sdot(x,y,[n,offx,incx,offy,incy])
    		Creating wrapper for Fortran function "ddot"("ddot")...
    		Constructing wrapper function "ddot"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  xy = ddot(x,y,[n,offx,incx,offy,incy])
    		Creating wrapper for Fortran function "cdotu"("wcdotu")...
    		Constructing wrapper function "cdotu"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  xy = cdotu(x,y,[n,offx,incx,offy,incy])
    		Creating wrapper for Fortran function "zdotu"("wzdotu")...
    		Constructing wrapper function "zdotu"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  xy = zdotu(x,y,[n,offx,incx,offy,incy])
    		Creating wrapper for Fortran function "cdotc"("wcdotc")...
    		Constructing wrapper function "cdotc"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  xy = cdotc(x,y,[n,offx,incx,offy,incy])
    		Creating wrapper for Fortran function "zdotc"("wzdotc")...
    		Constructing wrapper function "zdotc"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  xy = zdotc(x,y,[n,offx,incx,offy,incy])
    		Creating wrapper for Fortran function "snrm2"("wsnrm2")...
    		Constructing wrapper function "snrm2"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  n2 = snrm2(x,[n,offx,incx])
    		Creating wrapper for Fortran function "scnrm2"("wscnrm2")...
    		Constructing wrapper function "scnrm2"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  n2 = scnrm2(x,[n,offx,incx])
    		Creating wrapper for Fortran function "dnrm2"("dnrm2")...
    		Constructing wrapper function "dnrm2"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  n2 = dnrm2(x,[n,offx,incx])
    		Creating wrapper for Fortran function "dznrm2"("dznrm2")...
    		Constructing wrapper function "dznrm2"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  n2 = dznrm2(x,[n,offx,incx])
    		Creating wrapper for Fortran function "sasum"("wsasum")...
    		Constructing wrapper function "sasum"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  s = sasum(x,[n,offx,incx])
    		Creating wrapper for Fortran function "scasum"("wscasum")...
    		Constructing wrapper function "scasum"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  s = scasum(x,[n,offx,incx])
    		Creating wrapper for Fortran function "dasum"("dasum")...
    		Constructing wrapper function "dasum"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  s = dasum(x,[n,offx,incx])
    		Creating wrapper for Fortran function "dzasum"("dzasum")...
    		Constructing wrapper function "dzasum"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  s = dzasum(x,[n,offx,incx])
    		Constructing wrapper function "isamax"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  k = isamax(x,[n,offx,incx])
    		Constructing wrapper function "idamax"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  k = idamax(x,[n,offx,incx])
    		Constructing wrapper function "icamax"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  k = icamax(x,[n,offx,incx])
    		Constructing wrapper function "izamax"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  k = izamax(x,[n,offx,incx])
    		Constructing wrapper function "sgemv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = sgemv(alpha,a,x,[beta,y,offx,incx,offy,incy,trans,overwrite_y])
    		Constructing wrapper function "dgemv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = dgemv(alpha,a,x,[beta,y,offx,incx,offy,incy,trans,overwrite_y])
    		Constructing wrapper function "cgemv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = cgemv(alpha,a,x,[beta,y,offx,incx,offy,incy,trans,overwrite_y])
    		Constructing wrapper function "zgemv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = zgemv(alpha,a,x,[beta,y,offx,incx,offy,incy,trans,overwrite_y])
    		Constructing wrapper function "ssymv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = ssymv(alpha,a,x,[beta,y,offx,incx,offy,incy,lower,overwrite_y])
    		Constructing wrapper function "dsymv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = dsymv(alpha,a,x,[beta,y,offx,incx,offy,incy,lower,overwrite_y])
    		Constructing wrapper function "chemv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = chemv(alpha,a,x,[beta,y,offx,incx,offy,incy,lower,overwrite_y])
    		Constructing wrapper function "zhemv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = zhemv(alpha,a,x,[beta,y,offx,incx,offy,incy,lower,overwrite_y])
    		Constructing wrapper function "strmv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = strmv(a,x,[offx,incx,lower,trans,unitdiag,overwrite_x])
    		Constructing wrapper function "dtrmv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = dtrmv(a,x,[offx,incx,lower,trans,unitdiag,overwrite_x])
    		Constructing wrapper function "ctrmv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = ctrmv(a,x,[offx,incx,lower,trans,unitdiag,overwrite_x])
    		Constructing wrapper function "ztrmv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = ztrmv(a,x,[offx,incx,lower,trans,unitdiag,overwrite_x])
    		Constructing wrapper function "sger"...
    		  a = sger(alpha,x,y,[incx,incy,a,overwrite_x,overwrite_y,overwrite_a])
    		Constructing wrapper function "dger"...
    		  a = dger(alpha,x,y,[incx,incy,a,overwrite_x,overwrite_y,overwrite_a])
    		Constructing wrapper function "cgeru"...
    		  a = cgeru(alpha,x,y,[incx,incy,a,overwrite_x,overwrite_y,overwrite_a])
    		Constructing wrapper function "zgeru"...
    		  a = zgeru(alpha,x,y,[incx,incy,a,overwrite_x,overwrite_y,overwrite_a])
    		Constructing wrapper function "cgerc"...
    		  a = cgerc(alpha,x,y,[incx,incy,a,overwrite_x,overwrite_y,overwrite_a])
    		Constructing wrapper function "zgerc"...
    		  a = zgerc(alpha,x,y,[incx,incy,a,overwrite_x,overwrite_y,overwrite_a])
    		Constructing wrapper function "sgemm"...
    		  c = sgemm(alpha,a,b,[beta,c,trans_a,trans_b,overwrite_c])
    		Constructing wrapper function "dgemm"...
    		  c = dgemm(alpha,a,b,[beta,c,trans_a,trans_b,overwrite_c])
    		Constructing wrapper function "cgemm"...
    		  c = cgemm(alpha,a,b,[beta,c,trans_a,trans_b,overwrite_c])
    		Constructing wrapper function "zgemm"...
    		  c = zgemm(alpha,a,b,[beta,c,trans_a,trans_b,overwrite_c])
    	Wrote C/API module "fblas" to file "build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/blas/fblasmodule.c"
    	Fortran 77 wrappers are saved to "build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/blas/fblas-f2pywrappers.f"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
      adding 'build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/blas/fblas-f2pywrappers.f' to sources.
    building extension "scipy.lib.blas.cblas" sources
      adding 'scipy/lib/blas/cblas.pyf.src' to sources.
    from_template:> build/src.linux-x86_64-2.7/scipy/lib/blas/cblas.pyf
    Including file scipy/lib/blas/cblas_l1.pyf.src
    f2py options: []
    f2py: build/src.linux-x86_64-2.7/scipy/lib/blas/cblas.pyf
    Reading fortran codes...
    	Reading file 'build/src.linux-x86_64-2.7/scipy/lib/blas/cblas.pyf' (format:free)
    Line #33 in build/src.linux-x86_64-2.7/scipy/lib/blas/cblas.pyf:"  intent(c)"
    	All arguments will have attribute intent(c)
    Line #57 in build/src.linux-x86_64-2.7/scipy/lib/blas/cblas.pyf:"  intent(c)"
    	All arguments will have attribute intent(c)
    Line #81 in build/src.linux-x86_64-2.7/scipy/lib/blas/cblas.pyf:"  intent(c)"
    	All arguments will have attribute intent(c)
    Line #105 in build/src.linux-x86_64-2.7/scipy/lib/blas/cblas.pyf:"  intent(c)"
    	All arguments will have attribute intent(c)
    Post-processing...
    	Block: cblas
    			Block: saxpy
    			Block: daxpy
    			Block: caxpy
    			Block: zaxpy
    Post-processing (stage 2)...
    Building modules...
    	Building module "cblas"...
    		Constructing wrapper function "saxpy"...
    		  z = saxpy(x,y,[n,a,incx,incy,overwrite_y])
    		Constructing wrapper function "daxpy"...
    		  z = daxpy(x,y,[n,a,incx,incy,overwrite_y])
    		Constructing wrapper function "caxpy"...
    		  z = caxpy(x,y,[n,a,incx,incy,overwrite_y])
    		Constructing wrapper function "zaxpy"...
    		  z = zaxpy(x,y,[n,a,incx,incy,overwrite_y])
    	Wrote C/API module "cblas" to file "build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/blas/cblasmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.lib.lapack.flapack" sources
    from_template:> build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf
    Including file scipy/lib/lapack/flapack_user.pyf.src
    Including file scipy/lib/lapack/flapack_le.pyf.src
    Including file scipy/lib/lapack/flapack_lls.pyf.src
    Including file scipy/lib/lapack/flapack_esv.pyf.src
    Including file scipy/lib/lapack/flapack_gesv.pyf.src
    Including file scipy/lib/lapack/flapack_lec.pyf.src
    Including file scipy/lib/lapack/flapack_llsc.pyf.src
    Including file scipy/lib/lapack/flapack_sevc.pyf.src
    Including file scipy/lib/lapack/flapack_evc.pyf.src
    Including file scipy/lib/lapack/flapack_svdc.pyf.src
    Including file scipy/lib/lapack/flapack_gsevc.pyf.src
    Including file scipy/lib/lapack/flapack_gevc.pyf.src
    Including file scipy/lib/lapack/flapack_aux.pyf.src
    f2py options: []
    f2py: build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf
    Reading fortran codes...
    	Reading file 'build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf' (format:free)
    Line #1590 in build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:"  3*n-1"
    	crackline:3: No pattern for line
    Line #1612 in build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:"  3*n-1"
    	crackline:3: No pattern for line
    Line #1634 in build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:"  2*n-1"
    	crackline:3: No pattern for line
    Line #1656 in build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:"  2*n-1"
    	crackline:3: No pattern for line
    Line #1679 in build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:"  (compute_v?1+6*n+2*n*n:2*n+1)"
    	crackline:3: No pattern for line
    Line #1704 in build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:"  (compute_v?1+6*n+2*n*n:2*n+1)"
    	crackline:3: No pattern for line
    Line #1729 in build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:"  (compute_v?2*n+n*n:n+1)"
    	crackline:3: No pattern for line
    Line #1754 in build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:"  (compute_v?2*n+n*n:n+1)"
    	crackline:3: No pattern for line
    Line #2647 in build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:"     n"
    	crackline:3: No pattern for line
    Line #2668 in build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:"     n"
    	crackline:3: No pattern for line
    Line #2689 in build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:"     n"
    	crackline:3: No pattern for line
    Line #2710 in build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:"     n"
    	crackline:3: No pattern for line
    Post-processing...
    	Block: flapack
    			Block: gees__user__routines
    				Block: gees_user_interface
    					Block: sselect
    					Block: dselect
    					Block: cselect
    					Block: zselect
    			Block: sgesv
    			Block: dgesv
    			Block: cgesv
    			Block: zgesv
    			Block: sgbsv
    			Block: dgbsv
    			Block: cgbsv
    			Block: zgbsv
    			Block: sposv
    			Block: dposv
    			Block: cposv
    			Block: zposv
    			Block: sgelss
    			Block: dgelss
    			Block: cgelss
    			Block: zgelss
    			Block: ssyev
    			Block: dsyev
    			Block: cheev
    			Block: zheev
    			Block: ssyevd
    			Block: dsyevd
    			Block: cheevd
    			Block: zheevd
    			Block: ssyevr
    			Block: dsyevr
    			Block: cheevr
    			Block: zheevr
    			Block: sgees
    In: build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:flapack:unknown_interface:sgees
    get_useparameters: no module gees__user__routines info used by sgees
    			Block: dgees
    In: build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:flapack:unknown_interface:dgees
    get_useparameters: no module gees__user__routines info used by dgees
    			Block: cgees
    In: build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:flapack:unknown_interface:cgees
    get_useparameters: no module gees__user__routines info used by cgees
    			Block: zgees
    In: build/src.linux-x86_64-2.7/scipy/lib/lapack/flapack.pyf:flapack:unknown_interface:zgees
    get_useparameters: no module gees__user__routines info used by zgees
    			Block: sgeev
    			Block: dgeev
    			Block: cgeev
    			Block: zgeev
    			Block: sgesdd
    			Block: dgesdd
    			Block: cgesdd
    			Block: zgesdd
    			Block: ssygv
    			Block: dsygv
    			Block: chegv
    			Block: zhegv
    			Block: ssygvd
    			Block: dsygvd
    			Block: chegvd
    			Block: zhegvd
    			Block: sggev
    			Block: dggev
    			Block: cggev
    			Block: zggev
    			Block: sgetrf
    			Block: dgetrf
    			Block: cgetrf
    			Block: zgetrf
    			Block: spotrf
    			Block: dpotrf
    			Block: cpotrf
    			Block: zpotrf
    			Block: sgetrs
    			Block: dgetrs
    			Block: cgetrs
    			Block: zgetrs
    			Block: spotrs
    			Block: dpotrs
    			Block: cpotrs
    			Block: zpotrs
    			Block: sgetri
    			Block: dgetri
    			Block: cgetri
    			Block: zgetri
    			Block: spotri
    			Block: dpotri
    			Block: cpotri
    			Block: zpotri
    			Block: strtri
    			Block: dtrtri
    			Block: ctrtri
    			Block: ztrtri
    			Block: sgeqrf
    			Block: dgeqrf
    			Block: cgeqrf
    			Block: zgeqrf
    			Block: sorgqr
    			Block: dorgqr
    			Block: cungqr
    			Block: zungqr
    			Block: sgehrd
    			Block: dgehrd
    			Block: cgehrd
    			Block: zgehrd
    			Block: sgebal
    			Block: dgebal
    			Block: cgebal
    			Block: zgebal
    			Block: slauum
    			Block: dlauum
    			Block: clauum
    			Block: zlauum
    			Block: slaswp
    			Block: dlaswp
    			Block: claswp
    			Block: zlaswp
    Post-processing (stage 2)...
    Building modules...
    	Constructing call-back function "cb_sselect_in_gees__user__routines"
    	  def sselect(arg1,arg2): return sselect
    	Constructing call-back function "cb_dselect_in_gees__user__routines"
    	  def dselect(arg1,arg2): return dselect
    	Constructing call-back function "cb_cselect_in_gees__user__routines"
    	  def cselect(arg): return cselect
    	Constructing call-back function "cb_zselect_in_gees__user__routines"
    	  def zselect(arg): return zselect
    	Building module "flapack"...
    		Constructing wrapper function "sgesv"...
    		  lu,piv,x,info = sgesv(a,b,[overwrite_a,overwrite_b])
    		Constructing wrapper function "dgesv"...
    		  lu,piv,x,info = dgesv(a,b,[overwrite_a,overwrite_b])
    		Constructing wrapper function "cgesv"...
    		  lu,piv,x,info = cgesv(a,b,[overwrite_a,overwrite_b])
    		Constructing wrapper function "zgesv"...
    		  lu,piv,x,info = zgesv(a,b,[overwrite_a,overwrite_b])
    		Constructing wrapper function "sgbsv"...
    		  lub,piv,x,info = sgbsv(kl,ku,ab,b,[overwrite_ab,overwrite_b])
    		Constructing wrapper function "dgbsv"...
    		  lub,piv,x,info = dgbsv(kl,ku,ab,b,[overwrite_ab,overwrite_b])
    		Constructing wrapper function "cgbsv"...
    		  lub,piv,x,info = cgbsv(kl,ku,ab,b,[overwrite_ab,overwrite_b])
    		Constructing wrapper function "zgbsv"...
    		  lub,piv,x,info = zgbsv(kl,ku,ab,b,[overwrite_ab,overwrite_b])
    		Constructing wrapper function "sposv"...
    		  c,x,info = sposv(a,b,[lower,overwrite_a,overwrite_b])
    		Constructing wrapper function "dposv"...
    		  c,x,info = dposv(a,b,[lower,overwrite_a,overwrite_b])
    		Constructing wrapper function "cposv"...
    		  c,x,info = cposv(a,b,[lower,overwrite_a,overwrite_b])
    		Constructing wrapper function "zposv"...
    		  c,x,info = zposv(a,b,[lower,overwrite_a,overwrite_b])
    		Constructing wrapper function "sgelss"...
    		  v,x,s,rank,info = sgelss(a,b,[cond,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "dgelss"...
    		  v,x,s,rank,info = dgelss(a,b,[cond,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "cgelss"...
    		  v,x,s,rank,info = cgelss(a,b,[cond,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "zgelss"...
    		  v,x,s,rank,info = zgelss(a,b,[cond,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "ssyev"...
    		  w,v,info = ssyev(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "dsyev"...
    		  w,v,info = dsyev(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "cheev"...
    		  w,v,info = cheev(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "zheev"...
    		  w,v,info = zheev(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "ssyevd"...
    		  w,v,info = ssyevd(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "dsyevd"...
    		  w,v,info = dsyevd(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "cheevd"...
    		  w,v,info = cheevd(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "zheevd"...
    		  w,v,info = zheevd(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "ssyevr"...
    		  w,v,info = ssyevr(a,[compute_v,lower,vrange,irange,atol,lwork,overwrite_a])
    		Constructing wrapper function "dsyevr"...
    		  w,v,info = dsyevr(a,[compute_v,lower,vrange,irange,atol,lwork,overwrite_a])
    		Constructing wrapper function "cheevr"...
    		  w,v,info = cheevr(a,[compute_v,lower,vrange,irange,atol,lwork,overwrite_a])
    		Constructing wrapper function "zheevr"...
    		  w,v,info = zheevr(a,[compute_v,lower,vrange,irange,atol,lwork,overwrite_a])
    		Constructing wrapper function "sgees"...
    		  t,sdim,wr,wi,vs,info = sgees(sselect,a,[compute_v,sort_t,lwork,sselect_extra_args,overwrite_a])
    		Constructing wrapper function "dgees"...
    		  t,sdim,wr,wi,vs,info = dgees(dselect,a,[compute_v,sort_t,lwork,dselect_extra_args,overwrite_a])
    		Constructing wrapper function "cgees"...
    		  t,sdim,w,vs,info = cgees(cselect,a,[compute_v,sort_t,lwork,cselect_extra_args,overwrite_a])
    		Constructing wrapper function "zgees"...
    		  t,sdim,w,vs,info = zgees(zselect,a,[compute_v,sort_t,lwork,zselect_extra_args,overwrite_a])
    		Constructing wrapper function "sgeev"...
    		  wr,wi,vl,vr,info = sgeev(a,[compute_vl,compute_vr,lwork,overwrite_a])
    		Constructing wrapper function "dgeev"...
    		  wr,wi,vl,vr,info = dgeev(a,[compute_vl,compute_vr,lwork,overwrite_a])
    		Constructing wrapper function "cgeev"...
    		  w,vl,vr,info = cgeev(a,[compute_vl,compute_vr,lwork,overwrite_a])
    		Constructing wrapper function "zgeev"...
    		  w,vl,vr,info = zgeev(a,[compute_vl,compute_vr,lwork,overwrite_a])
    		Constructing wrapper function "sgesdd"...
    		  u,s,vt,info = sgesdd(a,[compute_uv,lwork,overwrite_a])
    		Constructing wrapper function "dgesdd"...
    		  u,s,vt,info = dgesdd(a,[compute_uv,lwork,overwrite_a])
    		Constructing wrapper function "cgesdd"...
    		  u,s,vt,info = cgesdd(a,[compute_uv,lwork,overwrite_a])
    		Constructing wrapper function "zgesdd"...
    		  u,s,vt,info = zgesdd(a,[compute_uv,lwork,overwrite_a])
    		Constructing wrapper function "ssygv"...
    		  w,v,info = ssygv(a,b,[itype,compute_v,lower,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "dsygv"...
    		  w,v,info = dsygv(a,b,[itype,compute_v,lower,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "chegv"...
    		  w,v,info = chegv(a,b,[itype,compute_v,lower,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "zhegv"...
    		  w,v,info = zhegv(a,b,[itype,compute_v,lower,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "ssygvd"...
    		  w,v,info = ssygvd(a,b,[itype,compute_v,lower,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "dsygvd"...
    		  w,v,info = dsygvd(a,b,[itype,compute_v,lower,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "chegvd"...
    		  w,v,info = chegvd(a,b,[itype,compute_v,lower,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "zhegvd"...
    		  w,v,info = zhegvd(a,b,[itype,compute_v,lower,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "sggev"...
    		  alphar,alphai,beta,vl,vr,info = sggev(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "dggev"...
    		  alphar,alphai,beta,vl,vr,info = dggev(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "cggev"...
    		  alpha,beta,vl,vr,info = cggev(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "zggev"...
    		  alpha,beta,vl,vr,info = zggev(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "sgetrf"...
    		  lu,piv,info = sgetrf(a,[overwrite_a])
    		Constructing wrapper function "dgetrf"...
    		  lu,piv,info = dgetrf(a,[overwrite_a])
    		Constructing wrapper function "cgetrf"...
    		  lu,piv,info = cgetrf(a,[overwrite_a])
    		Constructing wrapper function "zgetrf"...
    		  lu,piv,info = zgetrf(a,[overwrite_a])
    		Constructing wrapper function "spotrf"...
    		  c,info = spotrf(a,[lower,clean,overwrite_a])
    		Constructing wrapper function "dpotrf"...
    		  c,info = dpotrf(a,[lower,clean,overwrite_a])
    		Constructing wrapper function "cpotrf"...
    		  c,info = cpotrf(a,[lower,clean,overwrite_a])
    		Constructing wrapper function "zpotrf"...
    		  c,info = zpotrf(a,[lower,clean,overwrite_a])
    		Constructing wrapper function "sgetrs"...
    		  x,info = sgetrs(lu,piv,b,[trans,overwrite_b])
    		Constructing wrapper function "dgetrs"...
    		  x,info = dgetrs(lu,piv,b,[trans,overwrite_b])
    		Constructing wrapper function "cgetrs"...
    		  x,info = cgetrs(lu,piv,b,[trans,overwrite_b])
    		Constructing wrapper function "zgetrs"...
    		  x,info = zgetrs(lu,piv,b,[trans,overwrite_b])
    		Constructing wrapper function "spotrs"...
    		  x,info = spotrs(c,b,[lower,overwrite_b])
    		Constructing wrapper function "dpotrs"...
    		  x,info = dpotrs(c,b,[lower,overwrite_b])
    		Constructing wrapper function "cpotrs"...
    		  x,info = cpotrs(c,b,[lower,overwrite_b])
    		Constructing wrapper function "zpotrs"...
    		  x,info = zpotrs(c,b,[lower,overwrite_b])
    		Constructing wrapper function "sgetri"...
    		  inv_a,info = sgetri(lu,piv,[lwork,overwrite_lu])
    		Constructing wrapper function "dgetri"...
    		  inv_a,info = dgetri(lu,piv,[lwork,overwrite_lu])
    		Constructing wrapper function "cgetri"...
    		  inv_a,info = cgetri(lu,piv,[lwork,overwrite_lu])
    		Constructing wrapper function "zgetri"...
    		  inv_a,info = zgetri(lu,piv,[lwork,overwrite_lu])
    		Constructing wrapper function "spotri"...
    		  inv_a,info = spotri(c,[lower,overwrite_c])
    		Constructing wrapper function "dpotri"...
    		  inv_a,info = dpotri(c,[lower,overwrite_c])
    		Constructing wrapper function "cpotri"...
    		  inv_a,info = cpotri(c,[lower,overwrite_c])
    		Constructing wrapper function "zpotri"...
    		  inv_a,info = zpotri(c,[lower,overwrite_c])
    		Constructing wrapper function "strtri"...
    		  inv_c,info = strtri(c,[lower,unitdiag,overwrite_c])
    		Constructing wrapper function "dtrtri"...
    		  inv_c,info = dtrtri(c,[lower,unitdiag,overwrite_c])
    		Constructing wrapper function "ctrtri"...
    		  inv_c,info = ctrtri(c,[lower,unitdiag,overwrite_c])
    		Constructing wrapper function "ztrtri"...
    		  inv_c,info = ztrtri(c,[lower,unitdiag,overwrite_c])
    		Constructing wrapper function "sgeqrf"...
    		  qr,tau,info = sgeqrf(a,[lwork,overwrite_a])
    		Constructing wrapper function "dgeqrf"...
    		  qr,tau,info = dgeqrf(a,[lwork,overwrite_a])
    		Constructing wrapper function "cgeqrf"...
    		  qr,tau,info = cgeqrf(a,[lwork,overwrite_a])
    		Constructing wrapper function "zgeqrf"...
    		  qr,tau,info = zgeqrf(a,[lwork,overwrite_a])
    		Constructing wrapper function "sorgqr"...
    		  q,info = sorgqr(qr,tau,[lwork,overwrite_qr,overwrite_tau])
    		Constructing wrapper function "dorgqr"...
    		  q,info = dorgqr(qr,tau,[lwork,overwrite_qr,overwrite_tau])
    		Constructing wrapper function "cungqr"...
    		  q,info = cungqr(qr,tau,[lwork,overwrite_qr,overwrite_tau])
    		Constructing wrapper function "zungqr"...
    		  q,info = zungqr(qr,tau,[lwork,overwrite_qr,overwrite_tau])
    		Constructing wrapper function "sgehrd"...
    		  ht,tau,info = sgehrd(a,[lo,hi,lwork,overwrite_a])
    		Constructing wrapper function "dgehrd"...
    		  ht,tau,info = dgehrd(a,[lo,hi,lwork,overwrite_a])
    		Constructing wrapper function "cgehrd"...
    		  ht,tau,info = cgehrd(a,[lo,hi,lwork,overwrite_a])
    		Constructing wrapper function "zgehrd"...
    		  ht,tau,info = zgehrd(a,[lo,hi,lwork,overwrite_a])
    		Constructing wrapper function "sgebal"...
    		  ba,lo,hi,pivscale,info = sgebal(a,[scale,permute,overwrite_a])
    		Constructing wrapper function "dgebal"...
    		  ba,lo,hi,pivscale,info = dgebal(a,[scale,permute,overwrite_a])
    		Constructing wrapper function "cgebal"...
    		  ba,lo,hi,pivscale,info = cgebal(a,[scale,permute,overwrite_a])
    		Constructing wrapper function "zgebal"...
    		  ba,lo,hi,pivscale,info = zgebal(a,[scale,permute,overwrite_a])
    		Constructing wrapper function "slauum"...
    		  a,info = slauum(c,[lower,overwrite_c])
    		Constructing wrapper function "dlauum"...
    		  a,info = dlauum(c,[lower,overwrite_c])
    		Constructing wrapper function "clauum"...
    		  a,info = clauum(c,[lower,overwrite_c])
    		Constructing wrapper function "zlauum"...
    		  a,info = zlauum(c,[lower,overwrite_c])
    		Constructing wrapper function "slaswp"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  a = slaswp(a,piv,[k1,k2,off,inc,overwrite_a])
    		Constructing wrapper function "dlaswp"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  a = dlaswp(a,piv,[k1,k2,off,inc,overwrite_a])
    		Constructing wrapper function "claswp"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  a = claswp(a,piv,[k1,k2,off,inc,overwrite_a])
    		Constructing wrapper function "zlaswp"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  a = zlaswp(a,piv,[k1,k2,off,inc,overwrite_a])
    	Wrote C/API module "flapack" to file "build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/flapackmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.lib.lapack.clapack" sources
      adding 'scipy/lib/lapack/clapack.pyf.src' to sources.
    from_template:> build/src.linux-x86_64-2.7/scipy/lib/lapack/clapack.pyf
    f2py options: []
    f2py: build/src.linux-x86_64-2.7/scipy/lib/lapack/clapack.pyf
    Reading fortran codes...
    	Reading file 'build/src.linux-x86_64-2.7/scipy/lib/lapack/clapack.pyf' (format:free)
    Post-processing...
    	Block: clapack
    			Block: sgesv
    			Block: dgesv
    			Block: cgesv
    			Block: zgesv
    			Block: sposv
    			Block: dposv
    			Block: cposv
    			Block: zposv
    			Block: spotrf
    			Block: dpotrf
    			Block: cpotrf
    			Block: zpotrf
    			Block: spotrs
    			Block: dpotrs
    			Block: cpotrs
    			Block: zpotrs
    			Block: spotri
    			Block: dpotri
    			Block: cpotri
    			Block: zpotri
    			Block: slauum
    			Block: dlauum
    			Block: clauum
    			Block: zlauum
    			Block: strtri
    			Block: dtrtri
    			Block: ctrtri
    			Block: ztrtri
    Post-processing (stage 2)...
    Building modules...
    	Building module "clapack"...
    		Constructing wrapper function "sgesv"...
    		  lu,piv,x,info = sgesv(a,b,[rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "dgesv"...
    		  lu,piv,x,info = dgesv(a,b,[rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "cgesv"...
    		  lu,piv,x,info = cgesv(a,b,[rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "zgesv"...
    		  lu,piv,x,info = zgesv(a,b,[rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "sposv"...
    		  c,x,info = sposv(a,b,[lower,rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "dposv"...
    		  c,x,info = dposv(a,b,[lower,rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "cposv"...
    		  c,x,info = cposv(a,b,[lower,rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "zposv"...
    		  c,x,info = zposv(a,b,[lower,rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "spotrf"...
    		  c,info = spotrf(a,[lower,clean,rowmajor,overwrite_a])
    		Constructing wrapper function "dpotrf"...
    		  c,info = dpotrf(a,[lower,clean,rowmajor,overwrite_a])
    		Constructing wrapper function "cpotrf"...
    		  c,info = cpotrf(a,[lower,clean,rowmajor,overwrite_a])
    		Constructing wrapper function "zpotrf"...
    		  c,info = zpotrf(a,[lower,clean,rowmajor,overwrite_a])
    		Constructing wrapper function "spotrs"...
    		  x,info = spotrs(c,b,[lower,rowmajor,overwrite_b])
    		Constructing wrapper function "dpotrs"...
    		  x,info = dpotrs(c,b,[lower,rowmajor,overwrite_b])
    		Constructing wrapper function "cpotrs"...
    		  x,info = cpotrs(c,b,[lower,rowmajor,overwrite_b])
    		Constructing wrapper function "zpotrs"...
    		  x,info = zpotrs(c,b,[lower,rowmajor,overwrite_b])
    		Constructing wrapper function "spotri"...
    		  inv_a,info = spotri(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "dpotri"...
    		  inv_a,info = dpotri(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "cpotri"...
    		  inv_a,info = cpotri(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "zpotri"...
    		  inv_a,info = zpotri(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "slauum"...
    		  a,info = slauum(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "dlauum"...
    		  a,info = dlauum(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "clauum"...
    		  a,info = clauum(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "zlauum"...
    		  a,info = zlauum(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "strtri"...
    		  inv_c,info = strtri(c,[lower,unitdiag,rowmajor,overwrite_c])
    		Constructing wrapper function "dtrtri"...
    		  inv_c,info = dtrtri(c,[lower,unitdiag,rowmajor,overwrite_c])
    		Constructing wrapper function "ctrtri"...
    		  inv_c,info = ctrtri(c,[lower,unitdiag,rowmajor,overwrite_c])
    		Constructing wrapper function "ztrtri"...
    		  inv_c,info = ztrtri(c,[lower,unitdiag,rowmajor,overwrite_c])
    	Wrote C/API module "clapack" to file "build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.lib.lapack.calc_lwork" sources
    f2py options: []
    f2py:> build/src.linux-x86_64-2.7/scipy/lib/lapack/calc_lworkmodule.c
    Reading fortran codes...
    	Reading file 'scipy/lib/lapack/calc_lwork.f' (format:fix,strict)
    Post-processing...
    	Block: calc_lwork
    			Block: gehrd
    			Block: gesdd
    			Block: gelss
    			Block: getri
    			Block: geev
    			Block: heev
    			Block: syev
    			Block: gees
    			Block: geqrf
    			Block: gqr
    Post-processing (stage 2)...
    Building modules...
    	Building module "calc_lwork"...
    		Constructing wrapper function "gehrd"...
    		  minwrk,maxwrk = gehrd(prefix,n,[lo,hi])
    		Constructing wrapper function "gesdd"...
    		  minwrk,maxwrk = gesdd(prefix,m,n,[compute_uv])
    		Constructing wrapper function "gelss"...
    		  minwrk,maxwrk = gelss(prefix,m,n,nrhs)
    		Constructing wrapper function "getri"...
    		  minwrk,maxwrk = getri(prefix,n)
    		Constructing wrapper function "geev"...
    		  minwrk,maxwrk = geev(prefix,n,[compute_vl,compute_vr])
    		Constructing wrapper function "heev"...
    		  minwrk,maxwrk = heev(prefix,n,[lower])
    		Constructing wrapper function "syev"...
    		  minwrk,maxwrk = syev(prefix,n,[lower])
    		Constructing wrapper function "gees"...
    		  minwrk,maxwrk = gees(prefix,n,[compute_v])
    		Constructing wrapper function "geqrf"...
    		  minwrk,maxwrk = geqrf(prefix,m,n)
    		Constructing wrapper function "gqr"...
    		  minwrk,maxwrk = gqr(prefix,m,n)
    	Wrote C/API module "calc_lwork" to file "build/src.linux-x86_64-2.7/scipy/lib/lapack/calc_lworkmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.linalg._fblas" sources
    from_template:> build/src.linux-x86_64-2.7/scipy/linalg/fblas.pyf
    Including file scipy/linalg/fblas_l1.pyf.src
    Including file scipy/linalg/fblas_l2.pyf.src
    Including file scipy/linalg/fblas_l3.pyf.src
    Mismatch in number of replacements (base <prefix=s,d,c,z>) for <__l1=->. Ignoring.
    Mismatch in number of replacements (base <prefix6=s,d,c,z,c,z>) for <prefix=s,d,c,z>. Ignoring.
    f2py options: []
    f2py: build/src.linux-x86_64-2.7/scipy/linalg/fblas.pyf
    Reading fortran codes...
    	Reading file 'build/src.linux-x86_64-2.7/scipy/linalg/fblas.pyf' (format:free)
    Post-processing...
    	Block: _fblas
    			Block: srotg
    			Block: drotg
    			Block: crotg
    			Block: zrotg
    			Block: srotmg
    			Block: drotmg
    			Block: srot
    			Block: drot
    			Block: csrot
    			Block: zdrot
    			Block: srotm
    			Block: drotm
    			Block: sswap
    			Block: dswap
    			Block: cswap
    			Block: zswap
    			Block: sscal
    			Block: dscal
    			Block: cscal
    			Block: zscal
    			Block: csscal
    			Block: zdscal
    			Block: scopy
    			Block: dcopy
    			Block: ccopy
    			Block: zcopy
    			Block: saxpy
    			Block: daxpy
    			Block: caxpy
    			Block: zaxpy
    			Block: sdot
    			Block: ddot
    			Block: cdotu
    			Block: zdotu
    			Block: cdotc
    			Block: zdotc
    			Block: snrm2
    			Block: scnrm2
    			Block: dnrm2
    			Block: dznrm2
    			Block: sasum
    			Block: scasum
    			Block: dasum
    			Block: dzasum
    			Block: isamax
    			Block: idamax
    			Block: icamax
    			Block: izamax
    			Block: sgemv
    			Block: dgemv
    			Block: cgemv
    			Block: zgemv
    			Block: ssymv
    			Block: dsymv
    			Block: chemv
    			Block: zhemv
    			Block: strmv
    			Block: dtrmv
    			Block: ctrmv
    			Block: ztrmv
    			Block: sger
    			Block: dger
    			Block: cgeru
    			Block: zgeru
    			Block: cgerc
    			Block: zgerc
    			Block: sgemm
    			Block: dgemm
    			Block: cgemm
    			Block: zgemm
    			Block: ssymm
    			Block: dsymm
    			Block: csymm
    			Block: zsymm
    			Block: chemm
    			Block: zhemm
    			Block: ssyrk
    			Block: dsyrk
    			Block: csyrk
    			Block: zsyrk
    			Block: cherk
    			Block: zherk
    			Block: ssyr2k
    			Block: dsyr2k
    			Block: csyr2k
    			Block: zsyr2k
    			Block: cher2k
    			Block: zher2k
    Post-processing (stage 2)...
    Building modules...
    	Building module "_fblas"...
    		Constructing wrapper function "srotg"...
    		  c,s = srotg(a,b)
    		Constructing wrapper function "drotg"...
    		  c,s = drotg(a,b)
    		Constructing wrapper function "crotg"...
    		  c,s = crotg(a,b)
    		Constructing wrapper function "zrotg"...
    		  c,s = zrotg(a,b)
    		Constructing wrapper function "srotmg"...
    		  param = srotmg(d1,d2,x1,y1)
    		Constructing wrapper function "drotmg"...
    		  param = drotmg(d1,d2,x1,y1)
    		Constructing wrapper function "srot"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = srot(x,y,c,s,[n,offx,incx,offy,incy,overwrite_x,overwrite_y])
    		Constructing wrapper function "drot"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = drot(x,y,c,s,[n,offx,incx,offy,incy,overwrite_x,overwrite_y])
    		Constructing wrapper function "csrot"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = csrot(x,y,c,s,[n,offx,incx,offy,incy,overwrite_x,overwrite_y])
    		Constructing wrapper function "zdrot"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = zdrot(x,y,c,s,[n,offx,incx,offy,incy,overwrite_x,overwrite_y])
    		Constructing wrapper function "srotm"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = srotm(x,y,param,[n,offx,incx,offy,incy,overwrite_x,overwrite_y])
    		Constructing wrapper function "drotm"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = drotm(x,y,param,[n,offx,incx,offy,incy,overwrite_x,overwrite_y])
    		Constructing wrapper function "sswap"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = sswap(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "dswap"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = dswap(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "cswap"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = cswap(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "zswap"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,y = zswap(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "sscal"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = sscal(a,x,[n,offx,incx])
    		Constructing wrapper function "dscal"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = dscal(a,x,[n,offx,incx])
    		Constructing wrapper function "cscal"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = cscal(a,x,[n,offx,incx])
    		Constructing wrapper function "zscal"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = zscal(a,x,[n,offx,incx])
    		Constructing wrapper function "csscal"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = csscal(a,x,[n,offx,incx,overwrite_x])
    		Constructing wrapper function "zdscal"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = zdscal(a,x,[n,offx,incx,overwrite_x])
    		Constructing wrapper function "scopy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = scopy(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "dcopy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = dcopy(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "ccopy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = ccopy(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "zcopy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = zcopy(x,y,[n,offx,incx,offy,incy])
    		Constructing wrapper function "saxpy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  z = saxpy(x,y,[n,a,offx,incx,offy,incy])
    		Constructing wrapper function "daxpy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  z = daxpy(x,y,[n,a,offx,incx,offy,incy])
    		Constructing wrapper function "caxpy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  z = caxpy(x,y,[n,a,offx,incx,offy,incy])
    		Constructing wrapper function "zaxpy"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  z = zaxpy(x,y,[n,a,offx,incx,offy,incy])
    		Creating wrapper for Fortran function "sdot"("wsdot")...
    		Constructing wrapper function "sdot"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  xy = sdot(x,y,[n,offx,incx,offy,incy])
    		Creating wrapper for Fortran function "ddot"("ddot")...
    		Constructing wrapper function "ddot"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  xy = ddot(x,y,[n,offx,incx,offy,incy])
    		Creating wrapper for Fortran function "cdotu"("wcdotu")...
    		Constructing wrapper function "cdotu"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  xy = cdotu(x,y,[n,offx,incx,offy,incy])
    		Creating wrapper for Fortran function "zdotu"("wzdotu")...
    		Constructing wrapper function "zdotu"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  xy = zdotu(x,y,[n,offx,incx,offy,incy])
    		Creating wrapper for Fortran function "cdotc"("wcdotc")...
    		Constructing wrapper function "cdotc"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  xy = cdotc(x,y,[n,offx,incx,offy,incy])
    		Creating wrapper for Fortran function "zdotc"("wzdotc")...
    		Constructing wrapper function "zdotc"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  xy = zdotc(x,y,[n,offx,incx,offy,incy])
    		Creating wrapper for Fortran function "snrm2"("wsnrm2")...
    		Constructing wrapper function "snrm2"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  n2 = snrm2(x,[n,offx,incx])
    		Creating wrapper for Fortran function "scnrm2"("wscnrm2")...
    		Constructing wrapper function "scnrm2"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  n2 = scnrm2(x,[n,offx,incx])
    		Creating wrapper for Fortran function "dnrm2"("dnrm2")...
    		Constructing wrapper function "dnrm2"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  n2 = dnrm2(x,[n,offx,incx])
    		Creating wrapper for Fortran function "dznrm2"("dznrm2")...
    		Constructing wrapper function "dznrm2"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  n2 = dznrm2(x,[n,offx,incx])
    		Creating wrapper for Fortran function "sasum"("wsasum")...
    		Constructing wrapper function "sasum"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  s = sasum(x,[n,offx,incx])
    		Creating wrapper for Fortran function "scasum"("wscasum")...
    		Constructing wrapper function "scasum"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  s = scasum(x,[n,offx,incx])
    		Creating wrapper for Fortran function "dasum"("dasum")...
    		Constructing wrapper function "dasum"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  s = dasum(x,[n,offx,incx])
    		Creating wrapper for Fortran function "dzasum"("dzasum")...
    		Constructing wrapper function "dzasum"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  s = dzasum(x,[n,offx,incx])
    		Constructing wrapper function "isamax"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  k = isamax(x,[n,offx,incx])
    		Constructing wrapper function "idamax"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  k = idamax(x,[n,offx,incx])
    		Constructing wrapper function "icamax"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  k = icamax(x,[n,offx,incx])
    		Constructing wrapper function "izamax"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  k = izamax(x,[n,offx,incx])
    		Constructing wrapper function "sgemv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = sgemv(alpha,a,x,[beta,y,offx,incx,offy,incy,trans,overwrite_y])
    		Constructing wrapper function "dgemv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = dgemv(alpha,a,x,[beta,y,offx,incx,offy,incy,trans,overwrite_y])
    		Constructing wrapper function "cgemv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = cgemv(alpha,a,x,[beta,y,offx,incx,offy,incy,trans,overwrite_y])
    		Constructing wrapper function "zgemv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = zgemv(alpha,a,x,[beta,y,offx,incx,offy,incy,trans,overwrite_y])
    		Constructing wrapper function "ssymv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = ssymv(alpha,a,x,[beta,y,offx,incx,offy,incy,lower,overwrite_y])
    		Constructing wrapper function "dsymv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = dsymv(alpha,a,x,[beta,y,offx,incx,offy,incy,lower,overwrite_y])
    		Constructing wrapper function "chemv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = chemv(alpha,a,x,[beta,y,offx,incx,offy,incy,lower,overwrite_y])
    		Constructing wrapper function "zhemv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  y = zhemv(alpha,a,x,[beta,y,offx,incx,offy,incy,lower,overwrite_y])
    		Constructing wrapper function "strmv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = strmv(a,x,[offx,incx,lower,trans,unitdiag,overwrite_x])
    		Constructing wrapper function "dtrmv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = dtrmv(a,x,[offx,incx,lower,trans,unitdiag,overwrite_x])
    		Constructing wrapper function "ctrmv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = ctrmv(a,x,[offx,incx,lower,trans,unitdiag,overwrite_x])
    		Constructing wrapper function "ztrmv"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x = ztrmv(a,x,[offx,incx,lower,trans,unitdiag,overwrite_x])
    		Constructing wrapper function "sger"...
    		  a = sger(alpha,x,y,[incx,incy,a,overwrite_x,overwrite_y,overwrite_a])
    		Constructing wrapper function "dger"...
    		  a = dger(alpha,x,y,[incx,incy,a,overwrite_x,overwrite_y,overwrite_a])
    		Constructing wrapper function "cgeru"...
    		  a = cgeru(alpha,x,y,[incx,incy,a,overwrite_x,overwrite_y,overwrite_a])
    		Constructing wrapper function "zgeru"...
    		  a = zgeru(alpha,x,y,[incx,incy,a,overwrite_x,overwrite_y,overwrite_a])
    		Constructing wrapper function "cgerc"...
    		  a = cgerc(alpha,x,y,[incx,incy,a,overwrite_x,overwrite_y,overwrite_a])
    		Constructing wrapper function "zgerc"...
    		  a = zgerc(alpha,x,y,[incx,incy,a,overwrite_x,overwrite_y,overwrite_a])
    		Constructing wrapper function "sgemm"...
    		  c = sgemm(alpha,a,b,[beta,c,trans_a,trans_b,overwrite_c])
    		Constructing wrapper function "dgemm"...
    		  c = dgemm(alpha,a,b,[beta,c,trans_a,trans_b,overwrite_c])
    		Constructing wrapper function "cgemm"...
    		  c = cgemm(alpha,a,b,[beta,c,trans_a,trans_b,overwrite_c])
    		Constructing wrapper function "zgemm"...
    		  c = zgemm(alpha,a,b,[beta,c,trans_a,trans_b,overwrite_c])
    		Constructing wrapper function "ssymm"...
    		  c = ssymm(alpha,a,b,[beta,c,side,lower,overwrite_c])
    		Constructing wrapper function "dsymm"...
    		  c = dsymm(alpha,a,b,[beta,c,side,lower,overwrite_c])
    		Constructing wrapper function "csymm"...
    		  c = csymm(alpha,a,b,[beta,c,side,lower,overwrite_c])
    		Constructing wrapper function "zsymm"...
    		  c = zsymm(alpha,a,b,[beta,c,side,lower,overwrite_c])
    		Constructing wrapper function "chemm"...
    		  c = chemm(alpha,a,b,[beta,c,side,lower,overwrite_c])
    		Constructing wrapper function "zhemm"...
    		  c = zhemm(alpha,a,b,[beta,c,side,lower,overwrite_c])
    		Constructing wrapper function "ssyrk"...
    		  c = ssyrk(alpha,a,[beta,c,trans,lower,overwrite_c])
    		Constructing wrapper function "dsyrk"...
    		  c = dsyrk(alpha,a,[beta,c,trans,lower,overwrite_c])
    		Constructing wrapper function "csyrk"...
    		  c = csyrk(alpha,a,[beta,c,trans,lower,overwrite_c])
    		Constructing wrapper function "zsyrk"...
    		  c = zsyrk(alpha,a,[beta,c,trans,lower,overwrite_c])
    		Constructing wrapper function "cherk"...
    		  c = cherk(alpha,a,[beta,c,trans,lower,overwrite_c])
    		Constructing wrapper function "zherk"...
    		  c = zherk(alpha,a,[beta,c,trans,lower,overwrite_c])
    		Constructing wrapper function "ssyr2k"...
    		  c = ssyr2k(alpha,a,b,[beta,c,trans,lower,overwrite_c])
    		Constructing wrapper function "dsyr2k"...
    		  c = dsyr2k(alpha,a,b,[beta,c,trans,lower,overwrite_c])
    		Constructing wrapper function "csyr2k"...
    		  c = csyr2k(alpha,a,b,[beta,c,trans,lower,overwrite_c])
    		Constructing wrapper function "zsyr2k"...
    		  c = zsyr2k(alpha,a,b,[beta,c,trans,lower,overwrite_c])
    		Constructing wrapper function "cher2k"...
    		  c = cher2k(alpha,a,b,[beta,c,trans,lower,overwrite_c])
    		Constructing wrapper function "zher2k"...
    		  c = zher2k(alpha,a,b,[beta,c,trans,lower,overwrite_c])
    	Wrote C/API module "_fblas" to file "build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_fblasmodule.c"
    	Fortran 77 wrappers are saved to "build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_fblas-f2pywrappers.f"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
      adding 'build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_fblas-f2pywrappers.f' to sources.
    building extension "scipy.linalg._flapack" sources
    from_template:> build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf
    Including file scipy/linalg/flapack_user.pyf.src
    f2py options: []
    f2py: build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf
    Reading fortran codes...
    	Reading file 'build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf' (format:free)
    Line #4338 in build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf:"  char*,char*,char*,int*,int*,complex_float*,int*,complex_float*,int*,float*,float*,int*,int*,float*,int*,float*,complex_float*,int*,complex_float*,float*,int*,int*,int*"
    	crackline:3: No pattern for line
    Line #4418 in build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf:"  char*,char*,char*,int*,int*,complex_double*,int*,complex_double*,int*,double*,double*,int*,int*,double*,int*,double*,complex_double*,int*,complex_double*,double*,int*,int*,int*"
    	crackline:3: No pattern for line
    Line #4627 in build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf:"lprotoargument char*,int*,int *,int*,int*,float*,int*,int*,float*,int*,int*"
    	crackline:3: No pattern for line
    Line #4660 in build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf:"lprotoargument char*,int*,int *,int*,int*,double*,int*,int*,double*,int*,int*"
    	crackline:3: No pattern for line
    Line #4693 in build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf:"lprotoargument char*,int*,int *,int*,int*,complex_float*,int*,int*,complex_float*,int*,int*"
    	crackline:3: No pattern for line
    Line #4726 in build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf:"lprotoargument char*,int*,int *,int*,int*,complex_double*,int*,int*,complex_double*,int*,int*"
    	crackline:3: No pattern for line
    Post-processing...
    	Block: _flapack
    			Block: gees__user__routines
    					Block: sselect
    					Block: dselect
    					Block: cselect
    					Block: zselect
    			Block: gges__user__routines
    					Block: cselect
    					Block: zselect
    					Block: sselect
    					Block: dselect
    			Block: sgges
    In: build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf:_flapack:unknown_interface:sgges
    get_useparameters: no module gges__user__routines info used by sgges
    			Block: dgges
    In: build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf:_flapack:unknown_interface:dgges
    get_useparameters: no module gges__user__routines info used by dgges
    			Block: cgges
    In: build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf:_flapack:unknown_interface:cgges
    get_useparameters: no module gges__user__routines info used by cgges
    			Block: zgges
    In: build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf:_flapack:unknown_interface:zgges
    get_useparameters: no module gges__user__routines info used by zgges
    			Block: spbtrf
    			Block: dpbtrf
    			Block: cpbtrf
    			Block: zpbtrf
    			Block: spbtrs
    			Block: dpbtrs
    			Block: cpbtrs
    			Block: zpbtrs
    			Block: strtrs
    			Block: dtrtrs
    			Block: ctrtrs
    			Block: ztrtrs
    			Block: spbsv
    			Block: dpbsv
    			Block: cpbsv
    			Block: zpbsv
    			Block: sgebal
    			Block: dgebal
    			Block: cgebal
    			Block: zgebal
    			Block: sgehrd
    			Block: dgehrd
    			Block: cgehrd
    			Block: zgehrd
    			Block: sgbsv
    			Block: dgbsv
    			Block: cgbsv
    			Block: zgbsv
    			Block: sgesv
    			Block: dgesv
    			Block: cgesv
    			Block: zgesv
    			Block: sgetrf
    			Block: dgetrf
    			Block: cgetrf
    			Block: zgetrf
    			Block: sgetrs
    			Block: dgetrs
    			Block: cgetrs
    			Block: zgetrs
    			Block: sgetri
    			Block: dgetri
    			Block: cgetri
    			Block: zgetri
    			Block: sgesdd
    			Block: dgesdd
    			Block: cgesdd
    			Block: zgesdd
    			Block: sgelss
    			Block: dgelss
    			Block: cgelss
    			Block: zgelss
    			Block: sgeqp3
    			Block: dgeqp3
    			Block: cgeqp3
    			Block: zgeqp3
    			Block: sgeqrf
    			Block: dgeqrf
    			Block: cgeqrf
    			Block: zgeqrf
    			Block: sgerqf
    			Block: dgerqf
    			Block: cgerqf
    			Block: zgerqf
    			Block: sorgqr
    			Block: dorgqr
    			Block: cungqr
    			Block: zungqr
    			Block: sormqr
    			Block: dormqr
    			Block: cunmqr
    			Block: zunmqr
    			Block: sorgrq
    			Block: dorgrq
    			Block: cungrq
    			Block: zungrq
    			Block: sgeev
    			Block: dgeev
    			Block: cgeev
    			Block: zgeev
    			Block: sgegv
    			Block: dgegv
    			Block: cgegv
    			Block: zgegv
    			Block: ssyev
    			Block: dsyev
    			Block: cheev
    			Block: zheev
    			Block: ssyevd
    			Block: dsyevd
    			Block: cheevd
    			Block: zheevd
    			Block: sposv
    			Block: dposv
    			Block: cposv
    			Block: zposv
    			Block: spotrf
    			Block: dpotrf
    			Block: cpotrf
    			Block: zpotrf
    			Block: spotrs
    			Block: dpotrs
    			Block: cpotrs
    			Block: zpotrs
    			Block: spotri
    			Block: dpotri
    			Block: cpotri
    			Block: zpotri
    			Block: slauum
    			Block: dlauum
    			Block: clauum
    			Block: zlauum
    			Block: strtri
    			Block: dtrtri
    			Block: ctrtri
    			Block: ztrtri
    			Block: strsyl
    			Block: dtrsyl
    			Block: ctrsyl
    			Block: ztrsyl
    			Block: slaswp
    			Block: dlaswp
    			Block: claswp
    			Block: zlaswp
    			Block: cgees
    In: build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf:_flapack:unknown_interface:cgees
    get_useparameters: no module gees__user__routines info used by cgees
    			Block: zgees
    In: build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf:_flapack:unknown_interface:zgees
    get_useparameters: no module gees__user__routines info used by zgees
    			Block: sgees
    In: build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf:_flapack:unknown_interface:sgees
    get_useparameters: no module gees__user__routines info used by sgees
    			Block: dgees
    In: build/src.linux-x86_64-2.7/scipy/linalg/flapack.pyf:_flapack:unknown_interface:dgees
    get_useparameters: no module gees__user__routines info used by dgees
    			Block: sggev
    			Block: dggev
    			Block: cggev
    			Block: zggev
    			Block: ssbev
    			Block: dsbev
    			Block: ssbevd
    			Block: dsbevd
    			Block: ssbevx
    			Block: dsbevx
    			Block: chbevd
    			Block: zhbevd
    			Block: chbevx
    			Block: zhbevx
    			Block: dlamch
    			Block: slamch
    			Block: sgbtrf
    			Block: dgbtrf
    			Block: cgbtrf
    			Block: zgbtrf
    			Block: sgbtrs
    			Block: dgbtrs
    			Block: cgbtrs
    			Block: zgbtrs
    			Block: ssyevr
    			Block: dsyevr
    			Block: cheevr
    			Block: zheevr
    			Block: ssygv
    			Block: dsygv
    			Block: chegv
    			Block: zhegv
    			Block: ssygvd
    			Block: dsygvd
    			Block: chegvd
    			Block: zhegvd
    			Block: ssygvx
    			Block: dsygvx
    			Block: chegvx
    			Block: zhegvx
    Post-processing (stage 2)...
    Building modules...
    	Constructing call-back function "cb_sselect_in_gees__user__routines"
    	  def sselect(arg1,arg2): return sselect
    	Constructing call-back function "cb_dselect_in_gees__user__routines"
    	  def dselect(arg1,arg2): return dselect
    	Constructing call-back function "cb_cselect_in_gees__user__routines"
    	  def cselect(arg): return cselect
    	Constructing call-back function "cb_zselect_in_gees__user__routines"
    	  def zselect(arg): return zselect
    	Constructing call-back function "cb_cselect_in_gges__user__routines"
    	  def cselect(alpha,beta): return cselect
    	Constructing call-back function "cb_zselect_in_gges__user__routines"
    	  def zselect(alpha,beta): return zselect
    	Constructing call-back function "cb_sselect_in_gges__user__routines"
    	  def sselect(alphar,alphai,beta): return sselect
    	Constructing call-back function "cb_dselect_in_gges__user__routines"
    	  def dselect(alphar,alphai,beta): return dselect
    	Building module "_flapack"...
    		Constructing wrapper function "sgges"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  a,b,sdim,alphar,alphai,beta,vsl,vsr,work,info = sgges(sselect,a,b,[jobvsl,jobvsr,sort_t,ldvsl,ldvsr,lwork,sselect_extra_args,overwrite_a,overwrite_b])
    		Constructing wrapper function "dgges"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  a,b,sdim,alphar,alphai,beta,vsl,vsr,work,info = dgges(dselect,a,b,[jobvsl,jobvsr,sort_t,ldvsl,ldvsr,lwork,dselect_extra_args,overwrite_a,overwrite_b])
    		Constructing wrapper function "cgges"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  a,b,sdim,alpha,beta,vsl,vsr,work,info = cgges(cselect,a,b,[jobvsl,jobvsr,sort_t,ldvsl,ldvsr,lwork,cselect_extra_args,overwrite_a,overwrite_b])
    		Constructing wrapper function "zgges"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  a,b,sdim,alpha,beta,vsl,vsr,work,info = zgges(zselect,a,b,[jobvsl,jobvsr,sort_t,ldvsl,ldvsr,lwork,zselect_extra_args,overwrite_a,overwrite_b])
    		Constructing wrapper function "spbtrf"...
    		  c,info = spbtrf(ab,[lower,ldab,overwrite_ab])
    		Constructing wrapper function "dpbtrf"...
    		  c,info = dpbtrf(ab,[lower,ldab,overwrite_ab])
    		Constructing wrapper function "cpbtrf"...
    		  c,info = cpbtrf(ab,[lower,ldab,overwrite_ab])
    		Constructing wrapper function "zpbtrf"...
    		  c,info = zpbtrf(ab,[lower,ldab,overwrite_ab])
    		Constructing wrapper function "spbtrs"...
    		  x,info = spbtrs(ab,b,[lower,ldab,overwrite_b])
    		Constructing wrapper function "dpbtrs"...
    		  x,info = dpbtrs(ab,b,[lower,ldab,overwrite_b])
    		Constructing wrapper function "cpbtrs"...
    		  x,info = cpbtrs(ab,b,[lower,ldab,overwrite_b])
    		Constructing wrapper function "zpbtrs"...
    		  x,info = zpbtrs(ab,b,[lower,ldab,overwrite_b])
    		Constructing wrapper function "strtrs"...
    		  x,info = strtrs(a,b,[lower,trans,unitdiag,lda,overwrite_b])
    		Constructing wrapper function "dtrtrs"...
    		  x,info = dtrtrs(a,b,[lower,trans,unitdiag,lda,overwrite_b])
    		Constructing wrapper function "ctrtrs"...
    		  x,info = ctrtrs(a,b,[lower,trans,unitdiag,lda,overwrite_b])
    		Constructing wrapper function "ztrtrs"...
    		  x,info = ztrtrs(a,b,[lower,trans,unitdiag,lda,overwrite_b])
    		Constructing wrapper function "spbsv"...
    		  c,x,info = spbsv(ab,b,[lower,ldab,overwrite_ab,overwrite_b])
    		Constructing wrapper function "dpbsv"...
    		  c,x,info = dpbsv(ab,b,[lower,ldab,overwrite_ab,overwrite_b])
    		Constructing wrapper function "cpbsv"...
    		  c,x,info = cpbsv(ab,b,[lower,ldab,overwrite_ab,overwrite_b])
    		Constructing wrapper function "zpbsv"...
    		  c,x,info = zpbsv(ab,b,[lower,ldab,overwrite_ab,overwrite_b])
    		Constructing wrapper function "sgebal"...
    		  ba,lo,hi,pivscale,info = sgebal(a,[scale,permute,overwrite_a])
    		Constructing wrapper function "dgebal"...
    		  ba,lo,hi,pivscale,info = dgebal(a,[scale,permute,overwrite_a])
    		Constructing wrapper function "cgebal"...
    		  ba,lo,hi,pivscale,info = cgebal(a,[scale,permute,overwrite_a])
    		Constructing wrapper function "zgebal"...
    		  ba,lo,hi,pivscale,info = zgebal(a,[scale,permute,overwrite_a])
    		Constructing wrapper function "sgehrd"...
    		  ht,tau,info = sgehrd(a,[lo,hi,lwork,overwrite_a])
    		Constructing wrapper function "dgehrd"...
    		  ht,tau,info = dgehrd(a,[lo,hi,lwork,overwrite_a])
    		Constructing wrapper function "cgehrd"...
    		  ht,tau,info = cgehrd(a,[lo,hi,lwork,overwrite_a])
    		Constructing wrapper function "zgehrd"...
    		  ht,tau,info = zgehrd(a,[lo,hi,lwork,overwrite_a])
    		Constructing wrapper function "sgbsv"...
    		  lub,piv,x,info = sgbsv(kl,ku,ab,b,[overwrite_ab,overwrite_b])
    		Constructing wrapper function "dgbsv"...
    		  lub,piv,x,info = dgbsv(kl,ku,ab,b,[overwrite_ab,overwrite_b])
    		Constructing wrapper function "cgbsv"...
    		  lub,piv,x,info = cgbsv(kl,ku,ab,b,[overwrite_ab,overwrite_b])
    		Constructing wrapper function "zgbsv"...
    		  lub,piv,x,info = zgbsv(kl,ku,ab,b,[overwrite_ab,overwrite_b])
    		Constructing wrapper function "sgesv"...
    		  lu,piv,x,info = sgesv(a,b,[overwrite_a,overwrite_b])
    		Constructing wrapper function "dgesv"...
    		  lu,piv,x,info = dgesv(a,b,[overwrite_a,overwrite_b])
    		Constructing wrapper function "cgesv"...
    		  lu,piv,x,info = cgesv(a,b,[overwrite_a,overwrite_b])
    		Constructing wrapper function "zgesv"...
    		  lu,piv,x,info = zgesv(a,b,[overwrite_a,overwrite_b])
    		Constructing wrapper function "sgetrf"...
    		  lu,piv,info = sgetrf(a,[overwrite_a])
    		Constructing wrapper function "dgetrf"...
    		  lu,piv,info = dgetrf(a,[overwrite_a])
    		Constructing wrapper function "cgetrf"...
    		  lu,piv,info = cgetrf(a,[overwrite_a])
    		Constructing wrapper function "zgetrf"...
    		  lu,piv,info = zgetrf(a,[overwrite_a])
    		Constructing wrapper function "sgetrs"...
    		  x,info = sgetrs(lu,piv,b,[trans,overwrite_b])
    		Constructing wrapper function "dgetrs"...
    		  x,info = dgetrs(lu,piv,b,[trans,overwrite_b])
    		Constructing wrapper function "cgetrs"...
    		  x,info = cgetrs(lu,piv,b,[trans,overwrite_b])
    		Constructing wrapper function "zgetrs"...
    		  x,info = zgetrs(lu,piv,b,[trans,overwrite_b])
    		Constructing wrapper function "sgetri"...
    		  inv_a,info = sgetri(lu,piv,[lwork,overwrite_lu])
    		Constructing wrapper function "dgetri"...
    		  inv_a,info = dgetri(lu,piv,[lwork,overwrite_lu])
    		Constructing wrapper function "cgetri"...
    		  inv_a,info = cgetri(lu,piv,[lwork,overwrite_lu])
    		Constructing wrapper function "zgetri"...
    		  inv_a,info = zgetri(lu,piv,[lwork,overwrite_lu])
    		Constructing wrapper function "sgesdd"...
    		  u,s,vt,info = sgesdd(a,[compute_uv,full_matrices,lwork,overwrite_a])
    		Constructing wrapper function "dgesdd"...
    		  u,s,vt,info = dgesdd(a,[compute_uv,full_matrices,lwork,overwrite_a])
    		Constructing wrapper function "cgesdd"...
    		  u,s,vt,info = cgesdd(a,[compute_uv,full_matrices,lwork,overwrite_a])
    		Constructing wrapper function "zgesdd"...
    		  u,s,vt,info = zgesdd(a,[compute_uv,full_matrices,lwork,overwrite_a])
    		Constructing wrapper function "sgelss"...
    		  v,x,s,rank,work,info = sgelss(a,b,[cond,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "dgelss"...
    		  v,x,s,rank,work,info = dgelss(a,b,[cond,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "cgelss"...
    		  v,x,s,rank,work,info = cgelss(a,b,[cond,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "zgelss"...
    		  v,x,s,rank,work,info = zgelss(a,b,[cond,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "sgeqp3"...
    		  qr,jpvt,tau,work,info = sgeqp3(a,[lwork,overwrite_a])
    		Constructing wrapper function "dgeqp3"...
    		  qr,jpvt,tau,work,info = dgeqp3(a,[lwork,overwrite_a])
    		Constructing wrapper function "cgeqp3"...
    		  qr,jpvt,tau,work,info = cgeqp3(a,[lwork,overwrite_a])
    		Constructing wrapper function "zgeqp3"...
    		  qr,jpvt,tau,work,info = zgeqp3(a,[lwork,overwrite_a])
    		Constructing wrapper function "sgeqrf"...
    		  qr,tau,work,info = sgeqrf(a,[lwork,overwrite_a])
    		Constructing wrapper function "dgeqrf"...
    		  qr,tau,work,info = dgeqrf(a,[lwork,overwrite_a])
    		Constructing wrapper function "cgeqrf"...
    		  qr,tau,work,info = cgeqrf(a,[lwork,overwrite_a])
    		Constructing wrapper function "zgeqrf"...
    		  qr,tau,work,info = zgeqrf(a,[lwork,overwrite_a])
    		Constructing wrapper function "sgerqf"...
    		  qr,tau,work,info = sgerqf(a,[lwork,overwrite_a])
    		Constructing wrapper function "dgerqf"...
    		  qr,tau,work,info = dgerqf(a,[lwork,overwrite_a])
    		Constructing wrapper function "cgerqf"...
    		  qr,tau,work,info = cgerqf(a,[lwork,overwrite_a])
    		Constructing wrapper function "zgerqf"...
    		  qr,tau,work,info = zgerqf(a,[lwork,overwrite_a])
    		Constructing wrapper function "sorgqr"...
    		  q,work,info = sorgqr(a,tau,[lwork,overwrite_a])
    		Constructing wrapper function "dorgqr"...
    		  q,work,info = dorgqr(a,tau,[lwork,overwrite_a])
    		Constructing wrapper function "cungqr"...
    		  q,work,info = cungqr(a,tau,[lwork,overwrite_a])
    		Constructing wrapper function "zungqr"...
    		  q,work,info = zungqr(a,tau,[lwork,overwrite_a])
    		Constructing wrapper function "sormqr"...
    		  cq,work,info = sormqr(side,trans,a,tau,c,lwork,[overwrite_c])
    		Constructing wrapper function "dormqr"...
    		  cq,work,info = dormqr(side,trans,a,tau,c,lwork,[overwrite_c])
    		Constructing wrapper function "cunmqr"...
    		  cq,work,info = cunmqr(side,trans,a,tau,c,lwork,[overwrite_c])
    		Constructing wrapper function "zunmqr"...
    		  cq,work,info = zunmqr(side,trans,a,tau,c,lwork,[overwrite_c])
    		Constructing wrapper function "sorgrq"...
    		  q,work,info = sorgrq(a,tau,[lwork,overwrite_a])
    		Constructing wrapper function "dorgrq"...
    		  q,work,info = dorgrq(a,tau,[lwork,overwrite_a])
    		Constructing wrapper function "cungrq"...
    		  q,work,info = cungrq(a,tau,[lwork,overwrite_a])
    		Constructing wrapper function "zungrq"...
    		  q,work,info = zungrq(a,tau,[lwork,overwrite_a])
    		Constructing wrapper function "sgeev"...
    		  wr,wi,vl,vr,info = sgeev(a,[compute_vl,compute_vr,lwork,overwrite_a])
    		Constructing wrapper function "dgeev"...
    		  wr,wi,vl,vr,info = dgeev(a,[compute_vl,compute_vr,lwork,overwrite_a])
    		Constructing wrapper function "cgeev"...
    		  w,vl,vr,info = cgeev(a,[compute_vl,compute_vr,lwork,overwrite_a])
    		Constructing wrapper function "zgeev"...
    		  w,vl,vr,info = zgeev(a,[compute_vl,compute_vr,lwork,overwrite_a])
    		Constructing wrapper function "sgegv"...
    		  alphar,alphai,beta,vl,vr,info = sgegv(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "dgegv"...
    		  alphar,alphai,beta,vl,vr,info = dgegv(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "cgegv"...
    		  alpha,beta,vl,vr,info = cgegv(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "zgegv"...
    		  alpha,beta,vl,vr,info = zgegv(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "ssyev"...
    		  w,v,info = ssyev(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "dsyev"...
    		  w,v,info = dsyev(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "cheev"...
    		  w,v,info = cheev(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "zheev"...
    		  w,v,info = zheev(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "ssyevd"...
    		  w,v,info = ssyevd(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "dsyevd"...
    		  w,v,info = dsyevd(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "cheevd"...
    		  w,v,info = cheevd(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "zheevd"...
    		  w,v,info = zheevd(a,[compute_v,lower,lwork,overwrite_a])
    		Constructing wrapper function "sposv"...
    		  c,x,info = sposv(a,b,[lower,overwrite_a,overwrite_b])
    		Constructing wrapper function "dposv"...
    		  c,x,info = dposv(a,b,[lower,overwrite_a,overwrite_b])
    		Constructing wrapper function "cposv"...
    		  c,x,info = cposv(a,b,[lower,overwrite_a,overwrite_b])
    		Constructing wrapper function "zposv"...
    		  c,x,info = zposv(a,b,[lower,overwrite_a,overwrite_b])
    		Constructing wrapper function "spotrf"...
    		  c,info = spotrf(a,[lower,clean,overwrite_a])
    		Constructing wrapper function "dpotrf"...
    		  c,info = dpotrf(a,[lower,clean,overwrite_a])
    		Constructing wrapper function "cpotrf"...
    		  c,info = cpotrf(a,[lower,clean,overwrite_a])
    		Constructing wrapper function "zpotrf"...
    		  c,info = zpotrf(a,[lower,clean,overwrite_a])
    		Constructing wrapper function "spotrs"...
    		  x,info = spotrs(c,b,[lower,overwrite_b])
    		Constructing wrapper function "dpotrs"...
    		  x,info = dpotrs(c,b,[lower,overwrite_b])
    		Constructing wrapper function "cpotrs"...
    		  x,info = cpotrs(c,b,[lower,overwrite_b])
    		Constructing wrapper function "zpotrs"...
    		  x,info = zpotrs(c,b,[lower,overwrite_b])
    		Constructing wrapper function "spotri"...
    		  inv_a,info = spotri(c,[lower,overwrite_c])
    		Constructing wrapper function "dpotri"...
    		  inv_a,info = dpotri(c,[lower,overwrite_c])
    		Constructing wrapper function "cpotri"...
    		  inv_a,info = cpotri(c,[lower,overwrite_c])
    		Constructing wrapper function "zpotri"...
    		  inv_a,info = zpotri(c,[lower,overwrite_c])
    		Constructing wrapper function "slauum"...
    		  a,info = slauum(c,[lower,overwrite_c])
    		Constructing wrapper function "dlauum"...
    		  a,info = dlauum(c,[lower,overwrite_c])
    		Constructing wrapper function "clauum"...
    		  a,info = clauum(c,[lower,overwrite_c])
    		Constructing wrapper function "zlauum"...
    		  a,info = zlauum(c,[lower,overwrite_c])
    		Constructing wrapper function "strtri"...
    		  inv_c,info = strtri(c,[lower,unitdiag,overwrite_c])
    		Constructing wrapper function "dtrtri"...
    		  inv_c,info = dtrtri(c,[lower,unitdiag,overwrite_c])
    		Constructing wrapper function "ctrtri"...
    		  inv_c,info = ctrtri(c,[lower,unitdiag,overwrite_c])
    		Constructing wrapper function "ztrtri"...
    		  inv_c,info = ztrtri(c,[lower,unitdiag,overwrite_c])
    		Constructing wrapper function "strsyl"...
    		  x,scale,info = strsyl(a,b,c,[trana,tranb,isgn,overwrite_c])
    		Constructing wrapper function "dtrsyl"...
    		  x,scale,info = dtrsyl(a,b,c,[trana,tranb,isgn,overwrite_c])
    		Constructing wrapper function "ctrsyl"...
    		  x,scale,info = ctrsyl(a,b,c,[trana,tranb,isgn,overwrite_c])
    		Constructing wrapper function "ztrsyl"...
    		  x,scale,info = ztrsyl(a,b,c,[trana,tranb,isgn,overwrite_c])
    		Constructing wrapper function "slaswp"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  a = slaswp(a,piv,[k1,k2,off,inc,overwrite_a])
    		Constructing wrapper function "dlaswp"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  a = dlaswp(a,piv,[k1,k2,off,inc,overwrite_a])
    		Constructing wrapper function "claswp"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  a = claswp(a,piv,[k1,k2,off,inc,overwrite_a])
    		Constructing wrapper function "zlaswp"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  a = zlaswp(a,piv,[k1,k2,off,inc,overwrite_a])
    		Constructing wrapper function "cgees"...
    		  t,sdim,w,vs,work,info = cgees(cselect,a,[compute_v,sort_t,lwork,cselect_extra_args,overwrite_a])
    		Constructing wrapper function "zgees"...
    		  t,sdim,w,vs,work,info = zgees(zselect,a,[compute_v,sort_t,lwork,zselect_extra_args,overwrite_a])
    		Constructing wrapper function "sgees"...
    		  t,sdim,wr,wi,vs,work,info = sgees(sselect,a,[compute_v,sort_t,lwork,sselect_extra_args,overwrite_a])
    		Constructing wrapper function "dgees"...
    		  t,sdim,wr,wi,vs,work,info = dgees(dselect,a,[compute_v,sort_t,lwork,dselect_extra_args,overwrite_a])
    		Constructing wrapper function "sggev"...
    		  alphar,alphai,beta,vl,vr,work,info = sggev(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "dggev"...
    		  alphar,alphai,beta,vl,vr,work,info = dggev(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "cggev"...
    		  alpha,beta,vl,vr,work,info = cggev(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "zggev"...
    		  alpha,beta,vl,vr,work,info = zggev(a,b,[compute_vl,compute_vr,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "ssbev"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  w,z,info = ssbev(ab,[compute_v,lower,ldab,overwrite_ab])
    		Constructing wrapper function "dsbev"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  w,z,info = dsbev(ab,[compute_v,lower,ldab,overwrite_ab])
    		Constructing wrapper function "ssbevd"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  w,z,info = ssbevd(ab,[compute_v,lower,ldab,liwork,overwrite_ab])
    		Constructing wrapper function "dsbevd"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  w,z,info = dsbevd(ab,[compute_v,lower,ldab,liwork,overwrite_ab])
    		Constructing wrapper function "ssbevx"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  w,z,m,ifail,info = ssbevx(ab,vl,vu,il,iu,[ldab,compute_v,range,lower,abstol,mmax,overwrite_ab])
    		Constructing wrapper function "dsbevx"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  w,z,m,ifail,info = dsbevx(ab,vl,vu,il,iu,[ldab,compute_v,range,lower,abstol,mmax,overwrite_ab])
    		Constructing wrapper function "chbevd"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  w,z,info = chbevd(ab,[compute_v,lower,ldab,lrwork,liwork,overwrite_ab])
    		Constructing wrapper function "zhbevd"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  w,z,info = zhbevd(ab,[compute_v,lower,ldab,lrwork,liwork,overwrite_ab])
    		Constructing wrapper function "chbevx"...
    warning: callstatement is defined without callprotoargument
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  w,z,m,ifail,info = chbevx(ab,vl,vu,il,iu,[ldab,compute_v,range,lower,abstol,mmax,overwrite_ab])
    		Constructing wrapper function "zhbevx"...
    warning: callstatement is defined without callprotoargument
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  w,z,m,ifail,info = zhbevx(ab,vl,vu,il,iu,[ldab,compute_v,range,lower,abstol,mmax,overwrite_ab])
    		Creating wrapper for Fortran function "dlamch"("dlamch")...
    		Constructing wrapper function "dlamch"...
    		  dlamch = dlamch(cmach)
    		Creating wrapper for Fortran function "slamch"("wslamch")...
    		Constructing wrapper function "slamch"...
    		  slamch = slamch(cmach)
    		Constructing wrapper function "sgbtrf"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  lu,ipiv,info = sgbtrf(ab,kl,ku,[m,n,ldab,overwrite_ab])
    		Constructing wrapper function "dgbtrf"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  lu,ipiv,info = dgbtrf(ab,kl,ku,[m,n,ldab,overwrite_ab])
    		Constructing wrapper function "cgbtrf"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  lu,ipiv,info = cgbtrf(ab,kl,ku,[m,n,ldab,overwrite_ab])
    		Constructing wrapper function "zgbtrf"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  lu,ipiv,info = zgbtrf(ab,kl,ku,[m,n,ldab,overwrite_ab])
    		Constructing wrapper function "sgbtrs"...
    warning: callstatement is defined without callprotoargument
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,info = sgbtrs(ab,kl,ku,b,ipiv,[trans,n,ldab,ldb,overwrite_b])
    		Constructing wrapper function "dgbtrs"...
    warning: callstatement is defined without callprotoargument
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,info = dgbtrs(ab,kl,ku,b,ipiv,[trans,n,ldab,ldb,overwrite_b])
    		Constructing wrapper function "cgbtrs"...
    warning: callstatement is defined without callprotoargument
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,info = cgbtrs(ab,kl,ku,b,ipiv,[trans,n,ldab,ldb,overwrite_b])
    		Constructing wrapper function "zgbtrs"...
    warning: callstatement is defined without callprotoargument
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,info = zgbtrs(ab,kl,ku,b,ipiv,[trans,n,ldab,ldb,overwrite_b])
    		Constructing wrapper function "ssyevr"...
    		  w,z,info = ssyevr(a,[jobz,range,uplo,il,iu,lwork,overwrite_a])
    		Constructing wrapper function "dsyevr"...
    		  w,z,info = dsyevr(a,[jobz,range,uplo,il,iu,lwork,overwrite_a])
    		Constructing wrapper function "cheevr"...
    		  w,z,info = cheevr(a,[jobz,range,uplo,il,iu,lwork,overwrite_a])
    		Constructing wrapper function "zheevr"...
    		  w,z,info = zheevr(a,[jobz,range,uplo,il,iu,lwork,overwrite_a])
    		Constructing wrapper function "ssygv"...
    		  a,w,info = ssygv(a,b,[itype,jobz,uplo,overwrite_a,overwrite_b])
    		Constructing wrapper function "dsygv"...
    		  a,w,info = dsygv(a,b,[itype,jobz,uplo,overwrite_a,overwrite_b])
    		Constructing wrapper function "chegv"...
    		  a,w,info = chegv(a,b,[itype,jobz,uplo,overwrite_a,overwrite_b])
    		Constructing wrapper function "zhegv"...
    		  a,w,info = zhegv(a,b,[itype,jobz,uplo,overwrite_a,overwrite_b])
    		Constructing wrapper function "ssygvd"...
    		  a,w,info = ssygvd(a,b,[itype,jobz,uplo,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "dsygvd"...
    		  a,w,info = dsygvd(a,b,[itype,jobz,uplo,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "chegvd"...
    		  a,w,info = chegvd(a,b,[itype,jobz,uplo,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "zhegvd"...
    		  a,w,info = zhegvd(a,b,[itype,jobz,uplo,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "ssygvx"...
    		  w,z,ifail,info = ssygvx(a,b,iu,[itype,jobz,uplo,il,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "dsygvx"...
    		  w,z,ifail,info = dsygvx(a,b,iu,[itype,jobz,uplo,il,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "chegvx"...
    		  w,z,ifail,info = chegvx(a,b,iu,[itype,jobz,uplo,il,lwork,overwrite_a,overwrite_b])
    		Constructing wrapper function "zhegvx"...
    		  w,z,ifail,info = zhegvx(a,b,iu,[itype,jobz,uplo,il,lwork,overwrite_a,overwrite_b])
    	Wrote C/API module "_flapack" to file "build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c"
    	Fortran 77 wrappers are saved to "build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapack-f2pywrappers.f"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
      adding 'build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapack-f2pywrappers.f' to sources.
    building extension "scipy.linalg._cblas" sources
    from_template:> build/src.linux-x86_64-2.7/scipy/linalg/cblas.pyf
    Including file scipy/linalg/cblas_l1.pyf.src
    f2py options: []
    f2py: build/src.linux-x86_64-2.7/scipy/linalg/cblas.pyf
    Reading fortran codes...
    	Reading file 'build/src.linux-x86_64-2.7/scipy/linalg/cblas.pyf' (format:free)
    Line #33 in build/src.linux-x86_64-2.7/scipy/linalg/cblas.pyf:"  intent(c)"
    	All arguments will have attribute intent(c)
    Line #57 in build/src.linux-x86_64-2.7/scipy/linalg/cblas.pyf:"  intent(c)"
    	All arguments will have attribute intent(c)
    Line #81 in build/src.linux-x86_64-2.7/scipy/linalg/cblas.pyf:"  intent(c)"
    	All arguments will have attribute intent(c)
    Line #105 in build/src.linux-x86_64-2.7/scipy/linalg/cblas.pyf:"  intent(c)"
    	All arguments will have attribute intent(c)
    Post-processing...
    	Block: _cblas
    			Block: saxpy
    			Block: daxpy
    			Block: caxpy
    			Block: zaxpy
    Post-processing (stage 2)...
    Building modules...
    	Building module "_cblas"...
    		Constructing wrapper function "saxpy"...
    		  z = saxpy(x,y,[n,a,incx,incy,overwrite_y])
    		Constructing wrapper function "daxpy"...
    		  z = daxpy(x,y,[n,a,incx,incy,overwrite_y])
    		Constructing wrapper function "caxpy"...
    		  z = caxpy(x,y,[n,a,incx,incy,overwrite_y])
    		Constructing wrapper function "zaxpy"...
    		  z = zaxpy(x,y,[n,a,incx,incy,overwrite_y])
    	Wrote C/API module "_cblas" to file "build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_cblasmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.linalg._clapack" sources
    from_template:> build/src.linux-x86_64-2.7/scipy/linalg/clapack.pyf
    f2py options: []
    f2py: build/src.linux-x86_64-2.7/scipy/linalg/clapack.pyf
    Reading fortran codes...
    	Reading file 'build/src.linux-x86_64-2.7/scipy/linalg/clapack.pyf' (format:free)
    Post-processing...
    	Block: _clapack
    			Block: sgesv
    			Block: dgesv
    			Block: cgesv
    			Block: zgesv
    			Block: sposv
    			Block: dposv
    			Block: cposv
    			Block: zposv
    			Block: spotrf
    			Block: dpotrf
    			Block: cpotrf
    			Block: zpotrf
    			Block: spotrs
    			Block: dpotrs
    			Block: cpotrs
    			Block: zpotrs
    			Block: spotri
    			Block: dpotri
    			Block: cpotri
    			Block: zpotri
    			Block: slauum
    			Block: dlauum
    			Block: clauum
    			Block: zlauum
    			Block: strtri
    			Block: dtrtri
    			Block: ctrtri
    			Block: ztrtri
    Post-processing (stage 2)...
    Building modules...
    	Building module "_clapack"...
    		Constructing wrapper function "sgesv"...
    		  lu,piv,x,info = sgesv(a,b,[rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "dgesv"...
    		  lu,piv,x,info = dgesv(a,b,[rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "cgesv"...
    		  lu,piv,x,info = cgesv(a,b,[rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "zgesv"...
    		  lu,piv,x,info = zgesv(a,b,[rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "sposv"...
    		  c,x,info = sposv(a,b,[lower,rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "dposv"...
    		  c,x,info = dposv(a,b,[lower,rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "cposv"...
    		  c,x,info = cposv(a,b,[lower,rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "zposv"...
    		  c,x,info = zposv(a,b,[lower,rowmajor,overwrite_a,overwrite_b])
    		Constructing wrapper function "spotrf"...
    		  c,info = spotrf(a,[lower,clean,rowmajor,overwrite_a])
    		Constructing wrapper function "dpotrf"...
    		  c,info = dpotrf(a,[lower,clean,rowmajor,overwrite_a])
    		Constructing wrapper function "cpotrf"...
    		  c,info = cpotrf(a,[lower,clean,rowmajor,overwrite_a])
    		Constructing wrapper function "zpotrf"...
    		  c,info = zpotrf(a,[lower,clean,rowmajor,overwrite_a])
    		Constructing wrapper function "spotrs"...
    		  x,info = spotrs(c,b,[lower,rowmajor,overwrite_b])
    		Constructing wrapper function "dpotrs"...
    		  x,info = dpotrs(c,b,[lower,rowmajor,overwrite_b])
    		Constructing wrapper function "cpotrs"...
    		  x,info = cpotrs(c,b,[lower,rowmajor,overwrite_b])
    		Constructing wrapper function "zpotrs"...
    		  x,info = zpotrs(c,b,[lower,rowmajor,overwrite_b])
    		Constructing wrapper function "spotri"...
    		  inv_a,info = spotri(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "dpotri"...
    		  inv_a,info = dpotri(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "cpotri"...
    		  inv_a,info = cpotri(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "zpotri"...
    		  inv_a,info = zpotri(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "slauum"...
    		  a,info = slauum(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "dlauum"...
    		  a,info = dlauum(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "clauum"...
    		  a,info = clauum(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "zlauum"...
    		  a,info = zlauum(c,[lower,rowmajor,overwrite_c])
    		Constructing wrapper function "strtri"...
    		  inv_c,info = strtri(c,[lower,unitdiag,rowmajor,overwrite_c])
    		Constructing wrapper function "dtrtri"...
    		  inv_c,info = dtrtri(c,[lower,unitdiag,rowmajor,overwrite_c])
    		Constructing wrapper function "ctrtri"...
    		  inv_c,info = ctrtri(c,[lower,unitdiag,rowmajor,overwrite_c])
    		Constructing wrapper function "ztrtri"...
    		  inv_c,info = ztrtri(c,[lower,unitdiag,rowmajor,overwrite_c])
    	Wrote C/API module "_clapack" to file "build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.linalg._flinalg" sources
    f2py options: []
    f2py:> build/src.linux-x86_64-2.7/scipy/linalg/_flinalgmodule.c
    Reading fortran codes...
    	Reading file 'scipy/linalg/src/det.f' (format:fix,strict)
    	Reading file 'scipy/linalg/src/lu.f' (format:fix,strict)
    Post-processing...
    	Block: _flinalg
    {'attrspec': ['intent(out)']}
    In: :_flinalg:scipy/linalg/src/det.f:ddet_c
    vars2fortran: No typespec for argument "info".
    			Block: ddet_c
    {'attrspec': ['intent(out)']}
    In: :_flinalg:scipy/linalg/src/det.f:ddet_r
    vars2fortran: No typespec for argument "info".
    			Block: ddet_r
    {'attrspec': ['intent(out)']}
    In: :_flinalg:scipy/linalg/src/det.f:sdet_c
    vars2fortran: No typespec for argument "info".
    			Block: sdet_c
    {'attrspec': ['intent(out)']}
    In: :_flinalg:scipy/linalg/src/det.f:sdet_r
    vars2fortran: No typespec for argument "info".
    			Block: sdet_r
    {'attrspec': ['intent(out)']}
    In: :_flinalg:scipy/linalg/src/det.f:zdet_c
    vars2fortran: No typespec for argument "info".
    			Block: zdet_c
    {'attrspec': ['intent(out)']}
    In: :_flinalg:scipy/linalg/src/det.f:zdet_r
    vars2fortran: No typespec for argument "info".
    			Block: zdet_r
    {'attrspec': ['intent(out)']}
    In: :_flinalg:scipy/linalg/src/det.f:cdet_c
    vars2fortran: No typespec for argument "info".
    			Block: cdet_c
    {'attrspec': ['intent(out)']}
    In: :_flinalg:scipy/linalg/src/det.f:cdet_r
    vars2fortran: No typespec for argument "info".
    			Block: cdet_r
    {'attrspec': ['intent(out)']}
    In: :_flinalg:scipy/linalg/src/lu.f:dlu_c
    vars2fortran: No typespec for argument "info".
    			Block: dlu_c
    {'attrspec': ['intent(out)']}
    In: :_flinalg:scipy/linalg/src/lu.f:zlu_c
    vars2fortran: No typespec for argument "info".
    			Block: zlu_c
    {'attrspec': ['intent(out)']}
    In: :_flinalg:scipy/linalg/src/lu.f:slu_c
    vars2fortran: No typespec for argument "info".
    			Block: slu_c
    {'attrspec': ['intent(out)']}
    In: :_flinalg:scipy/linalg/src/lu.f:clu_c
    vars2fortran: No typespec for argument "info".
    			Block: clu_c
    Post-processing (stage 2)...
    Building modules...
    	Building module "_flinalg"...
    		Constructing wrapper function "ddet_c"...
    		  det,info = ddet_c(a,[overwrite_a])
    		Constructing wrapper function "ddet_r"...
    		  det,info = ddet_r(a,[overwrite_a])
    		Constructing wrapper function "sdet_c"...
    		  det,info = sdet_c(a,[overwrite_a])
    		Constructing wrapper function "sdet_r"...
    		  det,info = sdet_r(a,[overwrite_a])
    		Constructing wrapper function "zdet_c"...
    		  det,info = zdet_c(a,[overwrite_a])
    		Constructing wrapper function "zdet_r"...
    		  det,info = zdet_r(a,[overwrite_a])
    		Constructing wrapper function "cdet_c"...
    		  det,info = cdet_c(a,[overwrite_a])
    		Constructing wrapper function "cdet_r"...
    		  det,info = cdet_r(a,[overwrite_a])
    		Constructing wrapper function "dlu_c"...
    		  p,l,u,info = dlu_c(a,[permute_l,overwrite_a])
    		Constructing wrapper function "zlu_c"...
    		  p,l,u,info = zlu_c(a,[permute_l,overwrite_a])
    		Constructing wrapper function "slu_c"...
    		  p,l,u,info = slu_c(a,[permute_l,overwrite_a])
    		Constructing wrapper function "clu_c"...
    		  p,l,u,info = clu_c(a,[permute_l,overwrite_a])
    	Wrote C/API module "_flinalg" to file "build/src.linux-x86_64-2.7/scipy/linalg/_flinalgmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.linalg.calc_lwork" sources
    f2py options: []
    f2py:> build/src.linux-x86_64-2.7/scipy/linalg/calc_lworkmodule.c
    Reading fortran codes...
    	Reading file 'scipy/linalg/src/calc_lwork.f' (format:fix,strict)
    Post-processing...
    	Block: calc_lwork
    			Block: gehrd
    			Block: gesdd
    			Block: gelss
    			Block: getri
    			Block: geev
    			Block: heev
    			Block: syev
    			Block: gees
    			Block: geqrf
    			Block: gqr
    Post-processing (stage 2)...
    Building modules...
    	Building module "calc_lwork"...
    		Constructing wrapper function "gehrd"...
    		  minwrk,maxwrk = gehrd(prefix,n,lo,hi)
    		Constructing wrapper function "gesdd"...
    		  minwrk,maxwrk = gesdd(prefix,m,n,compute_uv)
    		Constructing wrapper function "gelss"...
    		  minwrk,maxwrk = gelss(prefix,m,n,nrhs)
    		Constructing wrapper function "getri"...
    		  minwrk,maxwrk = getri(prefix,n)
    		Constructing wrapper function "geev"...
    		  minwrk,maxwrk = geev(prefix,n,[compute_vl,compute_vr])
    		Constructing wrapper function "heev"...
    		  minwrk,maxwrk = heev(prefix,n,[lower])
    		Constructing wrapper function "syev"...
    		  minwrk,maxwrk = syev(prefix,n,[lower])
    		Constructing wrapper function "gees"...
    		  minwrk,maxwrk = gees(prefix,n,[compute_v])
    		Constructing wrapper function "geqrf"...
    		  minwrk,maxwrk = geqrf(prefix,m,n)
    		Constructing wrapper function "gqr"...
    		  minwrk,maxwrk = gqr(prefix,m,n)
    	Wrote C/API module "calc_lwork" to file "build/src.linux-x86_64-2.7/scipy/linalg/calc_lworkmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.linalg._interpolative" sources
    f2py options: []
    f2py: scipy/linalg/interpolative.pyf
    "object of type 'type' has no len()" in evaluating 'len(list)' (available names: [])
    "object of type 'type' has no len()" in evaluating 'len(list)' (available names: [])
    "object of type 'type' has no len()" in evaluating 'len(list)' (available names: [])
    "object of type 'type' has no len()" in evaluating 'len(list)' (available names: [])
    "object of type 'type' has no len()" in evaluating 'len(list)' (available names: [])
    "object of type 'type' has no len()" in evaluating 'len(list)' (available names: [])
    Reading fortran codes...
    	Reading file 'scipy/linalg/interpolative.pyf' (format:free)
    Post-processing...
    	Block: _interpolative
    			Block: id_srand
    			Block: idd_frm
    			Block: idd_sfrm
    			Block: idd_frmi
    			Block: idd_sfrmi
    			Block: iddp_id
    			Block: iddr_id
    			Block: idd_reconid
    			Block: idd_reconint
    			Block: idd_copycols
    			Block: idd_id2svd
    			Block: idd_snorm
    In: scipy/linalg/interpolative.pyf:_interpolative:unknown_interface:idd_snorm
    get_useparameters: no module idd__user__routines info used by idd_snorm
    			Block: idd_diffsnorm
    In: scipy/linalg/interpolative.pyf:_interpolative:unknown_interface:idd_diffsnorm
    get_useparameters: no module idd__user__routines info used by idd_diffsnorm
    			Block: iddr_svd
    			Block: iddp_svd
    			Block: iddp_aid
    			Block: idd_estrank
    			Block: iddp_asvd
    			Block: iddp_rid
    In: scipy/linalg/interpolative.pyf:_interpolative:unknown_interface:iddp_rid
    get_useparameters: no module idd__user__routines info used by iddp_rid
    			Block: idd_findrank
    In: scipy/linalg/interpolative.pyf:_interpolative:unknown_interface:idd_findrank
    get_useparameters: no module idd__user__routines info used by idd_findrank
    			Block: iddp_rsvd
    In: scipy/linalg/interpolative.pyf:_interpolative:unknown_interface:iddp_rsvd
    get_useparameters: no module idd__user__routines info used by iddp_rsvd
    			Block: iddr_aid
    			Block: iddr_aidi
    			Block: iddr_asvd
    			Block: iddr_rid
    In: scipy/linalg/interpolative.pyf:_interpolative:unknown_interface:iddr_rid
    get_useparameters: no module idd__user__routines info used by iddr_rid
    			Block: iddr_rsvd
    In: scipy/linalg/interpolative.pyf:_interpolative:unknown_interface:iddr_rsvd
    get_useparameters: no module idd__user__routines info used by iddr_rsvd
    			Block: idz_frm
    			Block: idz_sfrm
    			Block: idz_frmi
    			Block: idz_sfrmi
    			Block: idzp_id
    			Block: idzr_id
    			Block: idz_reconid
    			Block: idz_reconint
    			Block: idz_copycols
    			Block: idz_id2svd
    			Block: idz_snorm
    In: scipy/linalg/interpolative.pyf:_interpolative:unknown_interface:idz_snorm
    get_useparameters: no module idz__user__routines info used by idz_snorm
    			Block: idz_diffsnorm
    In: scipy/linalg/interpolative.pyf:_interpolative:unknown_interface:idz_diffsnorm
    get_useparameters: no module idz__user__routines info used by idz_diffsnorm
    			Block: idzr_svd
    			Block: idzp_svd
    			Block: idzp_aid
    			Block: idz_estrank
    			Block: idzp_asvd
    			Block: idzp_rid
    In: scipy/linalg/interpolative.pyf:_interpolative:unknown_interface:idzp_rid
    get_useparameters: no module idz__user__routines info used by idzp_rid
    			Block: idz_findrank
    In: scipy/linalg/interpolative.pyf:_interpolative:unknown_interface:idz_findrank
    get_useparameters: no module idz__user__routines info used by idz_findrank
    			Block: idzp_rsvd
    In: scipy/linalg/interpolative.pyf:_interpolative:unknown_interface:idzp_rsvd
    get_useparameters: no module idz__user__routines info used by idzp_rsvd
    			Block: idzr_aid
    			Block: idzr_aidi
    			Block: idzr_asvd
    			Block: idzr_rid
    In: scipy/linalg/interpolative.pyf:_interpolative:unknown_interface:idzr_rid
    get_useparameters: no module idz__user__routines info used by idzr_rid
    			Block: idzr_rsvd
    In: scipy/linalg/interpolative.pyf:_interpolative:unknown_interface:idzr_rsvd
    get_useparameters: no module idz__user__routines info used by idzr_rsvd
    	Block: idd__user__routines
    		Block: idd_user_interface
    			Block: matvect
    			Block: matvec
    			Block: matvect2
    			Block: matvec2
    	Block: idz__user__routines
    		Block: idz_user_interface
    			Block: matveca
    			Block: matvec
    			Block: matveca2
    			Block: matvec2
    Post-processing (stage 2)...
    Building modules...
    	Constructing call-back function "cb_matvect_in_idd__user__routines"
    	  def matvect(x,[m,n,p1,p2,p3,p4]): return y
    	Constructing call-back function "cb_matvec_in_idd__user__routines"
    	  def matvec(x,[n,m,p1,p2,p3,p4]): return y
    	Constructing call-back function "cb_matvect2_in_idd__user__routines"
    	  def matvect2(x,[m,n,p1,p2,p3,p4]): return y
    	Constructing call-back function "cb_matvec2_in_idd__user__routines"
    	  def matvec2(x,[n,m,p1,p2,p3,p4]): return y
    	Constructing call-back function "cb_matveca_in_idz__user__routines"
    	  def matveca(x,[m,n,p1,p2,p3,p4]): return y
    	Constructing call-back function "cb_matvec_in_idz__user__routines"
    	  def matvec(x,[n,m,p1,p2,p3,p4]): return y
    	Constructing call-back function "cb_matveca2_in_idz__user__routines"
    	  def matveca2(x,[m,n,p1,p2,p3,p4]): return y
    	Constructing call-back function "cb_matvec2_in_idz__user__routines"
    	  def matvec2(x,[n,m,p1,p2,p3,p4]): return y
    	Building module "_interpolative"...
    		Constructing wrapper function "id_srand"...
    		  r = id_srand(n)
    		Constructing wrapper function "id_srandi"...
    		  id_srandi(t)
    		Constructing wrapper function "id_srando"...
    		  id_srando()
    		Constructing wrapper function "idd_frm"...
    		  y = idd_frm(n,w,x,[m])
    		Constructing wrapper function "idd_sfrm"...
    		  y = idd_sfrm(l,n,w,x,[m])
    		Constructing wrapper function "idd_frmi"...
    		  n,w = idd_frmi(m)
    		Constructing wrapper function "idd_sfrmi"...
    		  n,w = idd_sfrmi(l,m)
    		Constructing wrapper function "iddp_id"...
    		  krank,list,rnorms = iddp_id(eps,a,[m,n])
    		Constructing wrapper function "iddr_id"...
    		  list,rnorms = iddr_id(a,krank,[m,n])
    		Constructing wrapper function "idd_reconid"...
    		  approx = idd_reconid(col,list,proj,[m,krank,n])
    		Constructing wrapper function "idd_reconint"...
    		  p = idd_reconint(list,proj,[n,krank])
    		Constructing wrapper function "idd_copycols"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  col = idd_copycols(a,krank,list,[m,n])
    		Constructing wrapper function "idd_id2svd"...
    		  u,v,s,ier = idd_id2svd(b,list,proj,[m,krank,n,w])
    		Constructing wrapper function "idd_snorm"...
    		  snorm,v = idd_snorm(m,n,matvect,matvec,its,[p1t,p2t,p3t,p4t,p1,p2,p3,p4,u,matvect_extra_args,matvec_extra_args])
    		Constructing wrapper function "idd_diffsnorm"...
    		  snorm = idd_diffsnorm(m,n,matvect,matvect2,matvec,matvec2,its,[p1t,p2t,p3t,p4t,p1t2,p2t2,p3t2,p4t2,p1,p2,p3,p4,p12,p22,p32,p42,w,matvect_extra_args,matvect2_extra_args,matvec_extra_args,matvec2_extra_args])
    		Constructing wrapper function "iddr_svd"...
    		  u,v,s,ier = iddr_svd(a,krank,[m,n,r])
    		Constructing wrapper function "iddp_svd"...
    		  krank,iu,iv,is,w,ier = iddp_svd(eps,a,[m,n])
    		Constructing wrapper function "iddp_aid"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  krank,list,proj = iddp_aid(eps,a,work,proj,[m,n])
    		Constructing wrapper function "idd_estrank"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  krank,ra = idd_estrank(eps,a,w,ra,[m,n])
    		Constructing wrapper function "iddp_asvd"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  krank,iu,iv,is,w,ier = iddp_asvd(eps,a,winit,w,[m,n])
    		Constructing wrapper function "iddp_rid"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  krank,list,proj,ier = iddp_rid(eps,m,n,matvect,proj,[p1,p2,p3,p4,matvect_extra_args])
    		Constructing wrapper function "idd_findrank"...
    		  krank,ra,ier = idd_findrank(eps,m,n,matvect,[p1,p2,p3,p4,w,matvect_extra_args])
    		Constructing wrapper function "iddp_rsvd"...
    		  krank,iu,iv,is,w,ier = iddp_rsvd(eps,m,n,matvect,matvec,[p1t,p2t,p3t,p4t,p1,p2,p3,p4,matvect_extra_args,matvec_extra_args])
    		Constructing wrapper function "iddr_aid"...
    		  list,proj = iddr_aid(a,krank,w,[m,n])
    		Constructing wrapper function "iddr_aidi"...
    		  w = iddr_aidi(m,n,krank)
    		Constructing wrapper function "iddr_asvd"...
    		  u,v,s,ier = iddr_asvd(a,krank,w,[m,n])
    		Constructing wrapper function "iddr_rid"...
    		  list,proj = iddr_rid(m,n,matvect,krank,[p1,p2,p3,p4,matvect_extra_args])
    		Constructing wrapper function "iddr_rsvd"...
    		  u,v,s,ier = iddr_rsvd(m,n,matvect,matvec,krank,[p1t,p2t,p3t,p4t,p1,p2,p3,p4,w,matvect_extra_args,matvec_extra_args])
    		Constructing wrapper function "idz_frm"...
    		  y = idz_frm(n,w,x,[m])
    		Constructing wrapper function "idz_sfrm"...
    		  y = idz_sfrm(l,n,w,x,[m])
    		Constructing wrapper function "idz_frmi"...
    		  n,w = idz_frmi(m)
    		Constructing wrapper function "idz_sfrmi"...
    		  n,w = idz_sfrmi(l,m)
    		Constructing wrapper function "idzp_id"...
    		  krank,list,rnorms = idzp_id(eps,a,[m,n])
    		Constructing wrapper function "idzr_id"...
    		  list,rnorms = idzr_id(a,krank,[m,n])
    		Constructing wrapper function "idz_reconid"...
    		  approx = idz_reconid(col,list,proj,[m,krank,n])
    		Constructing wrapper function "idz_reconint"...
    		  p = idz_reconint(list,proj,[n,krank])
    		Constructing wrapper function "idz_copycols"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  col = idz_copycols(a,krank,list,[m,n])
    		Constructing wrapper function "idz_id2svd"...
    		  u,v,s,ier = idz_id2svd(b,list,proj,[m,krank,n,w])
    		Constructing wrapper function "idz_snorm"...
    		  snorm,v = idz_snorm(m,n,matveca,matvec,its,[p1a,p2a,p3a,p4a,p1,p2,p3,p4,u,matveca_extra_args,matvec_extra_args])
    		Constructing wrapper function "idz_diffsnorm"...
    		  snorm = idz_diffsnorm(m,n,matveca,matveca2,matvec,matvec2,its,[p1a,p2a,p3a,p4a,p1a2,p2a2,p3a2,p4a2,p1,p2,p3,p4,p12,p22,p32,p42,w,matveca_extra_args,matveca2_extra_args,matvec_extra_args,matvec2_extra_args])
    		Constructing wrapper function "idzr_svd"...
    		  u,v,s,ier = idzr_svd(a,krank,[m,n,r])
    		Constructing wrapper function "idzp_svd"...
    		  krank,iu,iv,is,w,ier = idzp_svd(eps,a,[m,n])
    		Constructing wrapper function "idzp_aid"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  krank,list,proj = idzp_aid(eps,a,work,proj,[m,n])
    		Constructing wrapper function "idz_estrank"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  krank,ra = idz_estrank(eps,a,w,ra,[m,n])
    		Constructing wrapper function "idzp_asvd"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  krank,iu,iv,is,w,ier = idzp_asvd(eps,a,winit,w,[m,n])
    		Constructing wrapper function "idzp_rid"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  krank,list,proj,ier = idzp_rid(eps,m,n,matveca,proj,[p1,p2,p3,p4,matveca_extra_args])
    		Constructing wrapper function "idz_findrank"...
    		  krank,ra,ier = idz_findrank(eps,m,n,matveca,[p1,p2,p3,p4,w,matveca_extra_args])
    		Constructing wrapper function "idzp_rsvd"...
    		  krank,iu,iv,is,w,ier = idzp_rsvd(eps,m,n,matveca,matvec,[p1a,p2a,p3a,p4a,p1,p2,p3,p4,matveca_extra_args,matvec_extra_args])
    		Constructing wrapper function "idzr_aid"...
    		  list,proj = idzr_aid(a,krank,w,[m,n])
    		Constructing wrapper function "idzr_aidi"...
    		  w = idzr_aidi(m,n,krank)
    		Constructing wrapper function "idzr_asvd"...
    		  u,v,s,ier = idzr_asvd(a,krank,w,[m,n])
    		Constructing wrapper function "idzr_rid"...
    		  list,proj = idzr_rid(m,n,matveca,krank,[p1,p2,p3,p4,matveca_extra_args])
    		Constructing wrapper function "idzr_rsvd"...
    		  u,v,s,ier = idzr_rsvd(m,n,matveca,matvec,krank,[p1a,p2a,p3a,p4a,p1,p2,p3,p4,w,matveca_extra_args,matvec_extra_args])
    	Wrote C/API module "_interpolative" to file "build/src.linux-x86_64-2.7/scipy/linalg/_interpolativemodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.odr.__odrpack" sources
    building extension "scipy.optimize._minpack" sources
    building extension "scipy.optimize._zeros" sources
    building extension "scipy.optimize._lbfgsb" sources
    f2py options: []
    f2py: scipy/optimize/lbfgsb/lbfgsb.pyf
    Reading fortran codes...
    	Reading file 'scipy/optimize/lbfgsb/lbfgsb.pyf' (format:free)
    Post-processing...
    	Block: _lbfgsb
    			Block: setulb
    Post-processing (stage 2)...
    Building modules...
    	Building module "_lbfgsb"...
    		Constructing wrapper function "setulb"...
    		  setulb(m,x,l,u,nbd,f,g,factr,pgtol,wa,iwa,task,iprint,csave,lsave,isave,dsave,[n])
    	Wrote C/API module "_lbfgsb" to file "build/src.linux-x86_64-2.7/scipy/optimize/lbfgsb/_lbfgsbmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.optimize.moduleTNC" sources
    building extension "scipy.optimize._cobyla" sources
    f2py options: []
    f2py: scipy/optimize/cobyla/cobyla.pyf
    Reading fortran codes...
    	Reading file 'scipy/optimize/cobyla/cobyla.pyf' (format:free)
    Post-processing...
    	Block: _cobyla__user__routines
    		Block: _cobyla_user_interface
    			Block: calcfc
    	Block: _cobyla
    			Block: minimize
    In: scipy/optimize/cobyla/cobyla.pyf:_cobyla:unknown_interface:minimize
    get_useparameters: no module _cobyla__user__routines info used by minimize
    Post-processing (stage 2)...
    Building modules...
    	Constructing call-back function "cb_calcfc_in__cobyla__user__routines"
    	  def calcfc(x,con): return f
    	Building module "_cobyla"...
    		Constructing wrapper function "minimize"...
    		  x,dinfo = minimize(calcfc,m,x,rhobeg,rhoend,dinfo,[iprint,maxfun,calcfc_extra_args])
    	Wrote C/API module "_cobyla" to file "build/src.linux-x86_64-2.7/scipy/optimize/cobyla/_cobylamodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.optimize.minpack2" sources
    f2py options: []
    f2py: scipy/optimize/minpack2/minpack2.pyf
    Reading fortran codes...
    	Reading file 'scipy/optimize/minpack2/minpack2.pyf' (format:free)
    Post-processing...
    	Block: minpack2
    			Block: dcsrch
    			Block: dcstep
    Post-processing (stage 2)...
    Building modules...
    	Building module "minpack2"...
    		Constructing wrapper function "dcsrch"...
    		  stp,f,g,task = dcsrch(stp,f,g,ftol,gtol,xtol,task,stpmin,stpmax,isave,dsave)
    		Constructing wrapper function "dcstep"...
    		  stx,fx,dx,sty,fy,dy,stp,brackt = dcstep(stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,stpmin,stpmax)
    	Wrote C/API module "minpack2" to file "build/src.linux-x86_64-2.7/scipy/optimize/minpack2/minpack2module.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.optimize._slsqp" sources
    f2py options: []
    f2py: scipy/optimize/slsqp/slsqp.pyf
    Reading fortran codes...
    	Reading file 'scipy/optimize/slsqp/slsqp.pyf' (format:free)
    Post-processing...
    	Block: _slsqp
    			Block: slsqp
    Post-processing (stage 2)...
    Building modules...
    	Building module "_slsqp"...
    		Constructing wrapper function "slsqp"...
    		  slsqp(m,meq,x,xl,xu,f,c,g,a,acc,iter,mode,w,jw,[la,n,l_w,l_jw])
    	Wrote C/API module "_slsqp" to file "build/src.linux-x86_64-2.7/scipy/optimize/slsqp/_slsqpmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.optimize._nnls" sources
    f2py options: []
    f2py: scipy/optimize/nnls/nnls.pyf
    Reading fortran codes...
    	Reading file 'scipy/optimize/nnls/nnls.pyf' (format:free)
    crackline: groupcounter=1 groupname={0: '', 1: 'python module', 2: 'interface', 3: 'subroutine'}
    crackline: Mismatch of blocks encountered. Trying to fix it by assuming "end" statement.
    Post-processing...
    	Block: _nnls
    			Block: nnls
    Post-processing (stage 2)...
    Building modules...
    	Building module "_nnls"...
    		Constructing wrapper function "nnls"...
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    getarrdims:warning: assumed shape array, using 0 instead of '*'
    		  x,rnorm,mode = nnls(a,m,n,b,w,zz,index_bn,[mda,overwrite_a,overwrite_b])
    	Wrote C/API module "_nnls" to file "build/src.linux-x86_64-2.7/scipy/optimize/nnls/_nnlsmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.signal.sigtools" sources
    conv_template:> build/src.linux-x86_64-2.7/scipy/signal/lfilter.c
    conv_template:> build/src.linux-x86_64-2.7/scipy/signal/correlate_nd.c
    building extension "scipy.signal._spectral" sources
    building extension "scipy.signal.spline" sources
    building extension "scipy.sparse.linalg.isolve._iterative" sources
    from_template:> build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/STOPTEST2.f
    from_template:> build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/getbreak.f
    from_template:> build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/BiCGREVCOM.f
    from_template:> build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/BiCGSTABREVCOM.f
    from_template:> build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/CGREVCOM.f
    from_template:> build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/CGSREVCOM.f
    from_template:> build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/GMRESREVCOM.f
    from_template:> build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/QMRREVCOM.f
    from_template:> build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/_iterative.pyf
    f2py options: []
    f2py: build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/_iterative.pyf
    Reading fortran codes...
    	Reading file 'build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/_iterative.pyf' (format:free)
    Post-processing...
    	Block: _iterative
    			Block: sbicgrevcom
    			Block: dbicgrevcom
    			Block: cbicgrevcom
    			Block: zbicgrevcom
    			Block: sbicgstabrevcom
    			Block: dbicgstabrevcom
    			Block: cbicgstabrevcom
    			Block: zbicgstabrevcom
    			Block: scgrevcom
    			Block: dcgrevcom
    			Block: ccgrevcom
    			Block: zcgrevcom
    			Block: scgsrevcom
    			Block: dcgsrevcom
    			Block: ccgsrevcom
    			Block: zcgsrevcom
    			Block: sqmrrevcom
    			Block: dqmrrevcom
    			Block: cqmrrevcom
    			Block: zqmrrevcom
    			Block: sgmresrevcom
    			Block: dgmresrevcom
    			Block: cgmresrevcom
    			Block: zgmresrevcom
    			Block: sstoptest2
    			Block: dstoptest2
    			Block: cstoptest2
    			Block: zstoptest2
    Post-processing (stage 2)...
    Building modules...
    	Building module "_iterative"...
    		Constructing wrapper function "sbicgrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = sbicgrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "dbicgrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = dbicgrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "cbicgrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = cbicgrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "zbicgrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = zbicgrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "sbicgstabrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = sbicgstabrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "dbicgstabrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = dbicgstabrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "cbicgstabrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = cbicgstabrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "zbicgstabrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = zbicgstabrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "scgrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = scgrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "dcgrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = dcgrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "ccgrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = ccgrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "zcgrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = zcgrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "scgsrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = scgsrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "dcgsrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = dcgsrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "ccgsrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = ccgsrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "zcgsrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = zcgsrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "sqmrrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = sqmrrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "dqmrrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = dqmrrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "cqmrrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = cqmrrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "zqmrrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = zqmrrevcom(b,x,work,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "sgmresrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = sgmresrevcom(b,x,restrt,work,work2,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "dgmresrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = dgmresrevcom(b,x,restrt,work,work2,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "cgmresrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = cgmresrevcom(b,x,restrt,work,work2,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "zgmresrevcom"...
    		  x,iter,resid,info,ndx1,ndx2,sclr1,sclr2,ijob = zgmresrevcom(b,x,restrt,work,work2,iter,resid,info,ndx1,ndx2,ijob)
    		Constructing wrapper function "sstoptest2"...
    		  bnrm2,resid,info = sstoptest2(r,b,bnrm2,tol,info)
    		Constructing wrapper function "dstoptest2"...
    		  bnrm2,resid,info = dstoptest2(r,b,bnrm2,tol,info)
    		Constructing wrapper function "cstoptest2"...
    		  bnrm2,resid,info = cstoptest2(r,b,bnrm2,tol,info)
    		Constructing wrapper function "zstoptest2"...
    		  bnrm2,resid,info = zstoptest2(r,b,bnrm2,tol,info)
    	Wrote C/API module "_iterative" to file "build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/_iterativemodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.sparse.linalg.dsolve._superlu" sources
    building extension "scipy.sparse.linalg.dsolve.umfpack.__umfpack" sources
      adding 'scipy/sparse/linalg/dsolve/umfpack/umfpack.i' to sources.
    swig: scipy/sparse/linalg/dsolve/umfpack/umfpack.i
    swig -python -I/usr/include/suitesparse -I/usr/include/suitesparse -I/usr/include/suitesparse -I/usr/include/atlas -o build/src.linux-x86_64-2.7/scipy/sparse/linalg/dsolve/umfpack/_umfpack_wrap.c -outdir build/src.linux-x86_64-2.7/scipy/sparse/linalg/dsolve/umfpack scipy/sparse/linalg/dsolve/umfpack/umfpack.i
    building extension "scipy.sparse.linalg.eigen.arpack._arpack" sources
    from_template:> build/src.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/arpack.pyf
    f2py options: []
    f2py: build/src.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/arpack.pyf
    Reading fortran codes...
    	Reading file 'build/src.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/arpack.pyf' (format:free)
    Line #5 in build/src.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/arpack.pyf:"    <_rd=real,double precision>"
    	crackline:1: No pattern for line
    Line #6 in build/src.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/arpack.pyf:"    <_cd=complex,double complex>"
    	crackline:1: No pattern for line
    Post-processing...
    	Block: _arpack
    			Block: ssaupd
    			Block: dsaupd
    			Block: sseupd
    			Block: dseupd
    			Block: snaupd
    			Block: dnaupd
    			Block: sneupd
    			Block: dneupd
    			Block: cnaupd
    			Block: znaupd
    			Block: cneupd
    			Block: zneupd
    Post-processing (stage 2)...
    Building modules...
    	Building module "_arpack"...
    		Constructing wrapper function "ssaupd"...
    		  ido,tol,resid,v,iparam,ipntr,info = ssaupd(ido,bmat,which,nev,tol,resid,v,iparam,ipntr,workd,workl,info,[n,ncv,ldv,lworkl])
    		Constructing wrapper function "dsaupd"...
    		  ido,tol,resid,v,iparam,ipntr,info = dsaupd(ido,bmat,which,nev,tol,resid,v,iparam,ipntr,workd,workl,info,[n,ncv,ldv,lworkl])
    		Constructing wrapper function "sseupd"...
    		  d,z,info = sseupd(rvec,howmny,select,sigma,bmat,which,nev,tol,resid,v,iparam,ipntr,workd,workl,info,[ldz,n,ncv,ldv,lworkl])
    		Constructing wrapper function "dseupd"...
    		  d,z,info = dseupd(rvec,howmny,select,sigma,bmat,which,nev,tol,resid,v,iparam,ipntr,workd,workl,info,[ldz,n,ncv,ldv,lworkl])
    		Constructing wrapper function "snaupd"...
    		  ido,tol,resid,v,iparam,ipntr,info = snaupd(ido,bmat,which,nev,tol,resid,v,iparam,ipntr,workd,workl,info,[n,ncv,ldv,lworkl])
    		Constructing wrapper function "dnaupd"...
    		  ido,tol,resid,v,iparam,ipntr,info = dnaupd(ido,bmat,which,nev,tol,resid,v,iparam,ipntr,workd,workl,info,[n,ncv,ldv,lworkl])
    		Constructing wrapper function "sneupd"...
    		  dr,di,z,info = sneupd(rvec,howmny,select,sigmar,sigmai,workev,bmat,which,nev,tol,resid,v,iparam,ipntr,workd,workl,info,[ldz,n,ncv,ldv,lworkl])
    		Constructing wrapper function "dneupd"...
    		  dr,di,z,info = dneupd(rvec,howmny,select,sigmar,sigmai,workev,bmat,which,nev,tol,resid,v,iparam,ipntr,workd,workl,info,[ldz,n,ncv,ldv,lworkl])
    		Constructing wrapper function "cnaupd"...
    		  ido,tol,resid,v,iparam,ipntr,info = cnaupd(ido,bmat,which,nev,tol,resid,v,iparam,ipntr,workd,workl,rwork,info,[n,ncv,ldv,lworkl])
    		Constructing wrapper function "znaupd"...
    		  ido,tol,resid,v,iparam,ipntr,info = znaupd(ido,bmat,which,nev,tol,resid,v,iparam,ipntr,workd,workl,rwork,info,[n,ncv,ldv,lworkl])
    		Constructing wrapper function "cneupd"...
    		  d,z,info = cneupd(rvec,howmny,select,sigma,workev,bmat,which,nev,tol,resid,v,iparam,ipntr,workd,workl,rwork,info,[ldz,n,ncv,ldv,lworkl])
    		Constructing wrapper function "zneupd"...
    		  d,z,info = zneupd(rvec,howmny,select,sigma,workev,bmat,which,nev,tol,resid,v,iparam,ipntr,workd,workl,rwork,info,[ldz,n,ncv,ldv,lworkl])
    		Constructing COMMON block support for "debug"...
    		  logfil,ndigit,mgetv0,msaupd,msaup2,msaitr,mseigt,msapps,msgets,mseupd,mnaupd,mnaup2,mnaitr,mneigh,mnapps,mngets,mneupd,mcaupd,mcaup2,mcaitr,mceigh,mcapps,mcgets,mceupd
    		Constructing COMMON block support for "timing"...
    		  nopx,nbx,nrorth,nitref,nrstrt,tsaupd,tsaup2,tsaitr,tseigt,tsgets,tsapps,tsconv,tnaupd,tnaup2,tnaitr,tneigh,tngets,tnapps,tnconv,tcaupd,tcaup2,tcaitr,tceigh,tcgets,tcapps,tcconv,tmvopx,tmvbx,tgetv0,titref,trvec
    	Wrote C/API module "_arpack" to file "build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/_arpackmodule.c"
    	Fortran 77 wrappers are saved to "build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/_arpack-f2pywrappers.f"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
      adding 'build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/_arpack-f2pywrappers.f' to sources.
    building extension "scipy.sparse.sparsetools._csr" sources
    building extension "scipy.sparse.sparsetools._csc" sources
    building extension "scipy.sparse.sparsetools._coo" sources
    building extension "scipy.sparse.sparsetools._bsr" sources
    building extension "scipy.sparse.sparsetools._dia" sources
    building extension "scipy.sparse.sparsetools._csgraph" sources
    building extension "scipy.sparse.csgraph._shortest_path" sources
    building extension "scipy.sparse.csgraph._traversal" sources
    building extension "scipy.sparse.csgraph._min_spanning_tree" sources
    building extension "scipy.sparse.csgraph._tools" sources
    building extension "scipy.spatial.qhull" sources
    building extension "scipy.spatial.ckdtree" sources
    building extension "scipy.spatial._distance_wrap" sources
    building extension "scipy.special.specfun" sources
    f2py options: ['--no-wrap-functions']
    f2py: scipy/special/specfun.pyf
    Reading fortran codes...
    	Reading file 'scipy/special/specfun.pyf' (format:free)
    Post-processing...
    	Block: specfun
    			Block: clqmn
    			Block: lqmn
    			Block: clpmn
    			Block: jdzo
    			Block: bernob
    			Block: bernoa
    			Block: csphjy
    			Block: lpmns
    			Block: eulera
    			Block: clqn
    			Block: airyzo
    			Block: eulerb
    			Block: cva1
    			Block: lqnb
    			Block: lamv
    			Block: lagzo
    			Block: legzo
    			Block: pbdv
    			Block: cerzo
    			Block: lamn
    			Block: clpn
    			Block: lqmns
    			Block: chgm
    			Block: lpmn
    			Block: fcszo
    			Block: aswfb
    			Block: lqna
    			Block: cpbdn
    			Block: lpn
    			Block: fcoef
    			Block: sphi
    			Block: rcty
    			Block: lpni
    			Block: cyzo
    			Block: csphik
    			Block: sphj
    			Block: othpl
    			Block: klvnzo
    			Block: jyzo
    			Block: rctj
    			Block: herzo
    			Block: sphk
    			Block: pbvv
    			Block: segv
    			Block: sphy
    Post-processing (stage 2)...
    Building modules...
    	Building module "specfun"...
    		Constructing wrapper function "clqmn"...
    		  cqm,cqd = clqmn(m,n,z)
    		Constructing wrapper function "lqmn"...
    		  qm,qd = lqmn(m,n,x)
    		Constructing wrapper function "clpmn"...
    		  cpm,cpd = clpmn(m,n,x,y)
    		Constructing wrapper function "jdzo"...
    		  n,m,pcode,zo = jdzo(nt)
    		Constructing wrapper function "bernob"...
    		  bn = bernob(n)
    		Constructing wrapper function "bernoa"...
    		  bn = bernoa(n)
    		Constructing wrapper function "csphjy"...
    		  nm,csj,cdj,csy,cdy = csphjy(n,z)
    		Constructing wrapper function "lpmns"...
    		  pm,pd = lpmns(m,n,x)
    		Constructing wrapper function "eulera"...
    		  en = eulera(n)
    		Constructing wrapper function "clqn"...
    		  cqn,cqd = clqn(n,z)
    		Constructing wrapper function "airyzo"...
    		  xa,xb,xc,xd = airyzo(nt,[kf])
    		Constructing wrapper function "eulerb"...
    		  en = eulerb(n)
    		Constructing wrapper function "cva1"...
    		  cv = cva1(kd,m,q)
    		Constructing wrapper function "lqnb"...
    		  qn,qd = lqnb(n,x)
    		Constructing wrapper function "lamv"...
    		  vm,vl,dl = lamv(v,x)
    		Constructing wrapper function "lagzo"...
    		  x,w = lagzo(n)
    		Constructing wrapper function "legzo"...
    		  x,w = legzo(n)
    		Constructing wrapper function "pbdv"...
    		  dv,dp,pdf,pdd = pbdv(v,x)
    		Constructing wrapper function "cerzo"...
    		  zo = cerzo(nt)
    		Constructing wrapper function "lamn"...
    		  nm,bl,dl = lamn(n,x)
    		Constructing wrapper function "clpn"...
    		  cpn,cpd = clpn(n,z)
    		Constructing wrapper function "lqmns"...
    		  qm,qd = lqmns(m,n,x)
    		Constructing wrapper function "chgm"...
    		  hg = chgm(a,b,x)
    		Constructing wrapper function "lpmn"...
    		  pm,pd = lpmn(m,n,x)
    		Constructing wrapper function "fcszo"...
    		  zo = fcszo(kf,nt)
    		Constructing wrapper function "aswfb"...
    		  s1f,s1d = aswfb(m,n,c,x,kd,cv)
    		Constructing wrapper function "lqna"...
    		  qn,qd = lqna(n,x)
    		Constructing wrapper function "cpbdn"...
    		  cpb,cpd = cpbdn(n,z)
    		Constructing wrapper function "lpn"...
    		  pn,pd = lpn(n,x)
    		Constructing wrapper function "fcoef"...
    		  fc = fcoef(kd,m,q,a)
    		Constructing wrapper function "sphi"...
    		  nm,si,di = sphi(n,x)
    		Constructing wrapper function "rcty"...
    		  nm,ry,dy = rcty(n,x)
    		Constructing wrapper function "lpni"...
    		  pn,pd,pl = lpni(n,x)
    		Constructing wrapper function "cyzo"...
    		  zo,zv = cyzo(nt,kf,kc)
    		Constructing wrapper function "csphik"...
    		  nm,csi,cdi,csk,cdk = csphik(n,z)
    		Constructing wrapper function "sphj"...
    		  nm,sj,dj = sphj(n,x)
    		Constructing wrapper function "othpl"...
    		  pl,dpl = othpl(kf,n,x)
    		Constructing wrapper function "klvnzo"...
    		  zo = klvnzo(nt,kd)
    		Constructing wrapper function "jyzo"...
    		  rj0,rj1,ry0,ry1 = jyzo(n,nt)
    		Constructing wrapper function "rctj"...
    		  nm,rj,dj = rctj(n,x)
    		Constructing wrapper function "herzo"...
    		  x,w = herzo(n)
    		Constructing wrapper function "sphk"...
    		  nm,sk,dk = sphk(n,x)
    		Constructing wrapper function "pbvv"...
    		  vv,vp,pvf,pvd = pbvv(v,x)
    		Constructing wrapper function "segv"...
    		  cv,eg = segv(m,n,c,kd)
    		Constructing wrapper function "sphy"...
    		  nm,sy,dy = sphy(n,x)
    	Wrote C/API module "specfun" to file "build/src.linux-x86_64-2.7/scipy/special/specfunmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.special._ufuncs" sources
    conv_template:> build/src.linux-x86_64-2.7/scipy/special/_logit.c
    building extension "scipy.special._ufuncs_cxx" sources
    building extension "scipy.stats.statlib" sources
    f2py options: ['--no-wrap-functions']
    f2py: scipy/stats/statlib.pyf
    Reading fortran codes...
    	Reading file 'scipy/stats/statlib.pyf' (format:free)
    Post-processing...
    	Block: statlib
    			Block: swilk
    			Block: wprob
    			Block: gscale
    			Block: prho
    Post-processing (stage 2)...
    Building modules...
    	Building module "statlib"...
    		Constructing wrapper function "swilk"...
    		  a,w,pw,ifault = swilk(x,a,[init,n1])
    		Constructing wrapper function "wprob"...
    		  astart,a1,ifault = wprob(test,other)
    		Constructing wrapper function "gscale"...
    		  astart,a1,ifault = gscale(test,other)
    		Constructing wrapper function "prho"...
    		  ifault = prho(n,is)
    	Wrote C/API module "statlib" to file "build/src.linux-x86_64-2.7/scipy/stats/statlibmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.stats.vonmises_cython" sources
    building extension "scipy.stats._rank" sources
    building extension "scipy.stats.futil" sources
    f2py options: []
    f2py:> build/src.linux-x86_64-2.7/scipy/stats/futilmodule.c
    Reading fortran codes...
    	Reading file 'scipy/stats/futil.f' (format:fix,strict)
    Post-processing...
    	Block: futil
    			Block: dqsort
    			Block: dfreps
    Post-processing (stage 2)...
    Building modules...
    	Building module "futil"...
    		Constructing wrapper function "dqsort"...
    		  arr = dqsort(arr,[overwrite_arr])
    		Constructing wrapper function "dfreps"...
    		  replist,repnum,nlist = dfreps(arr)
    	Wrote C/API module "futil" to file "build/src.linux-x86_64-2.7/scipy/stats/futilmodule.c"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
    building extension "scipy.stats.mvn" sources
    f2py options: []
    f2py: scipy/stats/mvn.pyf
    Reading fortran codes...
    	Reading file 'scipy/stats/mvn.pyf' (format:free)
    Post-processing...
    	Block: mvn
    			Block: mvnun
    			Block: mvndst
    Post-processing (stage 2)...
    Building modules...
    	Building module "mvn"...
    		Constructing wrapper function "mvnun"...
    		  value,inform = mvnun(lower,upper,means,covar,[maxpts,abseps,releps])
    		Constructing wrapper function "mvndst"...
    		  error,value,inform = mvndst(lower,upper,infin,correl,[maxpts,abseps,releps])
    		Constructing COMMON block support for "dkblck"...
    		  ivls
    	Wrote C/API module "mvn" to file "build/src.linux-x86_64-2.7/scipy/stats/mvnmodule.c"
    	Fortran 77 wrappers are saved to "build/src.linux-x86_64-2.7/scipy/stats/mvn-f2pywrappers.f"
      adding 'build/src.linux-x86_64-2.7/fortranobject.c' to sources.
      adding 'build/src.linux-x86_64-2.7' to include_dirs.
      adding 'build/src.linux-x86_64-2.7/scipy/stats/mvn-f2pywrappers.f' to sources.
    building extension "scipy.ndimage._nd_image" sources
    building extension "scipy.ndimage._ni_label" sources
    building data_files sources
    build_src: building npy-pkg config files
    customize UnixCCompiler
    customize UnixCCompiler using build_clib
    customize Gnu95FCompiler
    customize Gnu95FCompiler
    customize Gnu95FCompiler using build_clib
    building 'dfftpack' library
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/fftpack/src/dfftpack/zffti1.f
    scipy/fftpack/src/dfftpack/zffti1.f: In function ‘zffti1’:
    scipy/fftpack/src/dfftpack/zffti1.f:13:0: warning: ‘ntry’ may be used uninitialized in this function [-Wmaybe-uninitialized]
       103 NTRY = NTRY+2
     ^
    gfortran:f77: scipy/fftpack/src/dfftpack/dcosti.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dffti.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dfftf.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dsint.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dfftb.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dfftb1.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dsinti.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dcosqi.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dsinqi.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dcosqf.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dsinqb.f
    gfortran:f77: scipy/fftpack/src/dfftpack/zfftf1.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dsinqf.f
    gfortran:f77: scipy/fftpack/src/dfftpack/zfftf.f
    gfortran:f77: scipy/fftpack/src/dfftpack/zfftb.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dfftf1.f
    gfortran:f77: scipy/fftpack/src/dfftpack/zffti.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dsint1.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dcosqb.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dcost.f
    gfortran:f77: scipy/fftpack/src/dfftpack/zfftb1.f
    gfortran:f77: scipy/fftpack/src/dfftpack/dffti1.f
    scipy/fftpack/src/dfftpack/dffti1.f: In function ‘dffti1’:
    scipy/fftpack/src/dfftpack/dffti1.f:13:0: warning: ‘ntry’ may be used uninitialized in this function [-Wmaybe-uninitialized]
       103 NTRY = NTRY+2
     ^
    ar: adding 23 object files to build/temp.linux-x86_64-2.7/libdfftpack.a
    building 'fftpack' library
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/fftpack/src/fftpack/rfftf.f
    gfortran:f77: scipy/fftpack/src/fftpack/sint.f
    gfortran:f77: scipy/fftpack/src/fftpack/cfftb1.f
    gfortran:f77: scipy/fftpack/src/fftpack/cfftf.f
    gfortran:f77: scipy/fftpack/src/fftpack/sinti.f
    gfortran:f77: scipy/fftpack/src/fftpack/sint1.f
    gfortran:f77: scipy/fftpack/src/fftpack/rfftb1.f
    gfortran:f77: scipy/fftpack/src/fftpack/cosqb.f
    gfortran:f77: scipy/fftpack/src/fftpack/rffti.f
    gfortran:f77: scipy/fftpack/src/fftpack/sinqf.f
    gfortran:f77: scipy/fftpack/src/fftpack/cosqf.f
    gfortran:f77: scipy/fftpack/src/fftpack/costi.f
    gfortran:f77: scipy/fftpack/src/fftpack/sinqi.f
    gfortran:f77: scipy/fftpack/src/fftpack/cfftb.f
    gfortran:f77: scipy/fftpack/src/fftpack/rfftf1.f
    gfortran:f77: scipy/fftpack/src/fftpack/cost.f
    gfortran:f77: scipy/fftpack/src/fftpack/cffti.f
    gfortran:f77: scipy/fftpack/src/fftpack/cosqi.f
    gfortran:f77: scipy/fftpack/src/fftpack/rfftb.f
    gfortran:f77: scipy/fftpack/src/fftpack/cffti1.f
    scipy/fftpack/src/fftpack/cffti1.f: In function ‘cffti1’:
    scipy/fftpack/src/fftpack/cffti1.f:12:0: warning: ‘ntry’ may be used uninitialized in this function [-Wmaybe-uninitialized]
       103 NTRY = NTRY+2
     ^
    gfortran:f77: scipy/fftpack/src/fftpack/cfftf1.f
    gfortran:f77: scipy/fftpack/src/fftpack/rffti1.f
    scipy/fftpack/src/fftpack/rffti1.f: In function ‘rffti1’:
    scipy/fftpack/src/fftpack/rffti1.f:12:0: warning: ‘ntry’ may be used uninitialized in this function [-Wmaybe-uninitialized]
       103 NTRY = NTRY+2
     ^
    gfortran:f77: scipy/fftpack/src/fftpack/sinqb.f
    ar: adding 23 object files to build/temp.linux-x86_64-2.7/libfftpack.a
    building 'linpack_lite' library
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/integrate/linpack_lite/zgbsl.f
    scipy/integrate/linpack_lite/zgbsl.f:73.21:
    
          dimag(zdumi) = (0.0d0,-1.0d0)*zdumi
                         1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/integrate/linpack_lite/zgbsl.f:72.21:
    
          dreal(zdumr) = zdumr
                         1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    gfortran:f77: scipy/integrate/linpack_lite/dgbsl.f
    gfortran:f77: scipy/integrate/linpack_lite/zgesl.f
    scipy/integrate/linpack_lite/zgesl.f:67.21:
    
          dimag(zdumi) = (0.0d0,-1.0d0)*zdumi
                         1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/integrate/linpack_lite/zgesl.f:66.21:
    
          dreal(zdumr) = zdumr
                         1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    gfortran:f77: scipy/integrate/linpack_lite/dgesl.f
    gfortran:f77: scipy/integrate/linpack_lite/zgbfa.f
    scipy/integrate/linpack_lite/zgbfa.f:95.21:
    
          dimag(zdumi) = (0.0d0,-1.0d0)*zdumi
                         1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/integrate/linpack_lite/zgbfa.f:94.21:
    
          dreal(zdumr) = zdumr
                         1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    gfortran:f77: scipy/integrate/linpack_lite/zgefa.f
    scipy/integrate/linpack_lite/zgefa.f:59.21:
    
          dimag(zdumi) = (0.0d0,-1.0d0)*zdumi
                         1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/integrate/linpack_lite/zgefa.f:58.21:
    
          dreal(zdumr) = zdumr
                         1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    gfortran:f77: scipy/integrate/linpack_lite/dgtsl.f
    gfortran:f77: scipy/integrate/linpack_lite/dgefa.f
    gfortran:f77: scipy/integrate/linpack_lite/dgbfa.f
    ar: adding 9 object files to build/temp.linux-x86_64-2.7/liblinpack_lite.a
    building 'mach' library
    using additional config_fc from setup script for fortran compiler: {'noopt': ('scipy/integrate/setup.py', 1)}
    customize Gnu95FCompiler
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/integrate/mach/d1mach.f
    gfortran:f77: scipy/integrate/mach/r1mach.f
    scipy/integrate/mach/r1mach.f:167.27:
    
                   CALL I1MCRA(SMALL, K, 16, 0, 0)
                               1
    Warning: Rank mismatch in argument 'a' at (1) (scalar and rank-1)
    scipy/integrate/mach/r1mach.f:168.27:
    
                   CALL I1MCRA(LARGE, K, 32751, 16777215, 16777215)
                               1
    Warning: Rank mismatch in argument 'a' at (1) (scalar and rank-1)
    scipy/integrate/mach/r1mach.f:169.27:
    
                   CALL I1MCRA(RIGHT, K, 15520, 0, 0)
                               1
    Warning: Rank mismatch in argument 'a' at (1) (scalar and rank-1)
    scipy/integrate/mach/r1mach.f:170.27:
    
                   CALL I1MCRA(DIVER, K, 15536, 0, 0)
                               1
    Warning: Rank mismatch in argument 'a' at (1) (scalar and rank-1)
    scipy/integrate/mach/r1mach.f:171.27:
    
                   CALL I1MCRA(LOG10, K, 16339, 4461392, 10451455)
                               1
    Warning: Rank mismatch in argument 'a' at (1) (scalar and rank-1)
    gfortran:f77: scipy/integrate/mach/xerror.f
    scipy/integrate/mach/xerror.f:1.37:
    
          SUBROUTINE XERROR(MESS,NMESS,L1,L2)
                                         1
    Warning: Unused dummy argument 'l1' at (1)
    scipy/integrate/mach/xerror.f:1.40:
    
          SUBROUTINE XERROR(MESS,NMESS,L1,L2)
                                            1
    Warning: Unused dummy argument 'l2' at (1)
    gfortran:f77: scipy/integrate/mach/i1mach.f
    ar: adding 4 object files to build/temp.linux-x86_64-2.7/libmach.a
    building 'quadpack' library
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/integrate/quadpack/dqcheb.f
    gfortran:f77: scipy/integrate/quadpack/dqagie.f
    scipy/integrate/quadpack/dqagie.f: In function ‘dqagie’:
    scipy/integrate/quadpack/dqagie.f:384:0: warning: ‘small’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(dabs(blist(maxerr)-alist(maxerr)).gt.small) go to 90
     ^
    scipy/integrate/quadpack/dqagie.f:372:0: warning: ‘ertest’ may be used uninitialized in this function [-Wmaybe-uninitialized]
        40   if(ierro.eq.3.or.erlarg.le.ertest) go to 60
     ^
    scipy/integrate/quadpack/dqagie.f:362:0: warning: ‘erlarg’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             erlarg = erlarg-erlast
     ^
    gfortran:f77: scipy/integrate/quadpack/dqpsrt.f
    gfortran:f77: scipy/integrate/quadpack/dqmomo.f
    scipy/integrate/quadpack/dqmomo.f:126.5:
    
       90 return
         1
    Warning: Label 90 at (1) defined but not used
    gfortran:f77: scipy/integrate/quadpack/dqagp.f
    gfortran:f77: scipy/integrate/quadpack/dqwgtf.f
    scipy/integrate/quadpack/dqwgtf.f:1.49:
    
          double precision function dqwgtf(x,omega,p2,p3,p4,integr)
                                                     1
    Warning: Unused dummy argument 'p2' at (1)
    scipy/integrate/quadpack/dqwgtf.f:1.52:
    
          double precision function dqwgtf(x,omega,p2,p3,p4,integr)
                                                        1
    Warning: Unused dummy argument 'p3' at (1)
    scipy/integrate/quadpack/dqwgtf.f:1.55:
    
          double precision function dqwgtf(x,omega,p2,p3,p4,integr)
                                                           1
    Warning: Unused dummy argument 'p4' at (1)
    gfortran:f77: scipy/integrate/quadpack/dqk15i.f
    gfortran:f77: scipy/integrate/quadpack/dqagpe.f
    scipy/integrate/quadpack/dqagpe.f: In function ‘dqagpe’:
    scipy/integrate/quadpack/dqagpe.f:196:0: warning: ‘k’ may be used uninitialized in this function [-Wmaybe-uninitialized]
          *  jlow,jupbnd,k,ksgn,ktmin,last,levcur,level,levmax,limit,maxerr,
     ^
    gfortran:f77: scipy/integrate/quadpack/dqawoe.f
    scipy/integrate/quadpack/dqawoe.f: In function ‘dqawoe’:
    scipy/integrate/quadpack/dqawoe.f:449:0: warning: ‘ertest’ may be used uninitialized in this function [-Wmaybe-uninitialized]
        70   if(ierro.eq.3.or.erlarg.le.ertest) go to 90
     ^
    scipy/integrate/quadpack/dqawoe.f:428:0: warning: ‘erlarg’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             erlarg = erlarg-erlast
     ^
    gfortran:f77: scipy/integrate/quadpack/dqag.f
    gfortran:f77: scipy/integrate/quadpack/dqawf.f
    gfortran:f77: scipy/integrate/quadpack/dqc25f.f
    scipy/integrate/quadpack/dqc25f.f: In function ‘dqc25f’:
    scipy/integrate/quadpack/dqc25f.f:103:0: warning: ‘m’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           integer i,iers,integr,isym,j,k,ksave,m,momcom,neval,maxp1,
     ^
    gfortran:f77: scipy/integrate/quadpack/dqk61.f
    gfortran:f77: scipy/integrate/quadpack/dqagse.f
    scipy/integrate/quadpack/dqagse.f: In function ‘dqagse’:
    scipy/integrate/quadpack/dqagse.f:376:0: warning: ‘small’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(dabs(blist(maxerr)-alist(maxerr)).gt.small) go to 90
     ^
    scipy/integrate/quadpack/dqagse.f:363:0: warning: ‘ertest’ may be used uninitialized in this function [-Wmaybe-uninitialized]
        40   if(ierro.eq.3.or.erlarg.le.ertest) go to 60
     ^
    scipy/integrate/quadpack/dqagse.f:353:0: warning: ‘erlarg’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             erlarg = erlarg-erlast
     ^
    gfortran:f77: scipy/integrate/quadpack/dqk15w.f
    gfortran:f77: scipy/integrate/quadpack/dqawc.f
    gfortran:f77: scipy/integrate/quadpack/dqawse.f
    gfortran:f77: scipy/integrate/quadpack/dqage.f
    gfortran:f77: scipy/integrate/quadpack/dqawo.f
    gfortran:f77: scipy/integrate/quadpack/dqk31.f
    gfortran:f77: scipy/integrate/quadpack/dqawce.f
    gfortran:f77: scipy/integrate/quadpack/dqelg.f
    gfortran:f77: scipy/integrate/quadpack/dqc25s.f
    gfortran:f77: scipy/integrate/quadpack/dqk15.f
    gfortran:f77: scipy/integrate/quadpack/dqng.f
    scipy/integrate/quadpack/dqng.f: In function ‘dqng’:
    scipy/integrate/quadpack/dqng.f:365:0: warning: ‘resasc’ may be used uninitialized in this function [-Wmaybe-uninitialized]
          *  abserr = resasc*dmin1(0.1d+01,(0.2d+03*abserr/resasc)**1.5d+00)
     ^
    scipy/integrate/quadpack/dqng.f:366:0: warning: ‘resabs’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           if (resabs.gt.uflow/(0.5d+02*epmach)) abserr = dmax1
     ^
    scipy/integrate/quadpack/dqng.f:363:0: warning: ‘res43’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           abserr = dabs((res87-res43)*hlgth)
     ^
    scipy/integrate/quadpack/dqng.f:348:0: warning: ‘res21’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           abserr = dabs((res43-res21)*hlgth)
     ^
    scipy/integrate/quadpack/dqng.f:82:0: warning: ‘ipx’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           integer ier,ipx,k,l,neval
     ^
    gfortran:f77: scipy/integrate/quadpack/dqwgts.f
    gfortran:f77: scipy/integrate/quadpack/dqwgtc.f
    scipy/integrate/quadpack/dqwgtc.f:1.54:
    
          double precision function dqwgtc(x,c,p2,p3,p4,kp)
                                                          1
    Warning: Unused dummy argument 'kp' at (1)
    scipy/integrate/quadpack/dqwgtc.f:1.45:
    
          double precision function dqwgtc(x,c,p2,p3,p4,kp)
                                                 1
    Warning: Unused dummy argument 'p2' at (1)
    scipy/integrate/quadpack/dqwgtc.f:1.48:
    
          double precision function dqwgtc(x,c,p2,p3,p4,kp)
                                                    1
    Warning: Unused dummy argument 'p3' at (1)
    scipy/integrate/quadpack/dqwgtc.f:1.51:
    
          double precision function dqwgtc(x,c,p2,p3,p4,kp)
                                                       1
    Warning: Unused dummy argument 'p4' at (1)
    gfortran:f77: scipy/integrate/quadpack/dqagi.f
    gfortran:f77: scipy/integrate/quadpack/dqags.f
    gfortran:f77: scipy/integrate/quadpack/dqk41.f
    gfortran:f77: scipy/integrate/quadpack/dqaws.f
    gfortran:f77: scipy/integrate/quadpack/dqk21.f
    gfortran:f77: scipy/integrate/quadpack/dqk51.f
    gfortran:f77: scipy/integrate/quadpack/dqawfe.f
    scipy/integrate/quadpack/dqawfe.f:267.10:
    
       10 l = dabs(omega)
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/integrate/quadpack/dqawfe.f: In function ‘dqawfe’:
    scipy/integrate/quadpack/dqawfe.f:356:0: warning: ‘drl’ may be used uninitialized in this function [-Wmaybe-uninitialized]
        70 if(abserr/dabs(result).gt.(errsum+drl)/dabs(psum(numrl2)))
     ^
    scipy/integrate/quadpack/dqawfe.f:316:0: warning: ‘ll’ may be used uninitialized in this function [-Wmaybe-uninitialized]
        20   psum(numrl2) = psum(ll)+rslst(lst)
     ^
    gfortran:f77: scipy/integrate/quadpack/dqc25c.f
    ar: adding 35 object files to build/temp.linux-x86_64-2.7/libquadpack.a
    building 'odepack' library
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/integrate/odepack/xsetf.f
    gfortran:f77: scipy/integrate/odepack/odrv.f
    gfortran:f77: scipy/integrate/odepack/decbt.f
    gfortran:f77: scipy/integrate/odepack/solbt.f
    gfortran:f77: scipy/integrate/odepack/cntnzu.f
    gfortran:f77: scipy/integrate/odepack/lsode.f
    scipy/integrate/odepack/lsode.f: In function ‘lsode’:
    scipy/integrate/odepack/lsode.f:1311:0: warning: ‘ihit’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           if (ihit) t = tcrit
     ^
    gfortran:f77: scipy/integrate/odepack/solsy.f
    scipy/integrate/odepack/solsy.f:1.39:
    
          subroutine solsy (wm, iwm, x, tem)
                                           1
    Warning: Unused dummy argument 'tem' at (1)
    gfortran:f77: scipy/integrate/odepack/fnorm.f
    gfortran:f77: scipy/integrate/odepack/roots.f
    gfortran:f77: scipy/integrate/odepack/xerrwv.f
    scipy/integrate/odepack/xerrwv.f:1.40:
    
          subroutine xerrwv (msg, nmes, nerr, level, ni, i1, i2, nr, r1, r2)
                                            1
    Warning: Unused dummy argument 'nerr' at (1)
    gfortran:f77: scipy/integrate/odepack/lsoibt.f
    scipy/integrate/odepack/lsoibt.f: In function ‘lsoibt’:
    scipy/integrate/odepack/lsoibt.f:1575:0: warning: ‘ihit’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           if (ihit) t = tcrit
     ^
    gfortran:f77: scipy/integrate/odepack/cfode.f
    gfortran:f77: scipy/integrate/odepack/cdrv.f
    gfortran:f77: scipy/integrate/odepack/mdp.f
    scipy/integrate/odepack/mdp.f: In function ‘mdp’:
    scipy/integrate/odepack/mdp.f:83:0: warning: ‘free’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               l(vi) = free
     ^
    gfortran:f77: scipy/integrate/odepack/nnfc.f
    gfortran:f77: scipy/integrate/odepack/nnsc.f
    gfortran:f77: scipy/integrate/odepack/lsodi.f
    scipy/integrate/odepack/lsodi.f: In function ‘lsodi’:
    scipy/integrate/odepack/lsodi.f:1521:0: warning: ‘ihit’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           if (ihit) t = tcrit
     ^
    gfortran:f77: scipy/integrate/odepack/stoda.f
    scipy/integrate/odepack/stoda.f: In function ‘stoda’:
    scipy/integrate/odepack/stoda.f:223:0: warning: ‘iredo’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           if (iredo .eq. 0) go to 690
     ^
    scipy/integrate/odepack/stoda.f:372:0: warning: ‘dsm’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           if (dsm .gt. 1.0d0) go to 500
     ^
    scipy/integrate/odepack/stoda.f:18:0: warning: ‘rh’ may be used uninitialized in this function [-Wmaybe-uninitialized]
          1   r, rh, rhdn, rhsm, rhup, told, vmnorm
     ^
    gfortran:f77: scipy/integrate/odepack/prepji.f
    gfortran:f77: scipy/integrate/odepack/nntc.f
    gfortran:f77: scipy/integrate/odepack/md.f
    gfortran:f77: scipy/integrate/odepack/srcma.f
    gfortran:f77: scipy/integrate/odepack/prep.f
    gfortran:f77: scipy/integrate/odepack/srcar.f
    gfortran:f77: scipy/integrate/odepack/sro.f
    gfortran:f77: scipy/integrate/odepack/stodi.f
    scipy/integrate/odepack/stodi.f: In function ‘stodi’:
    scipy/integrate/odepack/stodi.f:401:0: warning: ‘dsm’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           rhsm = 1.0d0/(1.2d0*dsm**exsm + 0.0000012d0)
     ^
    scipy/integrate/odepack/stodi.f:15:0: warning: ‘rh’ may be used uninitialized in this function [-Wmaybe-uninitialized]
          2   r, rh, rhdn, rhsm, rhup, told, vnorm
     ^
    scipy/integrate/odepack/stodi.f:211:0: warning: ‘iredo’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           if (iredo .eq. 0) go to 690
     ^
    gfortran:f77: scipy/integrate/odepack/iprep.f
    gfortran:f77: scipy/integrate/odepack/aigbt.f
    gfortran:f77: scipy/integrate/odepack/lsodar.f
    scipy/integrate/odepack/lsodar.f: In function ‘lsodar’:
    scipy/integrate/odepack/lsodar.f:1606:0: warning: ‘ihit’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           if (ihit) t = tcrit
     ^
    scipy/integrate/odepack/lsodar.f:1255:0: warning: ‘lenwm’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           len1s = len1s + lenwm
     ^
    gfortran:f77: scipy/integrate/odepack/slss.f
    scipy/integrate/odepack/slss.f:1.38:
    
          subroutine slss (wk, iwk, x, tem)
                                          1
    Warning: Unused dummy argument 'tem' at (1)
    gfortran:f77: scipy/integrate/odepack/stode.f
    scipy/integrate/odepack/stode.f: In function ‘stode’:
    scipy/integrate/odepack/stode.f:203:0: warning: ‘iredo’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           if (iredo .eq. 0) go to 690
     ^
    scipy/integrate/odepack/stode.f:326:0: warning: ‘dsm’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           if (dsm .gt. 1.0d0) go to 500
     ^
    scipy/integrate/odepack/stode.f:14:0: warning: ‘rh’ may be used uninitialized in this function [-Wmaybe-uninitialized]
          1   r, rh, rhdn, rhsm, rhup, told, vnorm
     ^
    gfortran:f77: scipy/integrate/odepack/jgroup.f
    gfortran:f77: scipy/integrate/odepack/prja.f
    gfortran:f77: scipy/integrate/odepack/lsoda.f
    scipy/integrate/odepack/lsoda.f: In function ‘lsoda’:
    scipy/integrate/odepack/lsoda.f:1424:0: warning: ‘ihit’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           if (ihit) t = tcrit
     ^
    scipy/integrate/odepack/lsoda.f:1112:0: warning: ‘lenwm’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           len1s = len1s + lenwm
     ^
    gfortran:f77: scipy/integrate/odepack/vnorm.f
    gfortran:f77: scipy/integrate/odepack/adjlr.f
    gfortran:f77: scipy/integrate/odepack/nsfc.f
    gfortran:f77: scipy/integrate/odepack/pjibt.f
    gfortran:f77: scipy/integrate/odepack/vode.f
    scipy/integrate/odepack/vode.f:2373.4:
    
     700  R = ONE/TQ(2)
        1
    Warning: Label 700 at (1) defined but not used
    scipy/integrate/odepack/vode.f:3514.40:
    
          SUBROUTINE XERRWD (MSG, NMES, NERR, LEVEL, NI, I1, I2, NR, R1, R2)
                                            1
    Warning: Unused dummy argument 'nerr' at (1)
    scipy/integrate/odepack/vode.f:3495.44:
    
          DOUBLE PRECISION FUNCTION D1MACH (IDUM)
                                                1
    Warning: Unused dummy argument 'idum' at (1)
    scipy/integrate/odepack/vode.f:2740.35:
    
         1                 F, JAC, PDUM, NFLAG, RPAR, IPAR)
                                       1
    Warning: Unused dummy argument 'pdum' at (1)
    scipy/integrate/odepack/vode.f:2739.42:
    
          SUBROUTINE DVNLSD (Y, YH, LDYH, VSAV, SAVF, EWT, ACOR, IWM, WM,
                                              1
    Warning: Unused dummy argument 'vsav' at (1)
    scipy/integrate/odepack/vode.f: In function ‘ixsav’:
    scipy/integrate/odepack/vode.f:3610:0: warning: ‘__result_ixsav’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           INTEGER FUNCTION IXSAV (IPAR, IVALUE, ISET)
     ^
    scipy/integrate/odepack/vode.f: In function ‘dvode’:
    scipy/integrate/odepack/vode.f:1487:0: warning: ‘ihit’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           IF (IHIT) T = TCRIT
     ^
    gfortran:f77: scipy/integrate/odepack/prjs.f
    gfortran:f77: scipy/integrate/odepack/prepj.f
    gfortran:f77: scipy/integrate/odepack/slsbt.f
    scipy/integrate/odepack/slsbt.f:1.39:
    
          subroutine slsbt (wm, iwm, x, tem)
                                           1
    Warning: Unused dummy argument 'tem' at (1)
    gfortran:f77: scipy/integrate/odepack/intdy.f
    gfortran:f77: scipy/integrate/odepack/srcom.f
    gfortran:f77: scipy/integrate/odepack/zvode.f
    scipy/integrate/odepack/zvode.f:2394.4:
    
     700  R = ONE/TQ(2)
        1
    Warning: Label 700 at (1) defined but not used
    scipy/integrate/odepack/zvode.f:2761.35:
    
         1                 F, JAC, PDUM, NFLAG, RPAR, IPAR)
                                       1
    Warning: Unused dummy argument 'pdum' at (1)
    scipy/integrate/odepack/zvode.f:2760.42:
    
          SUBROUTINE ZVNLSD (Y, YH, LDYH, VSAV, SAVF, EWT, ACOR, IWM, WM,
                                              1
    Warning: Unused dummy argument 'vsav' at (1)
    scipy/integrate/odepack/zvode.f: In function ‘zvode’:
    scipy/integrate/odepack/zvode.f:1502:0: warning: ‘ihit’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           IF (IHIT) T = TCRIT
     ^
    gfortran:f77: scipy/integrate/odepack/nroc.f
    gfortran:f77: scipy/integrate/odepack/ddasrt.f
    scipy/integrate/odepack/ddasrt.f:1538.3:
    
    770   MSG = 'DASSL--  RUN TERMINATED. APPARENT INFINITE LOOP'
       1
    Warning: Label 770 at (1) defined but not used
    scipy/integrate/odepack/ddasrt.f:1080.3:
    
    360   ITEMP = LPHI + NEQ
       1
    Warning: Label 360 at (1) defined but not used
    scipy/integrate/odepack/ddasrt.f:1022.3:
    
    300   CONTINUE
       1
    Warning: Label 300 at (1) defined but not used
    scipy/integrate/odepack/ddasrt.f:1096.19:
    
         *  RWORK(LGX),JROOT,IRT,RWORK(LROUND),INFO(3),
                       1
    Warning: Rank mismatch in argument 'jroot' at (1) (rank-1 and scalar)
    scipy/integrate/odepack/ddasrt.f:1106.19:
    
         *  RWORK(LGX),JROOT,IRT,RWORK(LROUND),INFO(3),
                       1
    Warning: Rank mismatch in argument 'jroot' at (1) (rank-1 and scalar)
    scipy/integrate/odepack/ddasrt.f:1134.19:
    
         *  RWORK(LGX),JROOT,IRT,RWORK(LROUND),INFO(3),
                       1
    Warning: Rank mismatch in argument 'jroot' at (1) (rank-1 and scalar)
    scipy/integrate/odepack/ddasrt.f:1298.19:
    
         *  RWORK(LGX),JROOT,IRT,RWORK(LROUND),INFO(3),
                       1
    Warning: Rank mismatch in argument 'jroot' at (1) (rank-1 and scalar)
    scipy/integrate/odepack/ddasrt.f:1932.40:
    
          SUBROUTINE XERRWV (MSG, NMES, NERR, LEVEL, NI, I1, I2, NR, R1, R2)
                                            1
    Warning: Unused dummy argument 'nerr' at (1)
    gfortran:f77: scipy/integrate/odepack/ainvg.f
    gfortran:f77: scipy/integrate/odepack/bnorm.f
    gfortran:f77: scipy/integrate/odepack/mdu.f
    gfortran:f77: scipy/integrate/odepack/mdi.f
    gfortran:f77: scipy/integrate/odepack/vmnorm.f
    gfortran:f77: scipy/integrate/odepack/rchek.f
    gfortran:f77: scipy/integrate/odepack/ewset.f
    gfortran:f77: scipy/integrate/odepack/ddassl.f
    scipy/integrate/odepack/ddassl.f:3153.5:
    
       30 IF (LEVEL.LE.0 .OR. (LEVEL.EQ.1 .AND. MKNTRL.LE.1)) RETURN
         1
    Warning: Label 30 at (1) defined but not used
    scipy/integrate/odepack/ddassl.f:1647.62:
    
          DOUBLE PRECISION FUNCTION DDANRM (NEQ, V, WT, RPAR, IPAR)
                                                                  1
    Warning: Unused dummy argument 'ipar' at (1)
    scipy/integrate/odepack/ddassl.f:1647.56:
    
          DOUBLE PRECISION FUNCTION DDANRM (NEQ, V, WT, RPAR, IPAR)
                                                            1
    Warning: Unused dummy argument 'rpar' at (1)
    scipy/integrate/odepack/ddassl.f:1605.64:
    
          SUBROUTINE DDAWTS (NEQ, IWT, RTOL, ATOL, Y, WT, RPAR, IPAR)
                                                                    1
    Warning: Unused dummy argument 'ipar' at (1)
    scipy/integrate/odepack/ddassl.f:1605.58:
    
          SUBROUTINE DDAWTS (NEQ, IWT, RTOL, ATOL, Y, WT, RPAR, IPAR)
                                                              1
    Warning: Unused dummy argument 'rpar' at (1)
    scipy/integrate/odepack/ddassl.f:3170.30:
    
          SUBROUTINE XERHLT (MESSG)
                                  1
    Warning: Unused dummy argument 'messg' at (1)
    scipy/integrate/odepack/ddassl.f: In function ‘ddastp’:
    scipy/integrate/odepack/ddassl.f:2456:0: warning: ‘terkm1’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     520   IF(TERKM1.LE.MIN(TERK,TERKP1))GO TO 540
     ^
    scipy/integrate/odepack/ddassl.f:2481:0: warning: ‘erkm1’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           R=(2.0D0*EST+0.0001D0)**(-1.0D0/TEMP2)
     ^
    scipy/integrate/odepack/ddassl.f: In function ‘ddaini’:
    scipy/integrate/odepack/ddassl.f:1857:0: warning: ‘s’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     350   IF (S*DELNRM .LE. 0.33D0) GO TO 400
     ^
    gfortran:f77: scipy/integrate/odepack/mdm.f
    gfortran:f77: scipy/integrate/odepack/blkdta000.f
    gfortran:f77: scipy/integrate/odepack/xsetun.f
    gfortran:f77: scipy/integrate/odepack/srcms.f
    gfortran:f77: scipy/integrate/odepack/lsodes.f
    scipy/integrate/odepack/lsodes.f: In function ‘lsodes’:
    scipy/integrate/odepack/lsodes.f:1716:0: warning: ‘ihit’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           if (ihit) t = tcrit
     ^
    ar: adding 50 object files to build/temp.linux-x86_64-2.7/libodepack.a
    ar: adding 10 object files to build/temp.linux-x86_64-2.7/libodepack.a
    building 'dop' library
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/integrate/dop/dopri5.f
    scipy/integrate/dop/dopri5.f:558.35:
    
          FUNCTION HINIT(N,FCN,X,Y,XEND,POSNEG,F0,F1,Y1,IORD,
                                       1
    Warning: Unused dummy argument 'xend' at (1)
    scipy/integrate/dop/dopri5.f: In function ‘dopcor’:
    scipy/integrate/dop/dopri5.f:491:0: warning: ‘nonsti’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                    NONSTI=NONSTI+1
     ^
    gfortran:f77: scipy/integrate/dop/dop853.f
    scipy/integrate/dop/dop853.f:364.42:
    
         &   SOLOUT,IOUT,IDID,NMAX,UROUND,METH,NSTIFF,SAFE,BETA,FAC1,FAC2,
                                              1
    Warning: Unused dummy argument 'meth' at (1)
    scipy/integrate/dop/dop853.f:791.38:
    
          FUNCTION HINIT853(N,FCN,X,Y,XEND,POSNEG,F0,F1,Y1,IORD,
                                          1
    Warning: Unused dummy argument 'xend' at (1)
    scipy/integrate/dop/dop853.f: In function ‘dp86co’:
    scipy/integrate/dop/dop853.f:686:0: warning: ‘nonsti’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                    NONSTI=NONSTI+1
     ^
    ar: adding 2 object files to build/temp.linux-x86_64-2.7/libdop.a
    building 'fitpack' library
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/interpolate/fitpack/evapol.f
    gfortran:f77: scipy/interpolate/fitpack/percur.f
    gfortran:f77: scipy/interpolate/fitpack/fpchep.f
    gfortran:f77: scipy/interpolate/fitpack/fpsuev.f
    gfortran:f77: scipy/interpolate/fitpack/fpcons.f
    scipy/interpolate/fitpack/fpcons.f:224.35:
    
            if(fpold-fp.gt.acc) npl1 = rn*fpms/(fpold-fp)
                                       1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/interpolate/fitpack/fpcons.f: In function ‘fpcons’:
    scipy/interpolate/fitpack/fpcons.f:225:0: warning: ‘nplus’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             nplus = min0(nplus*2,max0(npl1,nplus/2,1))
     ^
    scipy/interpolate/fitpack/fpcons.f:264:0: warning: ‘nmax’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(n.eq.nmax) go to 25
     ^
    scipy/interpolate/fitpack/fpcons.f:383:0: warning: ‘nk1’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(u(it).lt.t(l) .or. l.gt.nk1) go to 310
     ^
    scipy/interpolate/fitpack/fpcons.f:81:0: warning: ‘mm’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             t(i) = u(j)
     ^
    scipy/interpolate/fitpack/fpcons.f:224:0: warning: ‘fpold’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if(fpold-fp.gt.acc) npl1 = rn*fpms/(fpold-fp)
     ^
    scipy/interpolate/fitpack/fpcons.f:301:0: warning: ‘fpms’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           f3 = fpms
     ^
    scipy/interpolate/fitpack/fpcons.f:194:0: warning: ‘fp0’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             fpint(n) = fp0
     ^
    scipy/interpolate/fitpack/fpcons.f:418:0: warning: ‘acc’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if((f1-f2).gt.acc) go to 345
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpbacp.f
    gfortran:f77: scipy/interpolate/fitpack/spalde.f
    gfortran:f77: scipy/interpolate/fitpack/splint.f
    gfortran:f77: scipy/interpolate/fitpack/pogrid.f
    gfortran:f77: scipy/interpolate/fitpack/fprppo.f
    scipy/interpolate/fitpack/fprppo.f: In function ‘fprppo’:
    scipy/interpolate/fitpack/fprppo.f:48:0: warning: ‘j’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                 f(i) = c(j)
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpbisp.f
    gfortran:f77: scipy/interpolate/fitpack/parder.f
    gfortran:f77: scipy/interpolate/fitpack/fppogr.f
    scipy/interpolate/fitpack/fppogr.f:286.33:
    
            if(reducu.gt.acc) npl1 = rn*fpms/reducu
                                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/interpolate/fitpack/fppogr.f:293.33:
    
            if(reducv.gt.acc) npl1 = rn*fpms/reducv
                                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/interpolate/fitpack/fppogr.f: In function ‘fppogr’:
    scipy/interpolate/fitpack/fppogr.f:353:0: warning: ‘fpms’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           f3 = fpms
     ^
    scipy/interpolate/fitpack/fppogr.f:20:0: warning: ‘nplu’ may be used uninitialized in this function [-Wmaybe-uninitialized]
          * ktu,l,l1,l2,l3,l4,mpm,mumin,mu0,mu1,nn,nplu,nplv,npl1,nrintu,
     ^
    scipy/interpolate/fitpack/fppogr.f:260:0: warning: ‘nvmax’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if(nu.eq.numax .and. nv.eq.nvmax) go to 430
     ^
    scipy/interpolate/fitpack/fppogr.f:325:0: warning: ‘nve’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(nv.eq.nve) go to 270
     ^
    scipy/interpolate/fitpack/fppogr.f:260:0: warning: ‘numax’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if(nu.eq.numax .and. nv.eq.nvmax) go to 430
     ^
    scipy/interpolate/fitpack/fppogr.f:312:0: warning: ‘nue’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(nu.eq.nue) go to 270
     ^
    scipy/interpolate/fitpack/fppogr.f:385:0: warning: ‘acc’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if((f1-f2).gt.acc) go to 330
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpperi.f
    scipy/interpolate/fitpack/fpperi.f:339.35:
    
            if(fpold-fp.gt.acc) npl1 = rn*fpms/(fpold-fp)
                                       1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/interpolate/fitpack/fpperi.f: In function ‘fpperi’:
    scipy/interpolate/fitpack/fpperi.f:340:0: warning: ‘nplus’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             nplus = min0(nplus*2,max0(npl1,nplus/2,1))
     ^
    scipy/interpolate/fitpack/fpperi.f:375:0: warning: ‘nmax’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(n.eq.nmax) go to 5
     ^
    scipy/interpolate/fitpack/fpperi.f:410:0: warning: ‘n10’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           n11 = n10-1
     ^
    scipy/interpolate/fitpack/fpperi.f:16:0: warning: ‘i1’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           integer i,ich1,ich3,ij,ik,it,iter,i1,i2,i3,j,jk,jper,j1,j2,kk,
     ^
    scipy/interpolate/fitpack/fpperi.f:339:0: warning: ‘fpold’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if(fpold-fp.gt.acc) npl1 = rn*fpms/(fpold-fp)
     ^
    scipy/interpolate/fitpack/fpperi.f:409:0: warning: ‘fpms’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           f3 = fpms
     ^
    scipy/interpolate/fitpack/fpperi.f:407:0: warning: ‘fp0’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           f1 = fp0-s
     ^
    scipy/interpolate/fitpack/fpperi.f:574:0: warning: ‘acc’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if((f1-f2) .gt. acc) go to 585
     ^
    gfortran:f77: scipy/interpolate/fitpack/fourco.f
    gfortran:f77: scipy/interpolate/fitpack/splev.f
    scipy/interpolate/fitpack/splev.f:80.4:
    
      30  ier = 0
        1
    Warning: Label 30 at (1) defined but not used
    gfortran:f77: scipy/interpolate/fitpack/curev.f
    gfortran:f77: scipy/interpolate/fitpack/surfit.f
    gfortran:f77: scipy/interpolate/fitpack/fpcuro.f
    gfortran:f77: scipy/interpolate/fitpack/fprpsp.f
    gfortran:f77: scipy/interpolate/fitpack/fpsurf.f
    scipy/interpolate/fitpack/fpsurf.f:305.4:
    
     310    do 320 i=1,nrint
        1
    Warning: Label 310 at (1) defined but not used
    scipy/interpolate/fitpack/fpsurf.f:245.4:
    
     240      in = nummer(in)
        1
    Warning: Label 240 at (1) defined but not used
    scipy/interpolate/fitpack/fpsurf.f: In function ‘fpsurf’:
    scipy/interpolate/fitpack/fpsurf.f:567:0: warning: ‘nyy’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               ly = num1-lx*nyy
     ^
    scipy/interpolate/fitpack/fpsurf.f:433:0: warning: ‘nk1y’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           iband3 = kx1*nk1y
     ^
    scipy/interpolate/fitpack/fpsurf.f:21:0: warning: ‘nk1x’ may be used uninitialized in this function [-Wmaybe-uninitialized]
          * la,lf,lh,lwest,lx,ly,l1,l2,n,ncof,nk1x,nk1y,nminx,nminy,nreg,
     ^
    scipy/interpolate/fitpack/fpsurf.f:621:0: warning: ‘lwest’ may be used uninitialized in this function [-Wmaybe-uninitialized]
      780  ier = lwest
     ^
    scipy/interpolate/fitpack/fpsurf.f:19:0: warning: ‘iband1’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           integer i,iband,iband1,iband3,iband4,ibb,ichang,ich1,ich3,ii,
     ^
    scipy/interpolate/fitpack/fpsurf.f:425:0: warning: ‘fpms’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           f3 = fpms
     ^
    scipy/interpolate/fitpack/fpsurf.f:605:0: warning: ‘acc’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if((f1-f2).gt.acc) go to 750
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpinst.f
    gfortran:f77: scipy/interpolate/fitpack/fpcosp.f
    gfortran:f77: scipy/interpolate/fitpack/concon.f
    gfortran:f77: scipy/interpolate/fitpack/fpknot.f
    scipy/interpolate/fitpack/fpknot.f: In function ‘fpknot’:
    scipy/interpolate/fitpack/fpknot.f:42:0: warning: ‘number’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           next = number+1
     ^
    scipy/interpolate/fitpack/fpknot.f:40:0: warning: ‘maxpt’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           ihalf = maxpt/2+1
     ^
    scipy/interpolate/fitpack/fpknot.f:41:0: warning: ‘maxbeg’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           nrx = maxbeg+ihalf
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpregr.f
    scipy/interpolate/fitpack/fpregr.f:246.33:
    
            if(reducx.gt.acc) npl1 = rn*fpms/reducx
                                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/interpolate/fitpack/fpregr.f:253.33:
    
            if(reducy.gt.acc) npl1 = rn*fpms/reducy
                                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/interpolate/fitpack/fpregr.f: In function ‘fpregr’:
    scipy/interpolate/fitpack/fpregr.f:310:0: warning: ‘fpms’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           f3 = fpms
     ^
    scipy/interpolate/fitpack/fpregr.f:282:0: warning: ‘nye’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(ny.eq.nye) go to 250
     ^
    scipy/interpolate/fitpack/fpregr.f:269:0: warning: ‘nxe’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(nx.eq.nxe) go to 250
     ^
    scipy/interpolate/fitpack/fpregr.f:225:0: warning: ‘nmaxy’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if(nx.eq.nmaxx .and. ny.eq.nmaxy) go to 430
     ^
    scipy/interpolate/fitpack/fpregr.f:225:0: warning: ‘nmaxx’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/interpolate/fitpack/fpregr.f:341:0: warning: ‘acc’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if((f1-f2).gt.acc) go to 330
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpgrsp.f
    scipy/interpolate/fitpack/fpgrsp.f:348.4:
    
     400    if(nrold.eq.number) go to 420
        1
    Warning: Label 400 at (1) defined but not used
    scipy/interpolate/fitpack/fpgrsp.f: In function ‘fpgrsp’:
    scipy/interpolate/fitpack/fpgrsp.f:239:0: warning: ‘pinv’ may be used uninitialized in this function [-Wmaybe-uninitialized]
              b1(2,i) = fac*pinv
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpched.f
    gfortran:f77: scipy/interpolate/fitpack/regrid.f
    gfortran:f77: scipy/interpolate/fitpack/sproot.f
    gfortran:f77: scipy/interpolate/fitpack/fpchec.f
    gfortran:f77: scipy/interpolate/fitpack/fpbspl.f
    gfortran:f77: scipy/interpolate/fitpack/fpsysy.f
    gfortran:f77: scipy/interpolate/fitpack/fpspgr.f
    scipy/interpolate/fitpack/fpspgr.f:315.33:
    
            if(reducu.gt.acc) npl1 = rn*fpms/reducu
                                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/interpolate/fitpack/fpspgr.f:322.33:
    
            if(reducv.gt.acc) npl1 = rn*fpms/reducv
                                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/interpolate/fitpack/fpspgr.f: In function ‘fpspgr’:
    scipy/interpolate/fitpack/fpspgr.f:382:0: warning: ‘fpms’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           f3 = fpms
     ^
    scipy/interpolate/fitpack/fpspgr.f:20:0: warning: ‘nplu’ may be used uninitialized in this function [-Wmaybe-uninitialized]
          * ktu,l,l1,l2,l3,l4,mpm,mumin,mu0,mu1,nn,nplu,nplv,npl1,nrintu,
     ^
    scipy/interpolate/fitpack/fpspgr.f:287:0: warning: ‘nvmax’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if(nu.eq.numax .and. nv.eq.nvmax) go to 430
     ^
    scipy/interpolate/fitpack/fpspgr.f:354:0: warning: ‘nve’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(nv.eq.nve) go to 270
     ^
    scipy/interpolate/fitpack/fpspgr.f:287:0: warning: ‘numax’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if(nu.eq.numax .and. nv.eq.nvmax) go to 430
     ^
    scipy/interpolate/fitpack/fpspgr.f:341:0: warning: ‘nue’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(nu.eq.nue) go to 270
     ^
    scipy/interpolate/fitpack/fpspgr.f:414:0: warning: ‘acc’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if((f1-f2).gt.acc) go to 330
     ^
    gfortran:f77: scipy/interpolate/fitpack/fprota.f
    gfortran:f77: scipy/interpolate/fitpack/polar.f
    scipy/interpolate/fitpack/polar.f:353.10:
    
         * lbv,lco,lf,lff,lfp,lh,lq,lsu,lsv,lwest,maxit,ncest,ncc,nuu,
              1
    Warning: Unused variable 'jlbv' declared at (1)
    gfortran:f77: scipy/interpolate/fitpack/cocosp.f
    gfortran:f77: scipy/interpolate/fitpack/concur.f
    scipy/interpolate/fitpack/concur.f:287.21:
    
          real*8 tol,dist
                         1
    Warning: Unused variable 'dist' declared at (1)
    gfortran:f77: scipy/interpolate/fitpack/fpgrpa.f
    gfortran:f77: scipy/interpolate/fitpack/fpintb.f
    scipy/interpolate/fitpack/fpintb.f: In function ‘fpintb’:
    scipy/interpolate/fitpack/fpintb.f:92:0: warning: ‘h[5]’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                 aint(i) = aint(i)+h(i)*(arg-t(lj))/(t(li)-t(lj))
     ^
    scipy/interpolate/fitpack/fpintb.f:92:0: warning: ‘h[4]’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/interpolate/fitpack/fpintb.f:92:0: warning: ‘h[3]’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/interpolate/fitpack/fpintb.f:92:0: warning: ‘h[2]’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/interpolate/fitpack/fpintb.f:26:0: warning: ‘ia’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           integer i,ia,ib,it,j,j1,k,k1,l,li,lj,lk,l0,min
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpcurf.f
    scipy/interpolate/fitpack/fpcurf.f:186.35:
    
            if(fpold-fp.gt.acc) npl1 = rn*fpms/(fpold-fp)
                                       1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/interpolate/fitpack/fpcurf.f: In function ‘fpcurf’:
    scipy/interpolate/fitpack/fpcurf.f:187:0: warning: ‘nplus’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             nplus = min0(nplus*2,max0(npl1,nplus/2,1))
     ^
    scipy/interpolate/fitpack/fpcurf.f:219:0: warning: ‘nmax’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(n.eq.nmax) go to 10
     ^
    scipy/interpolate/fitpack/fpcurf.f:186:0: warning: ‘fpold’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if(fpold-fp.gt.acc) npl1 = rn*fpms/(fpold-fp)
     ^
    scipy/interpolate/fitpack/fpcurf.f:256:0: warning: ‘fpms’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           f3 = fpms
     ^
    scipy/interpolate/fitpack/fpcurf.f:162:0: warning: ‘fp0’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             fpint(n) = fp0
     ^
    scipy/interpolate/fitpack/fpcurf.f:335:0: warning: ‘acc’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if((f1-f2).gt.acc) go to 345
     ^
    gfortran:f77: scipy/interpolate/fitpack/dblint.f
    gfortran:f77: scipy/interpolate/fitpack/bispev.f
    gfortran:f77: scipy/interpolate/fitpack/bispeu.f
    scipy/interpolate/fitpack/bispeu.f:50.18:
    
          integer i,iw,lwest
                      1
    Warning: Unused variable 'iw' declared at (1)
    scipy/interpolate/fitpack/bispeu.f:44.37:
    
          integer nx,ny,kx,ky,m,lwrk,kwrk,ier
                                         1
    Warning: Unused variable 'kwrk' declared at (1)
    gfortran:f77: scipy/interpolate/fitpack/cualde.f
    gfortran:f77: scipy/interpolate/fitpack/fpcyt1.f
    gfortran:f77: scipy/interpolate/fitpack/fpadpo.f
    gfortran:f77: scipy/interpolate/fitpack/fpback.f
    gfortran:f77: scipy/interpolate/fitpack/fpdeno.f
    gfortran:f77: scipy/interpolate/fitpack/fppocu.f
    gfortran:f77: scipy/interpolate/fitpack/fppara.f
    scipy/interpolate/fitpack/fppara.f:202.35:
    
            if(fpold-fp.gt.acc) npl1 = rn*fpms/(fpold-fp)
                                       1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/interpolate/fitpack/fppara.f: In function ‘fppara’:
    scipy/interpolate/fitpack/fppara.f:203:0: warning: ‘nplus’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             nplus = min0(nplus*2,max0(npl1,nplus/2,1))
     ^
    scipy/interpolate/fitpack/fppara.f:242:0: warning: ‘nmax’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(n.eq.nmax) go to 10
     ^
    scipy/interpolate/fitpack/fppara.f:202:0: warning: ‘fpold’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if(fpold-fp.gt.acc) npl1 = rn*fpms/(fpold-fp)
     ^
    scipy/interpolate/fitpack/fppara.f:279:0: warning: ‘fpms’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           f3 = fpms
     ^
    scipy/interpolate/fitpack/fppara.f:174:0: warning: ‘fp0’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             fpint(n) = fp0
     ^
    scipy/interpolate/fitpack/fppara.f:378:0: warning: ‘acc’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if((f1-f2).gt.acc) go to 345
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpadno.f
    gfortran:f77: scipy/interpolate/fitpack/fppasu.f
    scipy/interpolate/fitpack/fppasu.f:272.33:
    
            if(reducu.gt.acc) npl1 = rn*fpms/reducu
                                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/interpolate/fitpack/fppasu.f:279.33:
    
            if(reducv.gt.acc) npl1 = rn*fpms/reducv
                                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/interpolate/fitpack/fppasu.f: In function ‘fppasu’:
    scipy/interpolate/fitpack/fppasu.f:336:0: warning: ‘fpms’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           f3 = fpms
     ^
    scipy/interpolate/fitpack/fppasu.f:308:0: warning: ‘nve’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(nv.eq.nve) go to 250
     ^
    scipy/interpolate/fitpack/fppasu.f:295:0: warning: ‘nue’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(nu.eq.nue) go to 250
     ^
    scipy/interpolate/fitpack/fppasu.f:251:0: warning: ‘nmaxv’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if(nu.eq.nmaxu .and. nv.eq.nmaxv) go to 430
     ^
    scipy/interpolate/fitpack/fppasu.f:251:0: warning: ‘nmaxu’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/interpolate/fitpack/fppasu.f:367:0: warning: ‘acc’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if((f1-f2).gt.acc) go to 330
     ^
    scipy/interpolate/fitpack/fppasu.f:231:0: warning: ‘perv’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               tv(l3) = tv(l1)+perv
     ^
    scipy/interpolate/fitpack/fppasu.f:209:0: warning: ‘peru’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               tu(l3) = tu(l1)+peru
     ^
    gfortran:f77: scipy/interpolate/fitpack/fprank.f
    gfortran:f77: scipy/interpolate/fitpack/fpcoco.f
    scipy/interpolate/fitpack/fpcoco.f: In function ‘fpcoco’:
    scipy/interpolate/fitpack/fpcoco.f:137:0: warning: ‘k’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if(k.gt.l) k = k-1
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpopsp.f
    scipy/interpolate/fitpack/fpopsp.f:58.16:
    
          real*8 res,sq,sqq,sq0,sq1,step1,step2,three
                    1
    Warning: Unused variable 'res' declared at (1)
    gfortran:f77: scipy/interpolate/fitpack/parcur.f
    gfortran:f77: scipy/interpolate/fitpack/fporde.f
    gfortran:f77: scipy/interpolate/fitpack/parsur.f
    gfortran:f77: scipy/interpolate/fitpack/curfit.f
    gfortran:f77: scipy/interpolate/fitpack/fpopdi.f
    gfortran:f77: scipy/interpolate/fitpack/fpgivs.f
    scipy/interpolate/fitpack/fpgivs.f: In function ‘fpgivs’:
    scipy/interpolate/fitpack/fpgivs.f:16:0: warning: ‘dd’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           cos = ww/dd
     ^
    gfortran:f77: scipy/interpolate/fitpack/surev.f
    gfortran:f77: scipy/interpolate/fitpack/fpcyt2.f
    gfortran:f77: scipy/interpolate/fitpack/fpsphe.f
    scipy/interpolate/fitpack/fpsphe.f:390.4:
    
     440    do 450 i=1,nrint
        1
    Warning: Label 440 at (1) defined but not used
    scipy/interpolate/fitpack/fpsphe.f:327.4:
    
     330      in = nummer(in)
        1
    Warning: Label 330 at (1) defined but not used
    scipy/interpolate/fitpack/fpsphe.f: In function ‘fpsphe’:
    scipy/interpolate/fitpack/fpsphe.f:519:0: warning: ‘ntt’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           if(ntt.le.4) iband4 = ncof
     ^
    scipy/interpolate/fitpack/fpsphe.f:614:0: warning: ‘nt4’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                    if(il.ne.3 .and. il.ne.nt4) go to 750
     ^
    scipy/interpolate/fitpack/fpsphe.f:693:0: warning: ‘np4’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               jrot = lt*np4+lp
     ^
    scipy/interpolate/fitpack/fpsphe.f:746:0: warning: ‘lwest’ may be used uninitialized in this function [-Wmaybe-uninitialized]
      925  ier = lwest
     ^
    scipy/interpolate/fitpack/fpsphe.f:21:0: warning: ‘iband1’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           integer i,iband,iband1,iband3,iband4,ich1,ich3,ii,ij,il,in,irot,
     ^
    scipy/interpolate/fitpack/fpsphe.f:510:0: warning: ‘fpms’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           f3 = fpms
     ^
    scipy/interpolate/fitpack/fpsphe.f:730:0: warning: ‘acc’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if((f1-f2).gt.acc) go to 905
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpdisc.f
    gfortran:f77: scipy/interpolate/fitpack/splder.f
    scipy/interpolate/fitpack/splder.f:84.4:
    
      30  ier = 0
        1
    Warning: Label 30 at (1) defined but not used
    scipy/interpolate/fitpack/splder.f: In function ‘splder’:
    scipy/interpolate/fitpack/splder.f:135:0: warning: ‘k2’ may be used uninitialized in this function [-Wmaybe-uninitialized]
      65     if(arg.ge.t(l) .or. l+1.eq.k2) go to 70
     ^
    gfortran:f77: scipy/interpolate/fitpack/sphere.f
    scipy/interpolate/fitpack/sphere.f:318.10:
    
         * lbp,lco,lf,lff,lfp,lh,lq,lst,lsp,lwest,maxit,ncest,ncc,ntt,
              1
    Warning: Unused variable 'jlbp' declared at (1)
    gfortran:f77: scipy/interpolate/fitpack/fprati.f
    gfortran:f77: scipy/interpolate/fitpack/spgrid.f
    gfortran:f77: scipy/interpolate/fitpack/fppola.f
    scipy/interpolate/fitpack/fppola.f:440.4:
    
     440    do 450 i=1,nrint
        1
    Warning: Label 440 at (1) defined but not used
    scipy/interpolate/fitpack/fppola.f:377.4:
    
     370      in = nummer(in)
        1
    Warning: Label 370 at (1) defined but not used
    scipy/interpolate/fitpack/fppola.f:23.25:
    
         * iter,i1,i2,i3,j,jl,jrot,j1,j2,k,l,la,lf,lh,ll,lu,lv,lwest,l1,l2,
                             1
    Warning: Unused variable 'jl' declared at (1)
    scipy/interpolate/fitpack/fppola.f: In function ‘fppola’:
    scipy/interpolate/fitpack/fppola.f:768:0: warning: ‘nv4’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               jrot = lu*nv4+lv
     ^
    scipy/interpolate/fitpack/fppola.f:578:0: warning: ‘nu4’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           nuu = nu4-iopt3-1
     ^
    scipy/interpolate/fitpack/fppola.f:821:0: warning: ‘lwest’ may be used uninitialized in this function [-Wmaybe-uninitialized]
      925  ier = lwest
     ^
    scipy/interpolate/fitpack/fppola.f:25:0: warning: ‘iband1’ may be used uninitialized in this function [-Wmaybe-uninitialized]
          * numin,nvmin,rank,iband1
     ^
    scipy/interpolate/fitpack/fppola.f:565:0: warning: ‘fpms’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           f3 = fpms
     ^
    scipy/interpolate/fitpack/fppola.f:805:0: warning: ‘acc’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if((f1-f2).gt.acc) go to 905
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpcsin.f
    gfortran:f77: scipy/interpolate/fitpack/fptrnp.f
    scipy/interpolate/fitpack/fptrnp.f: In function ‘fptrnp’:
    scipy/interpolate/fitpack/fptrnp.f:53:0: warning: ‘pinv’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               h(j) = b(n1,j)*pinv
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpbfout.f
    scipy/interpolate/fitpack/fpbfout.f: In function ‘fpbfou’:
    scipy/interpolate/fitpack/fpbfout.f:117:0: warning: ‘term’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             c2 = (hc(5)-hc(4))*term
     ^
    gfortran:f77: scipy/interpolate/fitpack/insert.f
    gfortran:f77: scipy/interpolate/fitpack/fpseno.f
    gfortran:f77: scipy/interpolate/fitpack/clocur.f
    gfortran:f77: scipy/interpolate/fitpack/fpgrre.f
    scipy/interpolate/fitpack/fpgrre.f: In function ‘fpgrre’:
    scipy/interpolate/fitpack/fpgrre.f:130:0: warning: ‘pinv’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               h(j) = bx(n1,j)*pinv
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpader.f
    gfortran:f77: scipy/interpolate/fitpack/fpgrdi.f
    scipy/interpolate/fitpack/fpgrdi.f:296.4:
    
     400    if(nrold.eq.number) go to 420
        1
    Warning: Label 400 at (1) defined but not used
    scipy/interpolate/fitpack/fpgrdi.f: In function ‘fpgrdi’:
    scipy/interpolate/fitpack/fpgrdi.f:204:0: warning: ‘pinv’ may be used uninitialized in this function [-Wmaybe-uninitialized]
              bb(2,i) = fac*pinv
     ^
    gfortran:f77: scipy/interpolate/fitpack/profil.f
    gfortran:f77: scipy/interpolate/fitpack/fptrpe.f
    scipy/interpolate/fitpack/fptrpe.f:17.21:
    
          integer i,iband,irot,it,ii,i2,i3,j,jj,l,mid,nmd,m2,m3,
                         1
    Warning: Unused variable 'iband' declared at (1)
    scipy/interpolate/fitpack/fptrpe.f: In function ‘fptrpe’:
    scipy/interpolate/fitpack/fptrpe.f:64:0: warning: ‘pinv’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               h(j) = b(n1,j)*pinv
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpfrno.f
    scipy/interpolate/fitpack/fpfrno.f: In function ‘fpfrno’:
    scipy/interpolate/fitpack/fpfrno.f:42:0: warning: ‘k’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           right(k) = count
     ^
    gfortran:f77: scipy/interpolate/fitpack/fpclos.f
    scipy/interpolate/fitpack/fpclos.f:395.35:
    
            if(fpold-fp.gt.acc) npl1 = rn*fpms/(fpold-fp)
                                       1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/interpolate/fitpack/fpclos.f: In function ‘fpclos’:
    scipy/interpolate/fitpack/fpclos.f:396:0: warning: ‘nplus’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             nplus = min0(nplus*2,max0(npl1,nplus/2,1))
     ^
    scipy/interpolate/fitpack/fpclos.f:438:0: warning: ‘nmax’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               if(n.eq.nmax) go to 5
     ^
    scipy/interpolate/fitpack/fpclos.f:473:0: warning: ‘n10’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           n11 = n10-1
     ^
    scipy/interpolate/fitpack/fpclos.f:16:0: warning: ‘i1’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           integer i,ich1,ich3,ij,ik,it,iter,i1,i2,i3,j,jj,jk,jper,j1,j2,kk,
     ^
    scipy/interpolate/fitpack/fpclos.f:395:0: warning: ‘fpold’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if(fpold-fp.gt.acc) npl1 = rn*fpms/(fpold-fp)
     ^
    scipy/interpolate/fitpack/fpclos.f:472:0: warning: ‘fpms’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           f3 = fpms
     ^
    scipy/interpolate/fitpack/fpclos.f:470:0: warning: ‘fp0’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           f1 = fp0-s
     ^
    scipy/interpolate/fitpack/fpclos.f:663:0: warning: ‘acc’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if((f1-f2) .gt. acc) go to 585
     ^
    ar: adding 50 object files to build/temp.linux-x86_64-2.7/libfitpack.a
    ar: adding 34 object files to build/temp.linux-x86_64-2.7/libfitpack.a
    building 'odrpack' library
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/odr/odrpack/d_odr.f
    scipy/odr/odrpack/d_odr.f:1014.13:
    
          NETA = MAX(TWO,P5-LOG10(ETA))
                 1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/odr/odrpack/d_odr.f:2955.13:
    
          NTOL = MAX(ONE,P5-LOG10(TOL))
                 1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/odr/odrpack/d_odr.f:6032.16:
    
                J = WORK(WRK3+I) - 1
                    1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/odr/odrpack/d_mprec.f
    gfortran:f77: scipy/odr/odrpack/dlunoc.f
    gfortran:f77: scipy/odr/odrpack/d_lpk.f
    ar: adding 4 object files to build/temp.linux-x86_64-2.7/libodrpack.a
    building 'minpack' library
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/optimize/minpack/lmpar.f
    gfortran:f77: scipy/optimize/minpack/lmstr.f
    scipy/optimize/minpack/lmstr.f: In function ‘lmstr’:
    scipy/optimize/minpack/lmstr.f:434:0: warning: ‘xnorm’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                 if (delta .le. xtol*xnorm) info = 2
     ^
    gfortran:f77: scipy/optimize/minpack/fdjac1.f
    gfortran:f77: scipy/optimize/minpack/lmder1.f
    gfortran:f77: scipy/optimize/minpack/hybrj1.f
    gfortran:f77: scipy/optimize/minpack/fdjac2.f
    gfortran:f77: scipy/optimize/minpack/lmdif1.f
    gfortran:f77: scipy/optimize/minpack/rwupdt.f
    gfortran:f77: scipy/optimize/minpack/hybrj.f
    scipy/optimize/minpack/hybrj.f: In function ‘hybrj’:
    scipy/optimize/minpack/hybrj.f:386:0: warning: ‘xnorm’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                 if (delta .le. xtol*xnorm .or. fnorm .eq. zero) info = 1
     ^
    gfortran:f77: scipy/optimize/minpack/enorm.f
    scipy/optimize/minpack/enorm.f: In function ‘enorm’:
    scipy/optimize/minpack/enorm.f:1:0: warning: ‘__result_enorm’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           double precision function enorm(n,x)
     ^
    gfortran:f77: scipy/optimize/minpack/r1mpyq.f
    scipy/optimize/minpack/r1mpyq.f: In function ‘r1mpyq’:
    scipy/optimize/minpack/r1mpyq.f:64:0: warning: ‘cos’ may be used uninitialized in this function [-Wmaybe-uninitialized]
              if (dabs(v(j)) .gt. one) sin = dsqrt(one-cos**2)
     ^
    gfortran:f77: scipy/optimize/minpack/dogleg.f
    gfortran:f77: scipy/optimize/minpack/hybrd.f
    scipy/optimize/minpack/hybrd.f: In function ‘hybrd’:
    scipy/optimize/minpack/hybrd.f:404:0: warning: ‘xnorm’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                 if (delta .le. xtol*xnorm .or. fnorm .eq. zero) info = 1
     ^
    gfortran:f77: scipy/optimize/minpack/lmstr1.f
    gfortran:f77: scipy/optimize/minpack/qrfac.f
    gfortran:f77: scipy/optimize/minpack/qrsolv.f
    gfortran:f77: scipy/optimize/minpack/r1updt.f
    gfortran:f77: scipy/optimize/minpack/dpmpar.f
    gfortran:f77: scipy/optimize/minpack/chkder.f
    gfortran:f77: scipy/optimize/minpack/lmder.f
    scipy/optimize/minpack/lmder.f: In function ‘lmder’:
    scipy/optimize/minpack/lmder.f:420:0: warning: ‘xnorm’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                 if (delta .le. xtol*xnorm) info = 2
     ^
    scipy/optimize/minpack/lmder.f:387:0: warning: ‘temp’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                    if (p1*fnorm1 .ge. fnorm .or. temp .lt. p1) temp = p1
     ^
    gfortran:f77: scipy/optimize/minpack/lmdif.f
    scipy/optimize/minpack/lmdif.f: In function ‘lmdif’:
    scipy/optimize/minpack/lmdif.f:422:0: warning: ‘xnorm’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                 if (delta .le. xtol*xnorm) info = 2
     ^
    scipy/optimize/minpack/lmdif.f:389:0: warning: ‘temp’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                    if (p1*fnorm1 .ge. fnorm .or. temp .lt. p1) temp = p1
     ^
    gfortran:f77: scipy/optimize/minpack/hybrd1.f
    gfortran:f77: scipy/optimize/minpack/qform.f
    ar: adding 23 object files to build/temp.linux-x86_64-2.7/libminpack.a
    building 'rootfind' library
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    x86_64-linux-gnu-gcc: scipy/optimize/Zeros/ridder.c
    x86_64-linux-gnu-gcc: scipy/optimize/Zeros/brentq.c
    x86_64-linux-gnu-gcc: scipy/optimize/Zeros/bisect.c
    In file included from scipy/optimize/Zeros/bisect.c:3:0:
    scipy/optimize/Zeros/zeros.h:16:15: warning: ‘dminarg1’ defined but not used [-Wunused-variable]
     static double dminarg1,dminarg2;
                   ^
    scipy/optimize/Zeros/zeros.h:16:24: warning: ‘dminarg2’ defined but not used [-Wunused-variable]
     static double dminarg1,dminarg2;
                            ^
    x86_64-linux-gnu-gcc: scipy/optimize/Zeros/brenth.c
    ar: adding 4 object files to build/temp.linux-x86_64-2.7/librootfind.a
    building 'superlu_src' library
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DUSE_VENDOR_BLAS=1 -Iscipy/sparse/linalg/dsolve/SuperLU/SRC -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/heap_relax_snode.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/heap_relax_snode.c:23:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dzsum1.c
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/slamch.c
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slamch.c: In function ‘slamc2_’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slamch.c:424:16: warning: unused variable ‘c__1’ [-Wunused-variable]
         static int c__1 = 1;
                    ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slamch.c: In function ‘slamc4_’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slamch.c:734:9: warning: variable ‘i__1’ set but not used [-Wunused-but-set-variable]
         int i__1;
             ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slamch.c: In function ‘slamc5_’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slamch.c:849:9: warning: variable ‘i__1’ set but not used [-Wunused-but-set-variable]
         int i__1;
             ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dgssv.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dgssv.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgssv.c: In function ‘dgssv’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:61:37: warning: ‘AA’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_FREE(addr) USER_FREE(addr)
                                         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgssv.c:139:18: note: ‘AA’ was declared here
         SuperMatrix *AA;/* A in SLU_NC format used by the factorization routine.*/
                      ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cpivotgrowth.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cpivotgrowth.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cpivotgrowth.c: In function ‘cPivotGrowth’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cpivotgrowth.c:63:15: warning: unused variable ‘temp_comp’ [-Wunused-variable]
         complex   temp_comp;
                   ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zpanel_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zpanel_dfs.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dldperm.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dldperm.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsequ.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsequ.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsrfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsrfs.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_heap_relax_snode.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_heap_relax_snode.c:11:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_scolumn_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_scolumn_dfs.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dpruneL.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dpruneL.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zmemory.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zmemory.c:11:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zmemory.c: In function ‘zLUMemXpand’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zmemory.c:444:5: warning: enumeration value ‘LLVL’ not handled in switch [-Wswitch]
         switch ( mem_type ) {
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zmemory.c:444:5: warning: enumeration value ‘ULVL’ not handled in switch [-Wswitch]
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/izmax1.c
    scipy/sparse/linalg/dsolve/SuperLU/SRC/izmax1.c: In function ‘izmax1_’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/izmax1.c:51:24: warning: variable ‘i__2’ set but not used [-Wunused-but-set-variable]
         int ret_val, i__1, i__2;
                            ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/izmax1.c:51:18: warning: variable ‘i__1’ set but not used [-Wunused-but-set-variable]
         int ret_val, i__1, i__2;
                      ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsitrf.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsitrf.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/scolumn_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/scolumn_dfs.c:24:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/scolumn_dfs.c: In function ‘scolumn_dfs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/scolumn_dfs.c:133:3: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
       if ( mem_error = sLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu) )
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/scolumn_dfs.c:176:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           sLUMemXpand(jcol,nextl,LSUB,&nzlmax,Glu) )
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zcolumn_bmod.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zcolumn_bmod.c:26:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zcolumn_bmod.c: In function ‘zcolumn_bmod’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zcolumn_bmod.c:234:3: warning: implicit declaration of function ‘ztrsv_’ [-Wimplicit-function-declaration]
       ztrsv_( "L", "N", "U", &segsze, &lusup[luptr],
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zcolumn_bmod.c:245:3: warning: implicit declaration of function ‘zgemv_’ [-Wimplicit-function-declaration]
       zgemv_( "N", &nrow, &segsze, &alpha, &lusup[luptr],
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zcolumn_bmod.c:288:2: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
      if (mem_error = zLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu))
      ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dcolumn_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dcolumn_dfs.c:24:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dcolumn_dfs.c: In function ‘dcolumn_dfs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dcolumn_dfs.c:133:3: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
       if ( mem_error = dLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu) )
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dcolumn_dfs.c:176:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           dLUMemXpand(jcol,nextl,LSUB,&nzlmax,Glu) )
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_dpivotL.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_dpivotL.c:15:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_csnode_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_csnode_dfs.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ccolumn_bmod.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ccolumn_bmod.c:26:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ccolumn_bmod.c: In function ‘ccolumn_bmod’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ccolumn_bmod.c:232:3: warning: implicit declaration of function ‘ctrsv_’ [-Wimplicit-function-declaration]
       ctrsv_( "L", "N", "U", &segsze, &lusup[luptr],
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ccolumn_bmod.c:243:3: warning: implicit declaration of function ‘cgemv_’ [-Wimplicit-function-declaration]
       cgemv_( "N", &nrow, &segsze, &alpha, &lusup[luptr],
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ccolumn_bmod.c:286:2: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
      if (mem_error = cLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu))
      ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dcopy_to_ucol.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dcopy_to_ucol.c:23:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dcopy_to_ucol.c: In function ‘dcopy_to_ucol’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dcopy_to_ucol.c:77:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           if (mem_error = dLUMemXpand(jcol, nextu, UCOL, &nzumax, Glu))
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dcopy_to_ucol.c:80:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           if (mem_error = dLUMemXpand(jcol, nextu, USUB, &nzumax, Glu))
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/claqgs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/claqgs.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sldperm.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sldperm.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dlamch.c
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dlamch.c: In function ‘dlamc2_’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dlamch.c:417:16: warning: unused variable ‘c__1’ [-Wunused-variable]
         static int c__1 = 1;
                    ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dlamch.c: In function ‘dlamc4_’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dlamch.c:722:9: warning: variable ‘i__1’ set but not used [-Wunused-but-set-variable]
         int i__1;
             ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dlamch.c: In function ‘dlamc5_’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dlamch.c:835:9: warning: variable ‘i__1’ set but not used [-Wunused-but-set-variable]
         int i__1;
             ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_dcolumn_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_dcolumn_dfs.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zgstrf.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zgstrf.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zlangs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zlangs.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zlangs.c: In function ‘zlangs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zlangs.c:116:5: warning: ‘value’ may be used uninitialized in this function [-Wmaybe-uninitialized]
         return (value);
         ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/get_perm_c.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/get_perm_c.c:11:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/get_perm_c.c: In function ‘get_perm_c’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/get_perm_c.c:372:5: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
         double t, SuperLU_timer_();
         ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/superlu_timer.c
    scipy/sparse/linalg/dsolve/SuperLU/SRC/superlu_timer.c:51:8: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     double SuperLU_timer_()
            ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zpivotL.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zpivotL.c:27:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dmemory.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dmemory.c:11:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dmemory.c: In function ‘dLUMemXpand’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dmemory.c:444:5: warning: enumeration value ‘LLVL’ not handled in switch [-Wswitch]
         switch ( mem_type ) {
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dmemory.c:444:5: warning: enumeration value ‘ULVL’ not handled in switch [-Wswitch]
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/csp_blas2.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/csp_blas2.c:17:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/csp_blas2.c: In function ‘sp_ctrsv’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/csp_blas2.c:159:7: warning: implicit declaration of function ‘ctrsv_’ [-Wimplicit-function-declaration]
           ctrsv_("L", "N", "U", &nsupc, &Lval[luptr], &nsupr,
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/csp_blas2.c:162:7: warning: implicit declaration of function ‘cgemv_’ [-Wimplicit-function-declaration]
           cgemv_("N", &nrow, &nsupc, &alpha, &Lval[luptr+nsupc],
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/csp_blas2.c: In function ‘sp_cgemv’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/csp_blas2.c:488:27: warning: suggest parentheses around ‘&&’ within ‘||’ [-Wparentheses]
      c_eq(&alpha, &comp_zero) &&
                               ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/qselect.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/qselect.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cdiagonal.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cdiagonal.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sp_coletree.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sp_coletree.c:27:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sp_coletree.c:243:6: warning: ‘etdfs’ defined but not used [-Wunused-function]
     void etdfs (
          ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dcolumn_bmod.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dcolumn_bmod.c:26:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dcolumn_bmod.c: In function ‘dcolumn_bmod’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dcolumn_bmod.c:219:3: warning: implicit declaration of function ‘dtrsv_’ [-Wimplicit-function-declaration]
       dtrsv_( "L", "N", "U", &segsze, &lusup[luptr],
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dcolumn_bmod.c:230:3: warning: implicit declaration of function ‘dgemv_’ [-Wimplicit-function-declaration]
       dgemv_( "N", &nrow, &segsze, &alpha, &lusup[luptr],
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dcolumn_bmod.c:273:2: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
      if (mem_error = dLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu))
      ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsrfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsrfs.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cpanel_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cpanel_dfs.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cldperm.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cldperm.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zcolumn_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zcolumn_dfs.c:24:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zcolumn_dfs.c: In function ‘zcolumn_dfs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zcolumn_dfs.c:133:3: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
       if ( mem_error = zLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu) )
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zcolumn_dfs.c:176:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           zLUMemXpand(jcol,nextl,LSUB,&nzlmax,Glu) )
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/slangs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/slangs.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slangs.c: In function ‘slangs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slangs.c:116:5: warning: ‘value’ may be used uninitialized in this function [-Wmaybe-uninitialized]
         return (value);
         ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsisx.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsisx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsisx.c: In function ‘sgsisx’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsisx.c:578:7: warning: suggest parentheses around operand of ‘!’ or change ‘&’ to ‘&&’ or ‘!’ to ‘~’ [-Wparentheses]
      if ( !mc64 & equil ) { /* Only perform equilibration, no row perm */
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsisx.c:667:16: warning: unused variable ‘rhs_work’ [-Wunused-variable]
             float *rhs_work;
                    ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsisx.c:413:14: warning: variable ‘diag_pivot_thresh’ set but not used [-Wunused-but-set-variable]
         float    diag_pivot_thresh;
                  ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsisx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:69:46: warning: ‘smlnum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MAX(x, y)  ( (x) > (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsisx.c:411:35: note: ‘smlnum’ was declared here
         float    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                                       ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsisx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:70:46: warning: ‘bignum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MIN(x, y)  ( (x) < (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsisx.c:411:27: note: ‘bignum’ was declared here
         float    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                               ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadtriple.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadtriple.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadtriple.c: In function ‘dreadrhs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadtriple.c:127:5: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
         FILE *fp, *fopen();
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadtriple.c: In function ‘dreadtriple’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadtriple.c:38:10: warning: ignoring return value of ‘scanf’, declared with attribute warn_unused_result [-Wunused-result]
         scanf("%d%d", n, nonz);
              ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadtriple.c:54:7: warning: ignoring return value of ‘scanf’, declared with attribute warn_unused_result [-Wunused-result]
      scanf("%d%d%lf\n", &row[nz], &col[nz], &val[nz]);
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadtriple.c: In function ‘dreadrhs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadtriple.c:136:13: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
           fscanf(fp, "%lf\n", &b[i]);
                 ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/spivotgrowth.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/spivotgrowth.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/clangs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/clangs.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/clangs.c: In function ‘clangs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/clangs.c:116:5: warning: ‘value’ may be used uninitialized in this function [-Wmaybe-uninitialized]
         return (value);
         ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsisx.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsisx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsisx.c: In function ‘cgsisx’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsisx.c:578:7: warning: suggest parentheses around operand of ‘!’ or change ‘&’ to ‘&&’ or ‘!’ to ‘~’ [-Wparentheses]
      if ( !mc64 & equil ) { /* Only perform equilibration, no row perm */
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsisx.c:667:18: warning: unused variable ‘rhs_work’ [-Wunused-variable]
             complex *rhs_work;
                      ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsisx.c:413:14: warning: variable ‘diag_pivot_thresh’ set but not used [-Wunused-but-set-variable]
         float    diag_pivot_thresh;
                  ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsisx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:69:46: warning: ‘smlnum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MAX(x, y)  ( (x) > (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsisx.c:411:35: note: ‘smlnum’ was declared here
         float    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                                       ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsisx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:70:46: warning: ‘bignum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MIN(x, y)  ( (x) < (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsisx.c:411:27: note: ‘bignum’ was declared here
         float    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                               ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zsnode_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zsnode_dfs.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zsnode_dfs.c: In function ‘zsnode_dfs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zsnode_dfs.c:81:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           if ( mem_error = zLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu) )
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zsnode_dfs.c:94:6: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
          if ( mem_error = zLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu) )
          ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zldperm.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zldperm.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_ccopy_to_ucol.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_ccopy_to_ucol.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_ccopy_to_ucol.c: In function ‘ilu_ccopy_to_ucol’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_ccopy_to_ucol.c:66:5: warning: implicit declaration of function ‘dlamch_’ [-Wimplicit-function-declaration]
         register float d_max = 0.0, d_min = 1.0 / dlamch_("Safe minimum");
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_ccopy_to_ucol.c:183:11: warning: ‘tmp’ may be used uninitialized in this function [-Wmaybe-uninitialized]
        sum->r += tmp;
               ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ssp_blas2.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ssp_blas2.c:17:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ssp_blas2.c: In function ‘sp_strsv’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ssp_blas2.c:155:7: warning: implicit declaration of function ‘strsv_’ [-Wimplicit-function-declaration]
           strsv_("L", "N", "U", &nsupc, &Lval[luptr], &nsupr,
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ssp_blas2.c:158:7: warning: implicit declaration of function ‘sgemv_’ [-Wimplicit-function-declaration]
           sgemv_("N", &nrow, &nsupc, &alpha, &Lval[luptr+nsupc],
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:79:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c: In function ‘zreadhb’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:191:29: warning: unused variable ‘key’ [-Wunused-variable]
         char buf[100], type[4], key[10];
                                 ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c: In function ‘zReadValues’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:159:7: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
      fgets(buf, 100, fp);    /* read a line at a time */
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c: In function ‘zreadhb’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:197:10: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
         fgets(buf, 100, fp);
              ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:209:8: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
      fscanf(fp, "%14c", buf); buf[14] = 0;
            ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:217:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%3c", type);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:218:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%11c", buf); /* pad */
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:224:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nrow);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:225:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", ncol);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:226:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nonz);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:227:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", &tmp);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:239:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:241:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:243:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%20c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:245:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%20c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c: In function ‘ReadVector’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadhb.c:137:7: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
      fgets(buf, 100, fp);    /* read a line at a time */
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c:72:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c: In function ‘zreadrb’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c:190:10: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
         fgets(buf, 100, fp);
              ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c:195:15: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
             fscanf(fp, "%14c", buf); buf[14] = 0;
                   ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c:202:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%3c", type);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c:203:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%11c", buf); /* pad */
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c:209:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nrow);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c:210:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", ncol);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c:211:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nonz);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c:212:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", &tmp);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c:224:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c:226:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c:228:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%20c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c: In function ‘ReadVector’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c:130:14: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
             fgets(buf, 100, fp);    /* read a line at a time */
                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c: In function ‘zReadValues’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadrb.c:152:7: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
      fgets(buf, 100, fp);    /* read a line at a time */
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_relax_snode.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_relax_snode.c:11:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dpivotL.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dpivotL.c:27:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsequ.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsequ.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/mark_relax.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/mark_relax.c:10:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zcopy_to_ucol.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zcopy_to_ucol.c:23:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zcopy_to_ucol.c: In function ‘zcopy_to_ucol’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zcopy_to_ucol.c:77:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           if (mem_error = zLUMemXpand(jcol, nextu, UCOL, &nzumax, Glu))
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zcopy_to_ucol.c:80:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           if (mem_error = zLUMemXpand(jcol, nextu, USUB, &nzumax, Glu))
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/xerbla.c
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/spanel_bmod.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/spanel_bmod.c:29:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/spanel_bmod.c:36:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern void scheck_tempv();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/spanel_bmod.c: In function ‘spanel_bmod’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/spanel_bmod.c:224:7: warning: implicit declaration of function ‘strsv_’ [-Wimplicit-function-declaration]
           strsv_( "L", "N", "U", &segsze, &lusup[luptr],
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/spanel_bmod.c:271:7: warning: implicit declaration of function ‘sgemv_’ [-Wimplicit-function-declaration]
           sgemv_("N", &block_nrow, &segsze, &alpha, &lusup[luptr1],
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zdrop_row.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zdrop_row.c:14:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zdrop_row.c: In function ‘ilu_zdrop_row’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zdrop_row.c:186:3: warning: implicit declaration of function ‘dcopy_’ [-Wimplicit-function-declaration]
       dcopy_(&len, dwork, &i_1, dwork2, &i_1);
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zdrop_row.c:75:19: warning: unused variable ‘zero’ [-Wunused-variable]
         doublecomplex zero = {0.0, 0.0};
                       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zdrop_row.c: At top level:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zdrop_row.c:25:12: warning: ‘_compare_’ defined but not used [-Wunused-function]
     static int _compare_(const void *a, const void *b)
                ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:79:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c: In function ‘dreadhb’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:181:29: warning: unused variable ‘key’ [-Wunused-variable]
         char buf[100], type[4], key[10];
                                 ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c: In function ‘dReadValues’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:157:7: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
      fgets(buf, 100, fp);    /* read a line at a time */
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c: In function ‘dreadhb’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:187:10: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
         fgets(buf, 100, fp);
              ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:199:8: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
      fscanf(fp, "%14c", buf); buf[14] = 0;
            ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:207:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%3c", type);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:208:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%11c", buf); /* pad */
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:214:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nrow);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:215:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", ncol);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:216:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nonz);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:217:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", &tmp);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:229:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:231:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:233:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%20c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:235:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%20c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c: In function ‘ReadVector’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadhb.c:137:7: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
      fgets(buf, 100, fp);    /* read a line at a time */
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dpanel_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dpanel_dfs.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_sdrop_row.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_sdrop_row.c:14:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_sdrop_row.c:25:12: warning: ‘_compare_’ defined but not used [-Wunused-function]
     static int _compare_(const void *a, const void *b)
                ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zpanel_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zpanel_dfs.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dutil.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dutil.c:26:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dutil.c: In function ‘dFillRHS’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dutil.c:358:15: warning: variable ‘Aval’ set but not used [-Wunused-but-set-variable]
         double   *Aval;
                   ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dutil.c: At top level:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dutil.c:464:1: warning: return type defaults to ‘int’ [-Wreturn-type]
     print_double_vec(char *what, int n, double *vec)
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dcomplex.c
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dlangs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dlangs.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dlangs.c: In function ‘dlangs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dlangs.c:116:5: warning: ‘value’ may be used uninitialized in this function [-Wmaybe-uninitialized]
         return (value);
         ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_dcopy_to_ucol.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_dcopy_to_ucol.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ccolumn_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ccolumn_dfs.c:24:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ccolumn_dfs.c: In function ‘ccolumn_dfs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ccolumn_dfs.c:133:3: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
       if ( mem_error = cLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu) )
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ccolumn_dfs.c:176:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           cLUMemXpand(jcol,nextl,LSUB,&nzlmax,Glu) )
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dsnode_bmod.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dsnode_bmod.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dsnode_bmod.c: In function ‘dsnode_bmod’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dsnode_bmod.c:97:2: warning: implicit declaration of function ‘dtrsv_’ [-Wimplicit-function-declaration]
      dtrsv_( "L", "N", "U", &nsupc, &lusup[luptr], &nsupr,
      ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dsnode_bmod.c:99:2: warning: implicit declaration of function ‘dgemv_’ [-Wimplicit-function-declaration]
      dgemv_( "N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr,
      ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dsnode_bmod.c:52:35: warning: unused variable ‘iptr’ [-Wunused-variable]
         int            isub, irow, i, iptr;
                                       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dsnode_bmod.c:52:32: warning: unused variable ‘i’ [-Wunused-variable]
         int            isub, irow, i, iptr;
                                    ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/util.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/util.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/util.c: In function ‘ilu_countnz’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/util.c:214:24: warning: variable ‘irep’ set but not used [-Wunused-but-set-variable]
         int          jlen, irep;
                            ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cpivotL.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cpivotL.c:27:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zsnode_bmod.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zsnode_bmod.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zsnode_bmod.c: In function ‘zsnode_bmod’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zsnode_bmod.c:98:2: warning: implicit declaration of function ‘ztrsv_’ [-Wimplicit-function-declaration]
      ztrsv_( "L", "N", "U", &nsupc, &lusup[luptr], &nsupr,
      ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zsnode_bmod.c:100:2: warning: implicit declaration of function ‘zgemv_’ [-Wimplicit-function-declaration]
      zgemv_( "N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr,
      ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zsnode_bmod.c:53:35: warning: unused variable ‘iptr’ [-Wunused-variable]
         int            isub, irow, i, iptr;
                                       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zsnode_bmod.c:53:32: warning: unused variable ‘i’ [-Wunused-variable]
         int            isub, irow, i, iptr;
                                    ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zsp_blas2.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zsp_blas2.c:17:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zsp_blas2.c: In function ‘sp_ztrsv’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zsp_blas2.c:159:7: warning: implicit declaration of function ‘ztrsv_’ [-Wimplicit-function-declaration]
           ztrsv_("L", "N", "U", &nsupc, &Lval[luptr], &nsupr,
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zsp_blas2.c:162:7: warning: implicit declaration of function ‘zgemv_’ [-Wimplicit-function-declaration]
           zgemv_("N", &nrow, &nsupc, &alpha, &Lval[luptr+nsupc],
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zsp_blas2.c: In function ‘sp_zgemv’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zsp_blas2.c:488:27: warning: suggest parentheses around ‘&&’ within ‘||’ [-Wparentheses]
      z_eq(&alpha, &comp_zero) &&
                               ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ddiagonal.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ddiagonal.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cgssv.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cgssv.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgssv.c: In function ‘cgssv’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:61:37: warning: ‘AA’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_FREE(addr) USER_FREE(addr)
                                         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgssv.c:139:18: note: ‘AA’ was declared here
         SuperMatrix *AA;/* A in SLU_NC format used by the factorization routine.*/
                      ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dgscon.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dgscon.c:20:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsequ.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsequ.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsitrf.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsitrf.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zlacon.c
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zlacon.c: In function ‘zlacon_’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zlacon.c:157:5: warning: implicit declaration of function ‘zcopy_’ [-Wimplicit-function-declaration]
         zcopy_(n, x, &c__1, v, &c__1);
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zlacon.c:163:1: warning: label ‘L90’ defined but not used [-Wunused-label]
     L90:
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/relax_snode.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/relax_snode.c:23:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zutil.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zutil.c:26:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zutil.c: In function ‘zFillRHS’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zutil.c:360:22: warning: variable ‘Aval’ set but not used [-Wunused-but-set-variable]
         doublecomplex   *Aval;
                          ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zutil.c: At top level:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zutil.c:468:1: warning: return type defaults to ‘int’ [-Wreturn-type]
     print_doublecomplex_vec(char *what, int n, doublecomplex *vec)
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zcopy_to_ucol.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zcopy_to_ucol.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zcopy_to_ucol.c: In function ‘ilu_zcopy_to_ucol’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zcopy_to_ucol.c:183:11: warning: ‘tmp’ may be used uninitialized in this function [-Wmaybe-uninitialized]
        sum->r += tmp;
               ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/mmd.c
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cmemory.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cmemory.c:11:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cmemory.c: In function ‘cLUMemXpand’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cmemory.c:444:5: warning: enumeration value ‘LLVL’ not handled in switch [-Wswitch]
         switch ( mem_type ) {
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cmemory.c:444:5: warning: enumeration value ‘ULVL’ not handled in switch [-Wswitch]
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ssnode_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ssnode_dfs.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ssnode_dfs.c: In function ‘ssnode_dfs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ssnode_dfs.c:81:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           if ( mem_error = sLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu) )
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ssnode_dfs.c:94:6: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
          if ( mem_error = sLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu) )
          ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/clacon.c
    scipy/sparse/linalg/dsolve/SuperLU/SRC/clacon.c: In function ‘clacon_’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/clacon.c:157:5: warning: implicit declaration of function ‘ccopy_’ [-Wimplicit-function-declaration]
         ccopy_(n, x, &c__1, v, &c__1);
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/clacon.c:163:1: warning: label ‘L90’ defined but not used [-Wunused-label]
     L90:
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cpanel_bmod.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cpanel_bmod.c:29:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cpanel_bmod.c:36:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern void ccheck_tempv();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cpanel_bmod.c: In function ‘cpanel_bmod’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cpanel_bmod.c:238:7: warning: implicit declaration of function ‘ctrsv_’ [-Wimplicit-function-declaration]
           ctrsv_( "L", "N", "U", &segsze, &lusup[luptr],
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cpanel_bmod.c:285:7: warning: implicit declaration of function ‘cgemv_’ [-Wimplicit-function-declaration]
           cgemv_("N", &block_nrow, &segsze, &alpha, &lusup[luptr1],
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsrfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsrfs.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ccopy_to_ucol.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ccopy_to_ucol.c:23:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ccopy_to_ucol.c: In function ‘ccopy_to_ucol’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ccopy_to_ucol.c:77:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           if (mem_error = cLUMemXpand(jcol, nextu, UCOL, &nzumax, Glu))
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ccopy_to_ucol.c:80:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           if (mem_error = cLUMemXpand(jcol, nextu, USUB, &nzumax, Glu))
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsrfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sgsrfs.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/slacon.c
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsequ.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsequ.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsitrf.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsitrf.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsitrf.c: In function ‘zgsitrf’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsitrf.c:229:12: warning: unused variable ‘one’ [-Wunused-variable]
         double one = 1.0;
                ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsitrf.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsitrf.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsitrf.c: In function ‘cgsitrf’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgsitrf.c:229:11: warning: unused variable ‘one’ [-Wunused-variable]
         float one = 1.0;
               ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_dsnode_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_dsnode_dfs.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sp_ienv.c
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sp_ienv.c: In function ‘sp_ienv’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sp_ienv.c:75:5: warning: implicit declaration of function ‘xerbla_’ [-Wimplicit-function-declaration]
         xerbla_("sp_ienv", &i);
         ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sgssv.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sgssv.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgssv.c: In function ‘sgssv’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:61:37: warning: ‘AA’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_FREE(addr) USER_FREE(addr)
                                         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgssv.c:139:18: note: ‘AA’ was declared here
         SuperMatrix *AA;/* A in SLU_NC format used by the factorization routine.*/
                      ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/csnode_bmod.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/csnode_bmod.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/csnode_bmod.c: In function ‘csnode_bmod’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/csnode_bmod.c:98:2: warning: implicit declaration of function ‘ctrsv_’ [-Wimplicit-function-declaration]
      ctrsv_( "L", "N", "U", &nsupc, &lusup[luptr], &nsupr,
      ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/csnode_bmod.c:100:2: warning: implicit declaration of function ‘cgemv_’ [-Wimplicit-function-declaration]
      cgemv_( "N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr,
      ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/csnode_bmod.c:53:35: warning: unused variable ‘iptr’ [-Wunused-variable]
         int            isub, irow, i, iptr;
                                       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/csnode_bmod.c:53:32: warning: unused variable ‘i’ [-Wunused-variable]
         int            isub, irow, i, iptr;
                                    ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/memory.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/memory.c:14:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sgstrf.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sgstrf.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/scolumn_bmod.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/scolumn_bmod.c:26:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/scolumn_bmod.c: In function ‘scolumn_bmod’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/scolumn_bmod.c:219:3: warning: implicit declaration of function ‘strsv_’ [-Wimplicit-function-declaration]
       strsv_( "L", "N", "U", &segsze, &lusup[luptr],
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/scolumn_bmod.c:230:3: warning: implicit declaration of function ‘sgemv_’ [-Wimplicit-function-declaration]
       sgemv_( "N", &nrow, &segsze, &alpha, &lusup[luptr],
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/scolumn_bmod.c:273:2: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
      if (mem_error = sLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu))
      ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sgscon.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sgscon.c:20:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zpanel_bmod.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zpanel_bmod.c:29:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zpanel_bmod.c:36:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern void zcheck_tempv();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zpanel_bmod.c: In function ‘zpanel_bmod’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zpanel_bmod.c:238:7: warning: implicit declaration of function ‘ztrsv_’ [-Wimplicit-function-declaration]
           ztrsv_( "L", "N", "U", &segsze, &lusup[luptr],
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zpanel_bmod.c:285:7: warning: implicit declaration of function ‘zgemv_’ [-Wimplicit-function-declaration]
           zgemv_("N", &block_nrow, &segsze, &alpha, &lusup[luptr1],
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dgstrf.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dgstrf.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/csp_blas3.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/csp_blas3.c:17:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cpruneL.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cpruneL.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/creadtriple.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/creadtriple.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadtriple.c: In function ‘creadrhs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadtriple.c:127:5: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
         FILE *fp, *fopen();
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadtriple.c: In function ‘creadtriple’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadtriple.c:38:10: warning: ignoring return value of ‘scanf’, declared with attribute warn_unused_result [-Wunused-result]
         scanf("%d%d", n, nonz);
              ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadtriple.c:54:7: warning: ignoring return value of ‘scanf’, declared with attribute warn_unused_result [-Wunused-result]
      scanf("%d%d%f%f\n", &row[nz], &col[nz], &val[nz].r, &val[nz].i);
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadtriple.c: In function ‘creadrhs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadtriple.c:136:13: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
           fscanf(fp, "%f%f\n", &b[i].r, &b[i].i);
                 ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c:72:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c: In function ‘dreadrb’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c:181:10: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
         fgets(buf, 100, fp);
              ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c:186:15: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
             fscanf(fp, "%14c", buf); buf[14] = 0;
                   ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c:193:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%3c", type);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c:194:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%11c", buf); /* pad */
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c:200:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nrow);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c:201:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", ncol);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c:202:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nonz);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c:203:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", &tmp);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c:215:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c:217:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c:219:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%20c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c: In function ‘ReadVector’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c:130:14: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
             fgets(buf, 100, fp);    /* read a line at a time */
                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c: In function ‘dReadValues’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dreadrb.c:151:14: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
             fgets(buf, 100, fp);    /* read a line at a time */
                  ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_cpanel_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_cpanel_dfs.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/csnode_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/csnode_dfs.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/csnode_dfs.c: In function ‘csnode_dfs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/csnode_dfs.c:81:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           if ( mem_error = cLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu) )
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/csnode_dfs.c:94:6: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
          if ( mem_error = cLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu) )
          ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ssp_blas3.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ssp_blas3.c:17:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/colamd.c
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zsp_blas3.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zsp_blas3.c:17:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sp_preorder.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sp_preorder.c:4:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_cdrop_row.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_cdrop_row.c:14:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_cdrop_row.c: In function ‘ilu_cdrop_row’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_cdrop_row.c:186:3: warning: implicit declaration of function ‘scopy_’ [-Wimplicit-function-declaration]
       scopy_(&len, swork, &i_1, swork2, &i_1);
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_cdrop_row.c:75:13: warning: unused variable ‘zero’ [-Wunused-variable]
         complex zero = {0.0, 0.0};
                 ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_cdrop_row.c: At top level:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_cdrop_row.c:25:12: warning: ‘_compare_’ defined but not used [-Wunused-function]
     static int _compare_(const void *a, const void *b)
                ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zlaqgs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zlaqgs.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/scopy_to_ucol.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/scopy_to_ucol.c:23:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/scopy_to_ucol.c: In function ‘scopy_to_ucol’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/scopy_to_ucol.c:77:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           if (mem_error = sLUMemXpand(jcol, nextu, UCOL, &nzumax, Glu))
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/scopy_to_ucol.c:80:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           if (mem_error = sLUMemXpand(jcol, nextu, USUB, &nzumax, Glu))
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sgssvx.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sgssvx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgssvx.c: In function ‘sgssvx’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgssvx.c:356:14: warning: variable ‘diag_pivot_thresh’ set but not used [-Wunused-but-set-variable]
         float    diag_pivot_thresh;
                  ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sgssvx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:69:46: warning: ‘smlnum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MAX(x, y)  ( (x) > (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgssvx.c:354:35: note: ‘smlnum’ was declared here
         float    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                                       ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sgssvx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:70:46: warning: ‘bignum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MIN(x, y)  ( (x) < (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgssvx.c:354:27: note: ‘bignum’ was declared here
         float    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                               ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/spanel_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/spanel_dfs.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/spivotL.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/spivotL.c:27:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zcolumn_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zcolumn_dfs.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c:72:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c: In function ‘sreadrb’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c:181:10: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
         fgets(buf, 100, fp);
              ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c:186:15: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
             fscanf(fp, "%14c", buf); buf[14] = 0;
                   ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c:193:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%3c", type);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c:194:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%11c", buf); /* pad */
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c:200:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nrow);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c:201:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", ncol);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c:202:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nonz);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c:203:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", &tmp);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c:215:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c:217:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c:219:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%20c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c: In function ‘ReadVector’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c:130:14: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
             fgets(buf, 100, fp);    /* read a line at a time */
                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c: In function ‘sReadValues’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadrb.c:151:14: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
             fgets(buf, 100, fp);    /* read a line at a time */
                  ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zpivotL.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zpivotL.c:15:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_spanel_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_spanel_dfs.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dsp_blas2.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dsp_blas2.c:17:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dsp_blas2.c: In function ‘sp_dtrsv’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dsp_blas2.c:155:7: warning: implicit declaration of function ‘dtrsv_’ [-Wimplicit-function-declaration]
           dtrsv_("L", "N", "U", &nsupc, &Lval[luptr], &nsupr,
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dsp_blas2.c:158:7: warning: implicit declaration of function ‘dgemv_’ [-Wimplicit-function-declaration]
           dgemv_("N", &nrow, &nsupc, &alpha, &Lval[luptr+nsupc],
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zsnode_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_zsnode_dfs.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:79:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c: In function ‘creadhb’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:191:29: warning: unused variable ‘key’ [-Wunused-variable]
         char buf[100], type[4], key[10];
                                 ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c: In function ‘cReadValues’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:159:7: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
      fgets(buf, 100, fp);    /* read a line at a time */
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c: In function ‘creadhb’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:197:10: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
         fgets(buf, 100, fp);
              ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:209:8: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
      fscanf(fp, "%14c", buf); buf[14] = 0;
            ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:217:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%3c", type);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:218:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%11c", buf); /* pad */
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:224:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nrow);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:225:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", ncol);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:226:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nonz);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:227:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", &tmp);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:239:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:241:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:243:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%20c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:245:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%20c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c: In function ‘ReadVector’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadhb.c:137:7: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
      fgets(buf, 100, fp);    /* read a line at a time */
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ssnode_bmod.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ssnode_bmod.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ssnode_bmod.c: In function ‘ssnode_bmod’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ssnode_bmod.c:97:2: warning: implicit declaration of function ‘strsv_’ [-Wimplicit-function-declaration]
      strsv_( "L", "N", "U", &nsupc, &lusup[luptr], &nsupr,
      ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ssnode_bmod.c:99:2: warning: implicit declaration of function ‘sgemv_’ [-Wimplicit-function-declaration]
      sgemv_( "N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr,
      ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ssnode_bmod.c:52:35: warning: unused variable ‘iptr’ [-Wunused-variable]
         int            isub, irow, i, iptr;
                                       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ssnode_bmod.c:52:32: warning: unused variable ‘i’ [-Wunused-variable]
         int            isub, irow, i, iptr;
                                    ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_scopy_to_ucol.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_scopy_to_ucol.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_scopy_to_ucol.c: In function ‘ilu_scopy_to_ucol’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_scopy_to_ucol.c:66:5: warning: implicit declaration of function ‘dlamch_’ [-Wimplicit-function-declaration]
         register float d_max = 0.0, d_min = 1.0 / dlamch_("Safe minimum");
         ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cgstrf.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cgstrf.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cgscon.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cgscon.c:20:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dgstrs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dgstrs.c:24:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgstrs.c: In function ‘dgstrs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgstrs.c:112:5: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
         void dprint_soln();
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgstrs.c:191:3: warning: implicit declaration of function ‘dtrsm_’ [-Wimplicit-function-declaration]
       dtrsm_("L", "L", "N", "U", &nsupc, &nrhs, &alpha,
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgstrs.c:194:3: warning: implicit declaration of function ‘dgemm_’ [-Wimplicit-function-declaration]
       dgemm_( "N", "N", &nrow, &nrhs, &nsupc, &alpha,
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgstrs.c:98:24: warning: unused variable ‘incy’ [-Wunused-variable]
         int      incx = 1, incy = 1;
                            ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgstrs.c:98:14: warning: unused variable ‘incx’ [-Wunused-variable]
         int      incx = 1, incy = 1;
                  ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_cpivotL.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_cpivotL.c:15:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zgscon.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zgscon.c:20:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/scomplex.c
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/scsum1.c
    scipy/sparse/linalg/dsolve/SuperLU/SRC/scsum1.c: In function ‘scsum1_’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/scsum1.c:46:15: warning: variable ‘i__2’ set but not used [-Wunused-but-set-variable]
         int i__1, i__2;
                   ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/scsum1.c:46:9: warning: variable ‘i__1’ set but not used [-Wunused-but-set-variable]
         int i__1, i__2;
             ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_ccolumn_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_ccolumn_dfs.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/slaqgs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/slaqgs.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dlaqgs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dlaqgs.c:19:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/smemory.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/smemory.c:11:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/smemory.c: In function ‘sLUMemXpand’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/smemory.c:444:5: warning: enumeration value ‘LLVL’ not handled in switch [-Wswitch]
         switch ( mem_type ) {
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/smemory.c:444:5: warning: enumeration value ‘ULVL’ not handled in switch [-Wswitch]
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_dpanel_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_dpanel_dfs.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cgssvx.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cgssvx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgssvx.c: In function ‘cgssvx’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgssvx.c:356:14: warning: variable ‘diag_pivot_thresh’ set but not used [-Wunused-but-set-variable]
         float    diag_pivot_thresh;
                  ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cgssvx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:69:46: warning: ‘smlnum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MAX(x, y)  ( (x) > (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgssvx.c:354:35: note: ‘smlnum’ was declared here
         float    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                                       ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cgssvx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:70:46: warning: ‘bignum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MIN(x, y)  ( (x) < (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgssvx.c:354:27: note: ‘bignum’ was declared here
         float    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                               ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zgssvx.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zgssvx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgssvx.c: In function ‘zgssvx’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgssvx.c:356:15: warning: variable ‘diag_pivot_thresh’ set but not used [-Wunused-but-set-variable]
         double    diag_pivot_thresh;
                   ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zgssvx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:69:46: warning: ‘smlnum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MAX(x, y)  ( (x) > (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgssvx.c:354:36: note: ‘smlnum’ was declared here
         double    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                                        ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zgssvx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:70:46: warning: ‘bignum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MIN(x, y)  ( (x) < (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgssvx.c:354:28: note: ‘bignum’ was declared here
         double    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                                ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cutil.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cutil.c:26:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cutil.c: In function ‘cFillRHS’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cutil.c:360:16: warning: variable ‘Aval’ set but not used [-Wunused-but-set-variable]
         complex   *Aval;
                    ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cutil.c: At top level:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cutil.c:468:1: warning: return type defaults to ‘int’ [-Wreturn-type]
     print_complex_vec(char *what, int n, complex *vec)
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadtriple.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadtriple.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadtriple.c: In function ‘sreadrhs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadtriple.c:127:5: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
         FILE *fp, *fopen();
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadtriple.c: In function ‘sreadtriple’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadtriple.c:38:10: warning: ignoring return value of ‘scanf’, declared with attribute warn_unused_result [-Wunused-result]
         scanf("%d%d", n, nonz);
              ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadtriple.c:54:7: warning: ignoring return value of ‘scanf’, declared with attribute warn_unused_result [-Wunused-result]
      scanf("%d%d%f\n", &row[nz], &col[nz], &val[nz]);
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadtriple.c: In function ‘sreadrhs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadtriple.c:136:13: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
           fscanf(fp, "%f\n", &b[i]);
                 ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zpivotgrowth.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zpivotgrowth.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zpivotgrowth.c: In function ‘zPivotGrowth’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zpivotgrowth.c:63:21: warning: unused variable ‘temp_comp’ [-Wunused-variable]
         doublecomplex   temp_comp;
                         ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_spivotL.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_spivotL.c:15:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dlacon.c
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zdiagonal.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zdiagonal.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zpruneL.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zpruneL.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zgstrs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zgstrs.c:24:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgstrs.c: In function ‘zgstrs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgstrs.c:113:5: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
         void zprint_soln();
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgstrs.c:193:3: warning: implicit declaration of function ‘ztrsm_’ [-Wimplicit-function-declaration]
       ztrsm_("L", "L", "N", "U", &nsupc, &nrhs, &alpha,
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgstrs.c:196:3: warning: implicit declaration of function ‘zgemm_’ [-Wimplicit-function-declaration]
       zgemm_( "N", "N", &nrow, &nrhs, &nsupc, &alpha,
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgstrs.c:98:24: warning: unused variable ‘incy’ [-Wunused-variable]
         int      incx = 1, incy = 1;
                            ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgstrs.c:98:14: warning: unused variable ‘incx’ [-Wunused-variable]
         int      incx = 1, incy = 1;
                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgstrs.c: In function ‘zprint_soln’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgstrs.c:349:4: warning: format ‘%f’ expects argument of type ‘double’, but argument 3 has type ‘doublecomplex’ [-Wformat=]
        printf("\t%d: %.4f\n", i, soln[i]);
        ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:79:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c: In function ‘sreadhb’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:181:29: warning: unused variable ‘key’ [-Wunused-variable]
         char buf[100], type[4], key[10];
                                 ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c: In function ‘sReadValues’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:157:7: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
      fgets(buf, 100, fp);    /* read a line at a time */
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c: In function ‘sreadhb’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:187:10: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
         fgets(buf, 100, fp);
              ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:199:8: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
      fscanf(fp, "%14c", buf); buf[14] = 0;
            ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:207:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%3c", type);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:208:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%11c", buf); /* pad */
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:214:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nrow);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:215:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", ncol);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:216:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nonz);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:217:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", &tmp);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:229:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:231:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:233:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%20c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:235:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%20c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c: In function ‘ReadVector’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sreadhb.c:137:7: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
      fgets(buf, 100, fp);    /* read a line at a time */
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dpanel_bmod.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dpanel_bmod.c:29:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dpanel_bmod.c:36:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern void dcheck_tempv();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dpanel_bmod.c: In function ‘dpanel_bmod’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dpanel_bmod.c:224:7: warning: implicit declaration of function ‘dtrsv_’ [-Wimplicit-function-declaration]
           dtrsv_( "L", "N", "U", &segsze, &lusup[luptr],
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dpanel_bmod.c:271:7: warning: implicit declaration of function ‘dgemv_’ [-Wimplicit-function-declaration]
           dgemv_("N", &block_nrow, &segsze, &alpha, &lusup[luptr1],
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadtriple.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadtriple.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadtriple.c: In function ‘zreadrhs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadtriple.c:127:5: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
         FILE *fp, *fopen();
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadtriple.c: In function ‘zreadtriple’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadtriple.c:38:10: warning: ignoring return value of ‘scanf’, declared with attribute warn_unused_result [-Wunused-result]
         scanf("%d%d", n, nonz);
              ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadtriple.c:54:7: warning: ignoring return value of ‘scanf’, declared with attribute warn_unused_result [-Wunused-result]
      scanf("%d%d%lf%lf\n", &row[nz], &col[nz], &val[nz].r, &val[nz].i);
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadtriple.c: In function ‘zreadrhs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zreadtriple.c:136:13: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
           fscanf(fp, "%lf%lf\n", &b[i].r, &b[i].i);
                 ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/cgstrs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/cgstrs.c:24:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgstrs.c: In function ‘cgstrs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgstrs.c:113:5: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
         void cprint_soln();
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgstrs.c:193:3: warning: implicit declaration of function ‘ctrsm_’ [-Wimplicit-function-declaration]
       ctrsm_("L", "L", "N", "U", &nsupc, &nrhs, &alpha,
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgstrs.c:196:3: warning: implicit declaration of function ‘cgemm_’ [-Wimplicit-function-declaration]
       cgemm_( "N", "N", &nrow, &nrhs, &nsupc, &alpha,
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgstrs.c:98:24: warning: unused variable ‘incy’ [-Wunused-variable]
         int      incx = 1, incy = 1;
                            ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgstrs.c:98:14: warning: unused variable ‘incx’ [-Wunused-variable]
         int      incx = 1, incy = 1;
                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgstrs.c: In function ‘cprint_soln’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/cgstrs.c:349:4: warning: format ‘%f’ expects argument of type ‘double’, but argument 3 has type ‘complex’ [-Wformat=]
        printf("\t%d: %.4f\n", i, soln[i]);
        ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dsnode_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dsnode_dfs.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dsnode_dfs.c: In function ‘dsnode_dfs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dsnode_dfs.c:81:7: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
           if ( mem_error = dLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu) )
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dsnode_dfs.c:94:6: warning: suggest parentheses around assignment used as truth value [-Wparentheses]
          if ( mem_error = dLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu) )
          ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sdiagonal.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sdiagonal.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/icmax1.c
    scipy/sparse/linalg/dsolve/SuperLU/SRC/icmax1.c: In function ‘icmax1_’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/icmax1.c:53:24: warning: variable ‘i__2’ set but not used [-Wunused-but-set-variable]
         int ret_val, i__1, i__2;
                            ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/icmax1.c:53:18: warning: variable ‘i__1’ set but not used [-Wunused-but-set-variable]
         int ret_val, i__1, i__2;
                      ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sutil.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sutil.c:26:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sutil.c: In function ‘sFillRHS’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sutil.c:358:14: warning: variable ‘Aval’ set but not used [-Wunused-but-set-variable]
         float   *Aval;
                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sutil.c: At top level:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sutil.c:464:1: warning: return type defaults to ‘int’ [-Wreturn-type]
     print_float_vec(char *what, int n, float *vec)
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsisx.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsisx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsisx.c: In function ‘zgsisx’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsisx.c:578:7: warning: suggest parentheses around operand of ‘!’ or change ‘&’ to ‘&&’ or ‘!’ to ‘~’ [-Wparentheses]
      if ( !mc64 & equil ) { /* Only perform equilibration, no row perm */
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsisx.c:667:24: warning: unused variable ‘rhs_work’ [-Wunused-variable]
             doublecomplex *rhs_work;
                            ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsisx.c:413:15: warning: variable ‘diag_pivot_thresh’ set but not used [-Wunused-but-set-variable]
         double    diag_pivot_thresh;
                   ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsisx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:69:46: warning: ‘smlnum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MAX(x, y)  ( (x) > (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsisx.c:411:36: note: ‘smlnum’ was declared here
         double    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                                        ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsisx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:70:46: warning: ‘bignum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MIN(x, y)  ( (x) < (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgsisx.c:411:28: note: ‘bignum’ was declared here
         double    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                                ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/spruneL.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/spruneL.c:25:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dpivotgrowth.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dpivotgrowth.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_ssnode_dfs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_ssnode_dfs.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/sgstrs.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_sdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/sgstrs.c:24:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgstrs.c: In function ‘sgstrs’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgstrs.c:112:5: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
         void sprint_soln();
         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgstrs.c:191:3: warning: implicit declaration of function ‘strsm_’ [-Wimplicit-function-declaration]
       strsm_("L", "L", "N", "U", &nsupc, &nrhs, &alpha,
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgstrs.c:194:3: warning: implicit declaration of function ‘sgemm_’ [-Wimplicit-function-declaration]
       sgemm_( "N", "N", &nrow, &nrhs, &nsupc, &alpha,
       ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgstrs.c:98:24: warning: unused variable ‘incy’ [-Wunused-variable]
         int      incx = 1, incy = 1;
                            ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/sgstrs.c:98:14: warning: unused variable ‘incx’ [-Wunused-variable]
         int      incx = 1, incy = 1;
                  ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/zgssv.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/zgssv.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgssv.c: In function ‘zgssv’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:61:37: warning: ‘AA’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_FREE(addr) USER_FREE(addr)
                                         ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/zgssv.c:139:18: note: ‘AA’ was declared here
         SuperMatrix *AA;/* A in SLU_NC format used by the factorization routine.*/
                      ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dsp_blas3.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dsp_blas3.c:17:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsisx.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsisx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsisx.c: In function ‘dgsisx’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsisx.c:578:7: warning: suggest parentheses around operand of ‘!’ or change ‘&’ to ‘&&’ or ‘!’ to ‘~’ [-Wparentheses]
      if ( !mc64 & equil ) { /* Only perform equilibration, no row perm */
           ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsisx.c:667:17: warning: unused variable ‘rhs_work’ [-Wunused-variable]
             double *rhs_work;
                     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsisx.c:413:15: warning: variable ‘diag_pivot_thresh’ set but not used [-Wunused-but-set-variable]
         double    diag_pivot_thresh;
                   ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsisx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:69:46: warning: ‘smlnum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MAX(x, y)  ( (x) > (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsisx.c:411:36: note: ‘smlnum’ was declared here
         double    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                                        ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsisx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:70:46: warning: ‘bignum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MIN(x, y)  ( (x) < (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgsisx.c:411:28: note: ‘bignum’ was declared here
         double    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                                ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/dgssvx.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dgssvx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgssvx.c: In function ‘dgssvx’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgssvx.c:356:15: warning: variable ‘diag_pivot_thresh’ set but not used [-Wunused-but-set-variable]
         double    diag_pivot_thresh;
                   ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dgssvx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:69:46: warning: ‘smlnum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MAX(x, y)  ( (x) > (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgssvx.c:354:36: note: ‘smlnum’ was declared here
         double    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                                        ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/dgssvx.c:12:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:70:46: warning: ‘bignum’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define SUPERLU_MIN(x, y)  ( (x) < (y) ? (x) : (y) )
                                                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/dgssvx.c:354:28: note: ‘bignum’ was declared here
         double    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
                                ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_ddrop_row.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_ddefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_ddrop_row.c:14:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/ilu_ddrop_row.c:25:12: warning: ‘_compare_’ defined but not used [-Wunused-function]
     static int _compare_(const void *a, const void *b)
                ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_cdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c:72:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c: In function ‘creadrb’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c:190:10: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
         fgets(buf, 100, fp);
              ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c:195:15: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
             fscanf(fp, "%14c", buf); buf[14] = 0;
                   ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c:202:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%3c", type);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c:203:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%11c", buf); /* pad */
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c:209:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nrow);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c:210:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", ncol);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c:211:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", nonz);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c:212:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%14c", buf); sscanf(buf, "%d", &tmp);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c:224:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c:226:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%16c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c:228:11: warning: ignoring return value of ‘fscanf’, declared with attribute warn_unused_result [-Wunused-result]
         fscanf(fp, "%20c", buf);
               ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c: In function ‘ReadVector’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c:130:14: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
             fgets(buf, 100, fp);    /* read a line at a time */
                  ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c: In function ‘cReadValues’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/creadrb.c:152:7: warning: ignoring return value of ‘fgets’, declared with attribute warn_unused_result [-Wunused-result]
      fgets(buf, 100, fp);    /* read a line at a time */
           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/SuperLU/SRC/lsame.c
    scipy/sparse/linalg/dsolve/SuperLU/SRC/lsame.c: In function ‘lsame_’:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/lsame.c:68:18: warning: suggest parentheses around ‘&&’ within ‘||’ [-Wparentheses]
      if (inta >= 129 && inta <= 137 || inta >= 145 && inta <= 153 || inta
                      ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/lsame.c:69:10: warning: suggest parentheses around ‘&&’ within ‘||’ [-Wparentheses]
       >= 162 && inta <= 169)
              ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/lsame.c:71:18: warning: suggest parentheses around ‘&&’ within ‘||’ [-Wparentheses]
      if (intb >= 129 && intb <= 137 || intb >= 145 && intb <= 153 || intb
                      ^
    scipy/sparse/linalg/dsolve/SuperLU/SRC/lsame.c:72:10: warning: suggest parentheses around ‘&&’ within ‘||’ [-Wparentheses]
       >= 162 && intb <= 169)
              ^
    ar: adding 50 object files to build/temp.linux-x86_64-2.7/libsuperlu_src.a
    ar: adding 50 object files to build/temp.linux-x86_64-2.7/libsuperlu_src.a
    ar: adding 50 object files to build/temp.linux-x86_64-2.7/libsuperlu_src.a
    ar: adding 23 object files to build/temp.linux-x86_64-2.7/libsuperlu_src.a
    building 'arpack_scipy' library
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-Iscipy/sparse/linalg/eigen/arpack/ARPACK/SRC -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sgetv0.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sgetv0.f:120.26:
    
         &   ( ido, bmat, itry, initv, n, j, v, ldv, resid, rnorm,
                              1
    Warning: Unused dummy argument 'itry' at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sgetv0.f:128:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sgetv0.f:128:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsortr.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsortc.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaup2.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaup2.f:322.5:
    
       10 continue
         1
    Warning: Label 10 at (1) defined but not used
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaup2.f:169.63:
    
         &   ( ido, bmat, n, which, nev, np, tol, resid, mode, iupd,
                                                                   1
    Warning: Unused dummy argument 'iupd' at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaup2.f:178:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaup2.f:178:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zgetv0.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zgetv0.f:116.26:
    
         &   ( ido, bmat, itry, initv, n, j, v, ldv, resid, rnorm,
                              1
    Warning: Unused dummy argument 'itry' at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zgetv0.f:124:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zgetv0.f:124:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssconv.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssconv.f:66:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssconv.f:66:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssconv.f:66:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssconv.f:66:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstatn.f
    stat.h:8.19:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstatn.f:24:
    
          real       t0, t1, t2, t3, t4, t5
                       1
    Warning: Unused variable 't0' declared at (1)
    stat.h:8.23:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstatn.f:24:
    
          real       t0, t1, t2, t3, t4, t5
                           1
    Warning: Unused variable 't1' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstatn.f:24:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstatn.f:24:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstatn.f:24:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstatn.f:24:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaupe.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstats.f
    stat.h:8.19:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstats.f:14:
    
          real       t0, t1, t2, t3, t4, t5
                       1
    Warning: Unused variable 't0' declared at (1)
    stat.h:8.23:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstats.f:14:
    
          real       t0, t1, t2, t3, t4, t5
                           1
    Warning: Unused variable 't1' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstats.f:14:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstats.f:14:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstats.f:14:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstats.f:14:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssgets.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssgets.f:100:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssgets.f:100:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssgets.f:100:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssgets.f:100:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snapps.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snapps.f:152:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snapps.f:152:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snapps.f:152:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snapps.f:152:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dgetv0.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dgetv0.f:120.26:
    
         &   ( ido, bmat, itry, initv, n, j, v, ldv, resid, rnorm,
                              1
    Warning: Unused dummy argument 'itry' at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dgetv0.f:128:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dgetv0.f:128:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaup2.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaup2.f:316.5:
    
       10 continue
         1
    Warning: Label 10 at (1) defined but not used
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaup2.f:175.63:
    
         &   ( ido, bmat, n, which, nev, np, tol, resid, mode, iupd,
                                                                   1
    Warning: Unused dummy argument 'iupd' at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaup2.f:184:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaup2.f:184:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znapps.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znapps.f:143:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znapps.f:143:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znapps.f:143:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znapps.f:143:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sneupd.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sneupd.f:587.17:
    
                jj = workl(bounds + ncv - j)
                     1
    Warning: Possible change of value in conversion from REAL(4) to INTEGER(4) at (1)
    stat.h:8.19:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sneupd.f:313:
    
          real       t0, t1, t2, t3, t4, t5
                       1
    Warning: Unused variable 't0' declared at (1)
    stat.h:8.23:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sneupd.f:313:
    
          real       t0, t1, t2, t3, t4, t5
                           1
    Warning: Unused variable 't1' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sneupd.f:313:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sneupd.f:313:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sneupd.f:313:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sneupd.f:313:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/slaqrb.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/slaqrb.f: In function ‘slaqrb’:
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/slaqrb.f:427:0: warning: ‘i2’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                    do 90 j = k, i2
     ^
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstats.f
    stat.h:8.19:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstats.f:14:
    
          real       t0, t1, t2, t3, t4, t5
                       1
    Warning: Unused variable 't0' declared at (1)
    stat.h:8.23:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstats.f:14:
    
          real       t0, t1, t2, t3, t4, t5
                           1
    Warning: Unused variable 't1' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstats.f:14:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstats.f:14:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstats.f:14:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstats.f:14:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaupd.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaupd.f:514.24:
    
             if (nb .le. 0)    nb = 1
                            1
    Warning: Nonconforming tab character at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaupd.f:515.32:
    
             if (tol .le. 0.0D+0  )   tol = dlamch ('EpsMach')
                                    1
    Warning: Nonconforming tab character at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaupd.f:518.29:
    
         &       ishift .ne. 2)    ishift = 1
                                 1
    Warning: Nonconforming tab character at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaupd.f:422.32:
    
         &           ldh, ldq, levec, mode, msglvl, mxiter, nb,
                                    1
    Warning: Unused variable 'levec' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaupd.f:388:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaupd.f:388:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaupd.f:388:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaupd.f:388:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaupd.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaupd.f:541.24:
    
             if (nb .le. 0)    nb = 1
                            1
    Warning: Nonconforming tab character at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaupd.f:542.28:
    
             if (tol .le. zero)   tol = dlamch ('EpsMach')
                                1
    Warning: Nonconforming tab character at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaupd.f:447.32:
    
         &           ldh, ldq, levec, mode, msglvl, mxiter, nb,
                                    1
    Warning: Unused variable 'levec' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaupd.f:415:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaupd.f:415:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaupd.f:415:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaupd.f:415:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaupd.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaupd.f:514.24:
    
             if (nb .le. 0)    nb = 1
                            1
    Warning: Nonconforming tab character at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaupd.f:515.32:
    
             if (tol .le. 0.0E+0  )   tol = wslamch('EpsMach')
                                    1
    Warning: Nonconforming tab character at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaupd.f:518.29:
    
         &       ishift .ne. 2)    ishift = 1
                                 1
    Warning: Nonconforming tab character at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaupd.f:422.32:
    
         &           ldh, ldq, levec, mode, msglvl, mxiter, nb,
                                    1
    Warning: Unused variable 'levec' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaupd.f:388:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaupd.f:388:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaupd.f:388:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaupd.f:388:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssesrt.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssaupd.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssaupd.f:417:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssaupd.f:417:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssaupd.f:417:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssaupd.f:417:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dngets.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dngets.f:96.40:
    
         &                    shiftr, shifti )
                                            1
    Warning: Unused dummy argument 'shifti' at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dngets.f:96.32:
    
         &                    shiftr, shifti )
                                    1
    Warning: Unused dummy argument 'shiftr' at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dngets.f:103:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dngets.f:103:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dngets.f:103:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dngets.f:103:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssaup2.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssaup2.f:809.5:
    
      130    continue
         1
    Warning: Label 130 at (1) defined but not used
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssaup2.f:324.5:
    
       10 continue
         1
    Warning: Label 10 at (1) defined but not used
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssaup2.f:180.63:
    
         &   ( ido, bmat, n, which, nev, np, tol, resid, mode, iupd,
                                                                   1
    Warning: Unused dummy argument 'iupd' at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssaup2.f:189:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssaup2.f:189:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsconv.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsconv.f:66:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsconv.f:66:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsconv.f:66:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsconv.f:66:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sngets.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sngets.f:96.40:
    
         &                    shiftr, shifti )
                                            1
    Warning: Unused dummy argument 'shifti' at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sngets.f:96.32:
    
         &                    shiftr, shifti )
                                    1
    Warning: Unused dummy argument 'shiftr' at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sngets.f:103:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sngets.f:103:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sngets.f:103:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sngets.f:103:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cngets.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cngets.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cngets.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cngets.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cngets.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dneigh.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnconv.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnconv.f:73:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnconv.f:73:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnconv.f:73:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnconv.f:73:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cneigh.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsaupd.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsaupd.f:417:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsaupd.f:417:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsaupd.f:417:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsaupd.f:417:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zneupd.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zneupd.f:520.17:
    
                jj = workl(bounds + ncv - j)
                     1
    Warning: Possible change of value in conversion from COMPLEX(8) to INTEGER(4) at (1)
    stat.h:8.19:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zneupd.f:260:
    
          real       t0, t1, t2, t3, t4, t5
                       1
    Warning: Unused variable 't0' declared at (1)
    stat.h:8.23:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zneupd.f:260:
    
          real       t0, t1, t2, t3, t4, t5
                           1
    Warning: Unused variable 't1' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zneupd.f:260:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zneupd.f:260:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zneupd.f:260:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zneupd.f:260:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaupd.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaupd.f:541.24:
    
             if (nb .le. 0)    nb = 1
                            1
    Warning: Nonconforming tab character at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaupd.f:542.28:
    
             if (tol .le. zero)   tol = wslamch('EpsMach')
                                1
    Warning: Nonconforming tab character at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaupd.f:447.32:
    
         &           ldh, ldq, levec, mode, msglvl, mxiter, nb,
                                    1
    Warning: Unused variable 'levec' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaupd.f:415:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaupd.f:415:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaupd.f:415:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaupd.f:415:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssaitr.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsaitr.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sseupd.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sseupd.f:499.17:
    
                jj = workl(bounds + ncv - j)
                     1
    Warning: Possible change of value in conversion from REAL(4) to INTEGER(4) at (1)
    stat.h:8.19:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sseupd.f:230:
    
          real       t0, t1, t2, t3, t4, t5
                       1
    Warning: Unused variable 't0' declared at (1)
    stat.h:8.23:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sseupd.f:230:
    
          real       t0, t1, t2, t3, t4, t5
                           1
    Warning: Unused variable 't1' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sseupd.f:230:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sseupd.f:230:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sseupd.f:230:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sseupd.f:230:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssortr.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snconv.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snconv.f:73:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snconv.f:73:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snconv.f:73:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snconv.f:73:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zsortc.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaupe.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sneigh.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dseigt.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dseigt.f:124.18:
    
          integer    i, k, msglvl
                      1
    Warning: Unused variable 'i' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dseigt.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dseigt.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dseigt.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dseigt.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaup2.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaup2.f:316.5:
    
       10 continue
         1
    Warning: Label 10 at (1) defined but not used
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaup2.f:175.63:
    
         &   ( ido, bmat, n, which, nev, np, tol, resid, mode, iupd,
                                                                   1
    Warning: Unused dummy argument 'iupd' at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaup2.f:184:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaup2.f:184:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssortc.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaitr.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/znaitr.f:209.33:
    
         &   (ido, bmat, n, k, np, nb, resid, rnorm, v, ldv, h, ldh,
                                     1
    Warning: Unused dummy argument 'nb' at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsapps.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsapps.f:139:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsapps.f:139:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsapps.f:139:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsapps.f:139:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnapps.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnapps.f:143:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnapps.f:143:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnapps.f:143:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnapps.f:143:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstatn.f
    stat.h:8.19:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstatn.f:24:
    
          real       t0, t1, t2, t3, t4, t5
                       1
    Warning: Unused variable 't0' declared at (1)
    stat.h:8.23:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstatn.f:24:
    
          real       t0, t1, t2, t3, t4, t5
                           1
    Warning: Unused variable 't1' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstatn.f:24:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstatn.f:24:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstatn.f:24:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstatn.f:24:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sstqrb.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsaup2.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsaup2.f:809.5:
    
      130    continue
         1
    Warning: Label 130 at (1) defined but not used
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsaup2.f:324.5:
    
       10 continue
         1
    Warning: Label 10 at (1) defined but not used
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsaup2.f:180.63:
    
         &   ( ido, bmat, n, which, nev, np, tol, resid, mode, iupd,
                                                                   1
    Warning: Unused dummy argument 'iupd' at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsaup2.f:189:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsaup2.f:189:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cstatn.f
    stat.h:8.19:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cstatn.f:16:
    
          real       t0, t1, t2, t3, t4, t5
                       1
    Warning: Unused variable 't0' declared at (1)
    stat.h:8.23:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cstatn.f:16:
    
          real       t0, t1, t2, t3, t4, t5
                           1
    Warning: Unused variable 't1' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cstatn.f:16:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cstatn.f:16:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cstatn.f:16:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cstatn.f:16:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsesrt.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/csortc.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssapps.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssapps.f:139:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssapps.f:139:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssapps.f:139:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/ssapps.f:139:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dseupd.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dseupd.f:499.17:
    
                jj = workl(bounds + ncv - j)
                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    stat.h:8.19:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dseupd.f:230:
    
          real       t0, t1, t2, t3, t4, t5
                       1
    Warning: Unused variable 't0' declared at (1)
    stat.h:8.23:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dseupd.f:230:
    
          real       t0, t1, t2, t3, t4, t5
                           1
    Warning: Unused variable 't1' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dseupd.f:230:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dseupd.f:230:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dseupd.f:230:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dseupd.f:230:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dstqrb.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dneupd.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dneupd.f:587.17:
    
                jj = workl(bounds + ncv - j)
                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    stat.h:8.19:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dneupd.f:313:
    
          real       t0, t1, t2, t3, t4, t5
                       1
    Warning: Unused variable 't0' declared at (1)
    stat.h:8.23:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dneupd.f:313:
    
          real       t0, t1, t2, t3, t4, t5
                           1
    Warning: Unused variable 't1' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dneupd.f:313:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dneupd.f:313:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dneupd.f:313:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dneupd.f:313:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaitr.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnaitr.f:210.33:
    
         &   (ido, bmat, n, k, np, nb, resid, rnorm, v, ldv, h, ldh,
                                     1
    Warning: Unused dummy argument 'nb' at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zneigh.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zneigh.f:108:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaitr.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/snaitr.f:210.33:
    
         &   (ido, bmat, n, k, np, nb, resid, rnorm, v, ldv, h, ldh,
                                     1
    Warning: Unused dummy argument 'nb' at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsgets.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsgets.f:100:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsgets.f:100:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsgets.f:100:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dsgets.f:100:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaup2.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaup2.f:322.5:
    
       10 continue
         1
    Warning: Label 10 at (1) defined but not used
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaup2.f:169.63:
    
         &   ( ido, bmat, n, which, nev, np, tol, resid, mode, iupd,
                                                                   1
    Warning: Unused dummy argument 'iupd' at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaup2.f:178:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaup2.f:178:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnapps.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnapps.f:152:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnapps.f:152:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnapps.f:152:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dnapps.f:152:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zngets.f
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zngets.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zngets.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zngets.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zngets.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zstatn.f
    stat.h:8.19:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zstatn.f:16:
    
          real       t0, t1, t2, t3, t4, t5
                       1
    Warning: Unused variable 't0' declared at (1)
    stat.h:8.23:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zstatn.f:16:
    
          real       t0, t1, t2, t3, t4, t5
                           1
    Warning: Unused variable 't1' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zstatn.f:16:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zstatn.f:16:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zstatn.f:16:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/zstatn.f:16:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaitr.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cnaitr.f:209.33:
    
         &   (ido, bmat, n, k, np, nb, resid, rnorm, v, ldv, h, ldh,
                                     1
    Warning: Unused dummy argument 'nb' at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dlaqrb.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dlaqrb.f: In function ‘dlaqrb’:
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/dlaqrb.f:427:0: warning: ‘i2’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                    do 90 j = k, i2
     ^
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cneupd.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cneupd.f:520.17:
    
                jj = workl(bounds + ncv - j)
                     1
    Warning: Possible change of value in conversion from COMPLEX(4) to INTEGER(4) at (1)
    stat.h:8.19:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cneupd.f:260:
    
          real       t0, t1, t2, t3, t4, t5
                       1
    Warning: Unused variable 't0' declared at (1)
    stat.h:8.23:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cneupd.f:260:
    
          real       t0, t1, t2, t3, t4, t5
                           1
    Warning: Unused variable 't1' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cneupd.f:260:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cneupd.f:260:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cneupd.f:260:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cneupd.f:260:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sseigt.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sseigt.f:124.18:
    
          integer    i, k, msglvl
                      1
    Warning: Unused variable 'i' declared at (1)
    stat.h:8.27:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sseigt.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                               1
    Warning: Unused variable 't2' declared at (1)
    stat.h:8.31:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sseigt.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                                   1
    Warning: Unused variable 't3' declared at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sseigt.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/sseigt.f:95:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cgetv0.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cgetv0.f:116.26:
    
         &   ( ido, bmat, itry, initv, n, j, v, ldv, resid, rnorm,
                              1
    Warning: Unused dummy argument 'itry' at (1)
    stat.h:8.35:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cgetv0.f:124:
    
          real       t0, t1, t2, t3, t4, t5
                                       1
    Warning: Unused variable 't4' declared at (1)
    stat.h:8.39:
        Included at scipy/sparse/linalg/eigen/arpack/ARPACK/SRC/cgetv0.f:124:
    
          real       t0, t1, t2, t3, t4, t5
                                           1
    Warning: Unused variable 't5' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/iswap.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/icnteq.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/second_NONE.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/second_NONE.f:17.27:
    
          REAL               T1
                               1
    Warning: Unused variable 't1' declared at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/second_NONE.f:20.36:
    
          REAL               TARRAY( 2 )
                                        1
    Warning: Unused variable 'tarray' declared at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/dmout.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/smout.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/zmout.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/cmout.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/cvout.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/icopy.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/ivout.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/dvout.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/zvout.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/iset.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/iset.f:6.43:
    
          subroutine iset (n, value, array, inc)
                                               1
    Warning: Unused dummy argument 'inc' at (1)
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/UTIL/svout.f
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/slahqr.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/slahqr.f: In function ‘slahqr’:
    scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/slahqr.f:327:0: warning: ‘i2’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                    DO 90 J = K, I2
     ^
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/dlahqr.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/dlahqr.f: In function ‘dlahqr’:
    scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/dlahqr.f:327:0: warning: ‘i2’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                    DO 90 J = K, I2
     ^
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/zlahqr.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/zlahqr.f:239.18:
    
                H21 = H( M+1, M )
                      1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/zlahqr.f:245.18:
    
                H10 = H( M, M-1 )
                      1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/zlahqr.f:253.15:
    
             H21 = H( L+1, L )
                   1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/zlahqr.f: In function ‘zlahqr’:
    scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/zlahqr.f:330:0: warning: ‘i2’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                          IF( I2.GT.J )
     ^
    gfortran:f77: scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/clahqr.f
    scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/clahqr.f:239.18:
    
                H21 = H( M+1, M )
                      1
    Warning: Possible change of value in conversion from COMPLEX(4) to REAL(4) at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/clahqr.f:245.18:
    
                H10 = H( M, M-1 )
                      1
    Warning: Possible change of value in conversion from COMPLEX(4) to REAL(4) at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/clahqr.f:253.15:
    
             H21 = H( L+1, L )
                   1
    Warning: Possible change of value in conversion from COMPLEX(4) to REAL(4) at (1)
    scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/clahqr.f: In function ‘clahqr’:
    scipy/sparse/linalg/eigen/arpack/ARPACK/LAPACK/clahqr.f:330:0: warning: ‘i2’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                          IF( I2.GT.J )
     ^
    gfortran:f77: /tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_g77_abi.f
    gfortran:f77: /tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_accelerate.f
    ar: adding 50 object files to build/temp.linux-x86_64-2.7/libarpack_scipy.a
    ar: adding 40 object files to build/temp.linux-x86_64-2.7/libarpack_scipy.a
    building 'sc_c_misc' library
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/tmp/pip_build_root/scipy/scipy/special -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -I/tmp/pip_build_root/scipy/scipy/special/c_misc -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    x86_64-linux-gnu-gcc: scipy/special/c_misc/gammasgn.c
    x86_64-linux-gnu-gcc: scipy/special/c_misc/gammaincinv.c
    In file included from scipy/special/c_misc/gammaincinv.c:7:0:
    /tmp/pip_build_root/scipy/scipy/special/sf_error.h:26:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     int sf_error_get_print();
     ^
    x86_64-linux-gnu-gcc: scipy/special/c_misc/fsolve.c
    x86_64-linux-gnu-gcc: scipy/special/c_misc/besselpoly.c
    x86_64-linux-gnu-gcc: scipy/special/c_misc/struve.c
    In file included from /tmp/pip_build_root/scipy/scipy/special/amos_wrappers.h:11:0,
                     from scipy/special/c_misc/struve.c:87:
    /tmp/pip_build_root/scipy/scipy/special/sf_error.h:26:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     int sf_error_get_print();
     ^
    scipy/special/c_misc/struve.c: In function ‘struve_power_series’:
    scipy/special/c_misc/struve.c:220:49: warning: unused variable ‘ctmp2’ [-Wunused-variable]
         double2_t cterm, csum, cdiv, z2, c2v, ctmp, ctmp2;
                                                     ^
    scipy/special/c_misc/struve.c: In function ‘struve_bessel_series’:
    scipy/special/c_misc/struve.c:295:12: warning: unused variable ‘sgn’ [-Wunused-variable]
         int n, sgn;
                ^
    ar: adding 5 object files to build/temp.linux-x86_64-2.7/libsc_c_misc.a
    building 'sc_cephes' library
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/tmp/pip_build_root/scipy/scipy/special -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -I/tmp/pip_build_root/scipy/scipy/special/c_misc -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    x86_64-linux-gnu-gcc: scipy/special/cephes/hyp2f1.c
    scipy/special/cephes/hyp2f1.c: In function ‘hys2f1’:
    scipy/special/cephes/hyp2f1.c:476:9: warning: variable ‘ia’ set but not used [-Wunused-but-set-variable]
         int ia, ib, intflag = 0;
             ^
    scipy/special/cephes/hyp2f1.c:474:39: warning: unused variable ‘t’ [-Wunused-variable]
         double f, g, h, k, m, s, u, umax, t;
                                           ^
    scipy/special/cephes/hyp2f1.c: In function ‘hyp2f1ra’:
    scipy/special/cephes/hyp2f1.c:552:12: warning: unused variable ‘m’ [-Wunused-variable]
         int n, m, da;
                ^
    x86_64-linux-gnu-gcc: scipy/special/cephes/igam.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/ellpk.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/stdtr.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/unity.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/exp10.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/powi.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/ndtr.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/struve.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/nbdtr.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/chdtr.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/btdtr.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/fdtr.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/j1.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/zeta.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/ellpe.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/polevl.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/chbevl.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/k1.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/shichi.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/incbet.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/airy.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/ellpj.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/i0.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/euclid.c
    scipy/special/cephes/euclid.c:48:6: warning: ‘radd’ defined but not used [-Wunused-function]
     void radd(f1, f2, f3)
          ^
    scipy/special/cephes/euclid.c:81:6: warning: ‘rsub’ defined but not used [-Wunused-function]
     void rsub(f1, f2, f3)
          ^
    scipy/special/cephes/euclid.c:113:6: warning: ‘rmul’ defined but not used [-Wunused-function]
     void rmul(ff1, ff2, ff3)
          ^
    scipy/special/cephes/euclid.c:144:6: warning: ‘rdiv’ defined but not used [-Wunused-function]
     void rdiv(ff1, ff2, ff3)
          ^
    x86_64-linux-gnu-gcc: scipy/special/cephes/kolmogorov.c
    In file included from scipy/special/cephes/mconf.h:71:0,
                     from scipy/special/cephes/kolmogorov.c:26:
    scipy/special/cephes/cephes_names.h:92:17: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     #define smirnov cephes_smirnov
                     ^
    scipy/special/cephes/kolmogorov.c:30:8: note: in expansion of macro ‘smirnov’
     double smirnov(n, e)
            ^
    scipy/special/cephes/cephes_names.h:94:20: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     #define kolmogorov cephes_kolmogorov
                        ^
    scipy/special/cephes/kolmogorov.c:79:8: note: in expansion of macro ‘kolmogorov’
     double kolmogorov(y)
            ^
    scipy/special/cephes/cephes_names.h:93:18: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     #define smirnovi cephes_smirnovi
                      ^
    scipy/special/cephes/kolmogorov.c:104:8: note: in expansion of macro ‘smirnovi’
     double smirnovi(n, p)
            ^
    scipy/special/cephes/cephes_names.h:95:17: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     #define kolmogi cephes_kolmogi
                     ^
    scipy/special/cephes/kolmogorov.c:147:8: note: in expansion of macro ‘kolmogi’
     double kolmogi(p)
            ^
    x86_64-linux-gnu-gcc: scipy/special/cephes/incbi.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/polrt.c
    scipy/special/cephes/polrt.c: In function ‘polrt’:
    scipy/special/cephes/polrt.c:178:18: warning: ‘xsav.i’ may be used uninitialized in this function [-Wmaybe-uninitialized]
         if (fabs(x.i / x.r) >= 1.0e-5) {
                      ^
    scipy/special/cephes/polrt.c:178:18: warning: ‘xsav.r’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    x86_64-linux-gnu-gcc: scipy/special/cephes/polyn.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/ellik.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/scipy_iv.c
    scipy/special/cephes/scipy_iv.c: In function ‘cephes_iv’:
    scipy/special/cephes/scipy_iv.c:81:15: warning: unused variable ‘vp’ [-Wunused-variable]
         double t, vp, ax, res;
                   ^
    scipy/special/cephes/scipy_iv.c: In function ‘iv_asymptotic’:
    scipy/special/cephes/scipy_iv.c:141:16: warning: unused variable ‘mup’ [-Wunused-variable]
         double mu, mup;
                    ^
    x86_64-linux-gnu-gcc: scipy/special/cephes/mtherr.c
    In file included from scipy/special/cephes/mtherr.c:60:0:
    /tmp/pip_build_root/scipy/scipy/special/sf_error.h:26:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     int sf_error_get_print();
     ^
    scipy/special/cephes/mtherr.c:68:14: warning: ‘ermsg’ defined but not used [-Wunused-variable]
     static char *ermsg[8] = {
                  ^
    x86_64-linux-gnu-gcc: scipy/special/cephes/dawsn.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/tandg.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/pdtr.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/gels.c
    scipy/special/cephes/gels.c: In function ‘gels’:
    scipy/special/cephes/gels.c:93:2: warning: implicit declaration of function ‘fabs’ [-Wimplicit-function-declaration]
      tb = fabs(A[L - 1]);
      ^
    scipy/special/cephes/gels.c:93:7: warning: incompatible implicit declaration of built-in function ‘fabs’ [enabled by default]
      tb = fabs(A[L - 1]);
           ^
    scipy/special/cephes/gels.c:173:11: warning: incompatible implicit declaration of built-in function ‘fabs’ [enabled by default]
          tb = fabs(A[LR - 1]);
               ^
    x86_64-linux-gnu-gcc: scipy/special/cephes/simq.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/fresnl.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/mmmpy.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/kn.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/polmisc.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/hyperg.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/j0.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/cpmul.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/i1.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/bdtr.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/beta.c
    scipy/special/cephes/beta.c: In function ‘lbeta_asymp’:
    scipy/special/cephes/beta.c:236:15: warning: unused variable ‘sum’ [-Wunused-variable]
         double r, sum;
                   ^
    x86_64-linux-gnu-gcc: scipy/special/cephes/ndtri.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/mtransp.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/sici.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/cbrt.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/setprec.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/mvmpy.c
    scipy/special/cephes/mvmpy.c:34:6: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     void mvmpy(r, c, A, V, Y)
          ^
    x86_64-linux-gnu-gcc: scipy/special/cephes/psi.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/gdtr.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/k0.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/igami.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/sincos.c
    In file included from scipy/special/cephes/mconf.h:71:0,
                     from scipy/special/cephes/sincos.c:104:
    scipy/special/cephes/cephes_names.h:78:16: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     #define sincos cephes_sincos
                    ^
    scipy/special/cephes/sincos.c:229:6: note: in expansion of macro ‘sincos’
     void sincos(x, s, c, flg)
          ^
    x86_64-linux-gnu-gcc: scipy/special/cephes/const.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/rgamma.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/ellie.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/jv.c
    scipy/special/cephes/jv.c: In function ‘cephes_jv’:
    scipy/special/cephes/jv.c:179:8: warning: label ‘underf’ defined but not used [-Wunused-label]
            underf:
            ^
    x86_64-linux-gnu-gcc: scipy/special/cephes/yn.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/sindg.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/exp2.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/simpsn.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/tukey.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/spence.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/round.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/zetac.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/gamma.c
    x86_64-linux-gnu-gcc: scipy/special/cephes/expn.c
    ar: adding 50 object files to build/temp.linux-x86_64-2.7/libsc_cephes.a
    ar: adding 22 object files to build/temp.linux-x86_64-2.7/libsc_cephes.a
    building 'sc_mach' library
    using additional config_fc from setup script for fortran compiler: {'noopt': ('scipy/special/setup.py', 1)}
    customize Gnu95FCompiler
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/special/mach/d1mach.f
    gfortran:f77: scipy/special/mach/r1mach.f
    scipy/special/mach/r1mach.f:167.27:
    
                   CALL I1MCRA(SMALL, K, 16, 0, 0)
                               1
    Warning: Rank mismatch in argument 'a' at (1) (scalar and rank-1)
    scipy/special/mach/r1mach.f:168.27:
    
                   CALL I1MCRA(LARGE, K, 32751, 16777215, 16777215)
                               1
    Warning: Rank mismatch in argument 'a' at (1) (scalar and rank-1)
    scipy/special/mach/r1mach.f:169.27:
    
                   CALL I1MCRA(RIGHT, K, 15520, 0, 0)
                               1
    Warning: Rank mismatch in argument 'a' at (1) (scalar and rank-1)
    scipy/special/mach/r1mach.f:170.27:
    
                   CALL I1MCRA(DIVER, K, 15536, 0, 0)
                               1
    Warning: Rank mismatch in argument 'a' at (1) (scalar and rank-1)
    scipy/special/mach/r1mach.f:171.27:
    
                   CALL I1MCRA(LOG10, K, 16339, 4461392, 10451455)
                               1
    Warning: Rank mismatch in argument 'a' at (1) (scalar and rank-1)
    gfortran:f77: scipy/special/mach/xerror.f
    scipy/special/mach/xerror.f:1.37:
    
          SUBROUTINE XERROR(MESS,NMESS,L1,L2)
                                         1
    Warning: Unused dummy argument 'l1' at (1)
    scipy/special/mach/xerror.f:1.40:
    
          SUBROUTINE XERROR(MESS,NMESS,L1,L2)
                                            1
    Warning: Unused dummy argument 'l2' at (1)
    gfortran:f77: scipy/special/mach/i1mach.f
    ar: adding 4 object files to build/temp.linux-x86_64-2.7/libsc_mach.a
    building 'sc_amos' library
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/special/amos/dgamln.f
    scipy/special/amos/dgamln.f: In function ‘dgamln’:
    scipy/special/amos/dgamln.f:1:0: warning: ‘__result_dgamln’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           DOUBLE PRECISION FUNCTION DGAMLN(Z,IERR)
     ^
    scipy/special/amos/dgamln.f:155:0: warning: ‘nz’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           ZINC = ZMIN - FLOAT(NZ)
     ^
    gfortran:f77: scipy/special/amos/zbinu.f
    gfortran:f77: scipy/special/amos/zexp.f
    gfortran:f77: scipy/special/amos/zunhj.f
    gfortran:f77: scipy/special/amos/zuni2.f
    gfortran:f77: scipy/special/amos/zacai.f
    gfortran:f77: scipy/special/amos/zasyi.f
    gfortran:f77: scipy/special/amos/zunk1.f
    scipy/special/amos/zunk1.f: In function ‘zunk1’:
    scipy/special/amos/zunk1.f:23:0: warning: ‘iflag’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           INTEGER I, IB, IFLAG, IFN, IL, INIT, INU, IUF, K, KDFLG, KFLAG,
     ^
    scipy/special/amos/zunk1.f:198:0: warning: ‘kflag’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           C1R = CSRR(KFLAG)
     ^
    gfortran:f77: scipy/special/amos/zmlri.f
    gfortran:f77: scipy/special/amos/zsqrt.f
    gfortran:f77: scipy/special/amos/zbunk.f
    gfortran:f77: scipy/special/amos/zseri.f
    gfortran:f77: scipy/special/amos/zbknu.f
    scipy/special/amos/zbknu.f: In function ‘zbknu’:
    scipy/special/amos/zbknu.f:426:0: warning: ‘cki’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             S2I = CKI*P2R + CKR*P2I + S1I
     ^
    scipy/special/amos/zbknu.f:425:0: warning: ‘ckr’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             S2R = CKR*P2R - CKI*P2I + S1R
     ^
    scipy/special/amos/zbknu.f:230:0: warning: ‘dnu2’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           FHS = DABS(0.25D0-DNU2)
     ^
    gfortran:f77: scipy/special/amos/zuoik.f
    gfortran:f77: scipy/special/amos/zbesj.f
    gfortran:f77: scipy/special/amos/zbesk.f
    gfortran:f77: scipy/special/amos/zuni1.f
    gfortran:f77: scipy/special/amos/zbesi.f
    gfortran:f77: scipy/special/amos/zdiv.f
    gfortran:f77: scipy/special/amos/zwrsk.f
    gfortran:f77: scipy/special/amos/zbiry.f
    gfortran:f77: scipy/special/amos/dsclmr.f
    gfortran:f77: scipy/special/amos/zunik.f
    gfortran:f77: scipy/special/amos/zbesy.f
    scipy/special/amos/zbesy.f:183.13:
    
          R1M5 = D1MACH(5)
                 1
    Warning: Possible change of value in conversion from REAL(8) to REAL(4) at (1)
    gfortran:f77: scipy/special/amos/zuchk.f
    gfortran:f77: scipy/special/amos/zairy.f
    gfortran:f77: scipy/special/amos/zmlt.f
    gfortran:f77: scipy/special/amos/zs1s2.f
    gfortran:f77: scipy/special/amos/zrati.f
    gfortran:f77: scipy/special/amos/zkscl.f
    gfortran:f77: scipy/special/amos/zbuni.f
    gfortran:f77: scipy/special/amos/zunk2.f
    scipy/special/amos/zunk2.f: In function ‘zunk2’:
    scipy/special/amos/zunk2.f:30:0: warning: ‘iflag’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           INTEGER I, IB, IFLAG, IFN, IL, IN, INU, IUF, K, KDFLG, KFLAG, KK,
     ^
    scipy/special/amos/zunk2.f:253:0: warning: ‘kflag’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           C1R = CSRR(KFLAG)
     ^
    gfortran:f77: scipy/special/amos/zshch.f
    gfortran:f77: scipy/special/amos/zlog.f
    gfortran:f77: scipy/special/amos/fdump.f
    gfortran:f77: scipy/special/amos/zbesh.f
    gfortran:f77: scipy/special/amos/zacon.f
    scipy/special/amos/zacon.f: In function ‘zacon’:
    scipy/special/amos/zacon.f:166:0: warning: ‘sc2r’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             S1R = SC1R*CSSR(KFLAG)
     ^
    scipy/special/amos/zacon.f:167:0: warning: ‘sc2i’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             S1I = SC1I*CSSR(KFLAG)
     ^
    gfortran:f77: scipy/special/amos/zabs.f
    ar: adding 38 object files to build/temp.linux-x86_64-2.7/libsc_amos.a
    building 'sc_cdf' library
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/special/cdflib/gamln.f
    scipy/special/cdflib/gamln.f:44.10:
    
          n = a - 1.25D0
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/special/cdflib/dzror.f
    scipy/special/cdflib/dzror.f:92.72:
    
          ASSIGN 10 TO i99999
                                                                            1
    Warning: Deleted feature: ASSIGN statement at (1)
    scipy/special/cdflib/dzror.f:100.72:
    
          ASSIGN 20 TO i99999
                                                                            1
    Warning: Deleted feature: ASSIGN statement at (1)
    scipy/special/cdflib/dzror.f:181.72:
    
          ASSIGN 200 TO i99999
                                                                            1
    Warning: Deleted feature: ASSIGN statement at (1)
    scipy/special/cdflib/dzror.f:281.72:
    
          GO TO i99999
                                                                            1
    Warning: Deleted feature: Assigned GOTO statement at (1)
    scipy/special/cdflib/dzror.f:184.5:
    
      200 fb = fx
         1
    Warning: Label 200 at (1) defined but not used
    scipy/special/cdflib/dzror.f:106.5:
    
       20 IF (.NOT. (fb.LT.0.0D0)) GO TO 40
         1
    Warning: Label 20 at (1) defined but not used
    scipy/special/cdflib/dzror.f:95.5:
    
       10 fb = fx
         1
    Warning: Label 10 at (1) defined but not used
    gfortran:f77: scipy/special/cdflib/basym.f
    gfortran:f77: scipy/special/cdflib/rcomp.f
    gfortran:f77: scipy/special/cdflib/ipmpar.f
    gfortran:f77: scipy/special/cdflib/cumnbn.f
    gfortran:f77: scipy/special/cdflib/dt1.f
    gfortran:f77: scipy/special/cdflib/esum.f
    gfortran:f77: scipy/special/cdflib/bcorr.f
    gfortran:f77: scipy/special/cdflib/cdfpoi.f
    gfortran:f77: scipy/special/cdflib/rexp.f
    gfortran:f77: scipy/special/cdflib/cumbin.f
    gfortran:f77: scipy/special/cdflib/spmpar.f
    gfortran:f77: scipy/special/cdflib/cdft.f
    gfortran:f77: scipy/special/cdflib/stvaln.f
    gfortran:f77: scipy/special/cdflib/brcomp.f
    scipy/special/cdflib/brcomp.f:78.10:
    
          n = b0 - 1.0D0
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/special/cdflib/cumgam.f
    gfortran:f77: scipy/special/cdflib/exparg.f
    gfortran:f77: scipy/special/cdflib/cdfgam.f
    gfortran:f77: scipy/special/cdflib/rlog.f
    gfortran:f77: scipy/special/cdflib/cdfchi.f
    scipy/special/cdflib/cdfchi.f: In function ‘cdfchi’:
    scipy/special/cdflib/cdfchi.f:177:0: warning: ‘porq’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               IF (porq.GT.1.5D0) THEN
     ^
    gfortran:f77: scipy/special/cdflib/cdffnc.f
    gfortran:f77: scipy/special/cdflib/cumbet.f
    gfortran:f77: scipy/special/cdflib/dinvr.f
    scipy/special/cdflib/dinvr.f:99.72:
    
          ASSIGN 10 TO i99999
                                                                            1
    Warning: Deleted feature: ASSIGN statement at (1)
    scipy/special/cdflib/dinvr.f:105.72:
    
          ASSIGN 20 TO i99999
                                                                            1
    Warning: Deleted feature: ASSIGN statement at (1)
    scipy/special/cdflib/dinvr.f:142.72:
    
          ASSIGN 90 TO i99999
                                                                            1
    Warning: Deleted feature: ASSIGN statement at (1)
    scipy/special/cdflib/dinvr.f:167.72:
    
          ASSIGN 130 TO i99999
                                                                            1
    Warning: Deleted feature: ASSIGN statement at (1)
    scipy/special/cdflib/dinvr.f:202.72:
    
          ASSIGN 200 TO i99999
                                                                            1
    Warning: Deleted feature: ASSIGN statement at (1)
    scipy/special/cdflib/dinvr.f:237.72:
    
          ASSIGN 270 TO i99999
                                                                            1
    Warning: Deleted feature: ASSIGN statement at (1)
    scipy/special/cdflib/dinvr.f:346.72:
    
          GO TO i99999
                                                                            1
    Warning: Deleted feature: Assigned GOTO statement at (1)
    scipy/special/cdflib/dinvr.f:240.5:
    
      270 CONTINUE
         1
    Warning: Label 270 at (1) defined but not used
    scipy/special/cdflib/dinvr.f:205.5:
    
      200 yy = fx
         1
    Warning: Label 200 at (1) defined but not used
    scipy/special/cdflib/dinvr.f:170.5:
    
      130 yy = fx
         1
    Warning: Label 130 at (1) defined but not used
    scipy/special/cdflib/dinvr.f:145.5:
    
       90 yy = fx
         1
    Warning: Label 90 at (1) defined but not used
    scipy/special/cdflib/dinvr.f:108.5:
    
       20 fbig = fx
         1
    Warning: Label 20 at (1) defined but not used
    scipy/special/cdflib/dinvr.f:102.5:
    
       10 fsmall = fx
         1
    Warning: Label 10 at (1) defined but not used
    gfortran:f77: scipy/special/cdflib/cdfnbn.f
    gfortran:f77: scipy/special/cdflib/brcmp1.f
    scipy/special/cdflib/brcmp1.f:77.10:
    
          n = b0 - 1.0D0
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/special/cdflib/devlpl.f
    gfortran:f77: scipy/special/cdflib/apser.f
    gfortran:f77: scipy/special/cdflib/fpser.f
    gfortran:f77: scipy/special/cdflib/dinvnr.f
    gfortran:f77: scipy/special/cdflib/cumfnc.f
    scipy/special/cdflib/cumfnc.f:116.14:
    
          icent = xnonc
                  1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/special/cdflib/cumnor.f
    gfortran:f77: scipy/special/cdflib/cumpoi.f
    gfortran:f77: scipy/special/cdflib/rlog1.f
    gfortran:f77: scipy/special/cdflib/gamma_fort.f
    scipy/special/cdflib/gamma_fort.f:1.6:
    
          DOUBLE PRECISION FUNCTION gamma(a)
          1
    Warning: 'gamma' declared at (1) is also the name of an intrinsic.  It can only be called via an explicit interface or if declared EXTERNAL.
    scipy/special/cdflib/gamma_fort.f:124.10:
    
          n = x
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/cdflib/gamma_fort.f: In function ‘gamma’:
    scipy/special/cdflib/gamma_fort.f:149:0: warning: ‘s’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           IF (a.LT.0.0D0) gamma = (1.0D0/ (gamma*s))/x
     ^
    gfortran:f77: scipy/special/cdflib/bpser.f
    scipy/special/cdflib/bpser.f:57.10:
    
          m = b0 - 1.0D0
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/special/cdflib/erf.f
    scipy/special/cdflib/erf.f:1.6:
    
          DOUBLE PRECISION FUNCTION erf(x)
          1
    Warning: 'erf' declared at (1) is also the name of an intrinsic.  It can only be called via an explicit interface or if declared EXTERNAL.
    gfortran:f77: scipy/special/cdflib/cdfchn.f
    gfortran:f77: scipy/special/cdflib/gaminv.f
    scipy/special/cdflib/gaminv.f: In function ‘gaminv’:
    scipy/special/cdflib/gaminv.f:168:0: warning: ‘b’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           IF (b.GT.bmin(iop)) GO TO 220
     ^
    gfortran:f77: scipy/special/cdflib/cdff.f
    gfortran:f77: scipy/special/cdflib/gratio.f
    gfortran:f77: scipy/special/cdflib/gam1.f
    gfortran:f77: scipy/special/cdflib/alngam.f
    gfortran:f77: scipy/special/cdflib/cumtnc.f
    gfortran:f77: scipy/special/cdflib/gsumln.f
    gfortran:f77: scipy/special/cdflib/algdiv.f
    gfortran:f77: scipy/special/cdflib/cdftnc.f
    gfortran:f77: scipy/special/cdflib/bratio.f
    scipy/special/cdflib/bratio.f:166.10:
    
      160 n = b0
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/special/cdflib/cumchn.f
    gfortran:f77: scipy/special/cdflib/erfc1.f
    gfortran:f77: scipy/special/cdflib/gamln1.f
    gfortran:f77: scipy/special/cdflib/cdfnor.f
    gfortran:f77: scipy/special/cdflib/bfrac.f
    gfortran:f77: scipy/special/cdflib/alnrel.f
    gfortran:f77: scipy/special/cdflib/psi_fort.f
    gfortran:f77: scipy/special/cdflib/grat1.f
    gfortran:f77: scipy/special/cdflib/cdfbet.f
    gfortran:f77: scipy/special/cdflib/cdfbin.f
    gfortran:f77: scipy/special/cdflib/cumt.f
    gfortran:f77: scipy/special/cdflib/cumf.f
    gfortran:f77: scipy/special/cdflib/bgrat.f
    gfortran:f77: scipy/special/cdflib/bup.f
    scipy/special/cdflib/bup.f:32.11:
    
          mu = abs(exparg(1))
               1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/cdflib/bup.f:33.10:
    
          k = exparg(0)
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/cdflib/bup.f:55.22:
    
          IF (r.LT.t) k = r
                          1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/special/cdflib/cumchi.f
    gfortran:f77: scipy/special/cdflib/betaln.f
    scipy/special/cdflib/betaln.f:55.10:
    
          n = a - 1.0D0
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/cdflib/betaln.f:69.10:
    
       60 n = b - 1.0D0
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/cdflib/betaln.f:80.10:
    
       80 n = a - 1.0D0
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    ar: adding 50 object files to build/temp.linux-x86_64-2.7/libsc_cdf.a
    ar: adding 14 object files to build/temp.linux-x86_64-2.7/libsc_cdf.a
    building 'sc_specfun' library
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/special/specfun/specfun.f
    scipy/special/specfun/specfun.f:49.27:
    
                     CDN=CMPLX(PD,0.0D0)
                               1
    Warning: Conversion from REAL(8) to default-kind COMPLEX(4) at (1) might loose precision, consider using the KIND argument
    scipy/special/specfun/specfun.f:57.24:
    
                  CDN=CMPLX(G0,0.0D0)
                            1
    Warning: Conversion from REAL(8) to default-kind COMPLEX(4) at (1) might loose precision, consider using the KIND argument
    scipy/special/specfun/specfun.f:248.16:
    
            Z=CMPLX(X,Y)
                    1
    Warning: Conversion from REAL(8) to default-kind COMPLEX(4) at (1) might loose precision, consider using the KIND argument
    scipy/special/specfun/specfun.f:4887.17:
    
                  M1=X-1
                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:1195.14:
    
               NN=N1-(N1-N0)/(1.0D0-F0/F1)
                  1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:1234.14:
    
               NN=N1-(N1-N0)/(1.0D0-F0/F1)
                  1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:1558.13:
    
               N=XA
                 1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:1563.13:
    
               N=XA-.5
                 1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:1805.18:
    
            CER=CMPLX(ERR,ERI)
                      1
    Warning: Conversion from REAL(8) to default-kind COMPLEX(4) at (1) might loose precision, consider using the KIND argument
    scipy/special/specfun/specfun.f:2223.16:
    
            Z=CMPLX(X,Y)
                    1
    Warning: Conversion from REAL(8) to default-kind COMPLEX(4) at (1) might loose precision, consider using the KIND argument
    scipy/special/specfun/specfun.f:2530.16:
    
               ISGN=1.0D0
                    1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:3360.12:
    
            LB0=0.0D0
                1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:4843.19:
    
               Z=CMPLX(PX,PY)
                       1
    Warning: Conversion from REAL(8) to default-kind COMPLEX(4) at (1) might loose precision, consider using the KIND argument
    scipy/special/specfun/specfun.f:8125.11:
    
            ID=15-ABS(D1-D2)
               1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:9310.20:
    
            IF (IL1) NM=ABS(A)
                        1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:9311.20:
    
            IF (IL2) NM=ABS(AA)
                        1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:9330.14:
    
    20         ID=ABS(LOG10(RA))
                  1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:9801.10:
    
            N=ABS(B-1)
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:9841.11:
    
            ID=15-ABS(DA1-DA2)
               1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:9876.12:
    
            ID1=15-ABS(DB1-DB2)
                1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:9887.12:
    
            ID2=0.0D0
                1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/special/specfun/specfun.f:5230.16:
    
            Z=CMPLX(X,Y)
                    1
    Warning: Conversion from REAL(8) to default-kind COMPLEX(4) at (1) might loose precision, consider using the KIND argument
    scipy/special/specfun/specfun.f:8705.2:
    
     6         FNAN=DNAN()
      1
    scipy/special/specfun/specfun.f:8665.72:
    
               IF (JM+1.GT.251) GOTO 6
                                                                            2
    Warning: Legacy Extension: Label at (1) is not in the same block as the GOTO statement at (2)
    scipy/special/specfun/specfun.f:7736.19:
    
               Z=CMPLX(PX,PY)
                       1
    Warning: Conversion from REAL(8) to default-kind COMPLEX(4) at (1) might loose precision, consider using the KIND argument
    scipy/special/specfun/specfun.f:9935.19:
    
            ZERO=CMPLX(X,Y)
                       1
    Warning: Conversion from REAL(8) to default-kind COMPLEX(4) at (1) might loose precision, consider using the KIND argument
    scipy/special/specfun/specfun.f:10189.17:
    
            CI=CMPLX(0.0D0,1.0D0)
                     1
    Warning: Conversion from REAL(8) to default-kind COMPLEX(4) at (1) might loose precision, consider using the KIND argument
    scipy/special/specfun/specfun.f:12123.16:
    
            Z=CMPLX(X,Y)
                    1
    Warning: Conversion from REAL(8) to default-kind COMPLEX(4) at (1) might loose precision, consider using the KIND argument
    scipy/special/specfun/specfun.f:5581.34:
    
            DOUBLE COMPLEX Z, CEI, IMF
                                      1
    Warning: Unused variable 'imf' declared at (1)
    scipy/special/specfun/specfun.f: In function ‘fcoef’:
    scipy/special/specfun/specfun.f:8679:0: warning: ‘jm’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                      FC(JM) = 1.0D0
     ^
    scipy/special/specfun/specfun.f: In function ‘cik01’:
    scipy/special/specfun/specfun.f:12708:0: warning: ‘IMAGPART_EXPR <cw>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                   IF (CDABS((CS-CW)/CS).LT.1.0D-15) GO TO 45
     ^
    scipy/special/specfun/specfun.f:12708:0: warning: ‘REALPART_EXPR <cw>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘cjynb’:
    scipy/special/specfun/specfun.f:6790:0: warning: ‘IMAGPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                   CS0=(CBS+CF)/CDCOS(Z)
     ^
    scipy/special/specfun/specfun.f:6790:0: warning: ‘REALPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘ciknb’:
    scipy/special/specfun/specfun.f:12358:0: warning: ‘IMAGPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             CS0=CDEXP(Z1)/(CBS-CF)
     ^
    scipy/special/specfun/specfun.f:12358:0: warning: ‘REALPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘csphjy’:
    scipy/special/specfun/specfun.f:1158:0: warning: ‘IMAGPART_EXPR <cs>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     20            CSJ(K)=CS*CSJ(K)
     ^
    scipy/special/specfun/specfun.f:1158:0: warning: ‘REALPART_EXPR <cs>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘cjylv’:
    scipy/special/specfun/specfun.f:1448:0: warning: ‘IMAGPART_EXPR <cfy>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             CDYV=-V/Z*CBYV+CFY
     ^
    scipy/special/specfun/specfun.f:1448:0: warning: ‘REALPART_EXPR <cfy>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘cisia’:
    scipy/special/specfun/specfun.f:2070:0: warning: ‘bj[0]’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                XS=BJ(1)
     ^
    scipy/special/specfun/specfun.f: In function ‘clqn’:
    scipy/special/specfun/specfun.f:2237:0: warning: ‘IMAGPART_EXPR <cqf0>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                CQF0=CQ0
     ^
    scipy/special/specfun/specfun.f:2237:0: warning: ‘REALPART_EXPR <cqf0>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘cjyva’:
    scipy/special/specfun/specfun.f:3491:0: warning: ‘IMAGPART_EXPR <cs>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     80            CBJ(K)=CS*CBJ(K)
     ^
    scipy/special/specfun/specfun.f:3491:0: warning: ‘REALPART_EXPR <cs>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f:3470:0: warning: ‘IMAGPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                   CF=2.0D0*(K+V0-1.0D0)/Z*CF1-CF0
     ^
    scipy/special/specfun/specfun.f:3470:0: warning: ‘REALPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f:3427:0: warning: ‘IMAGPART_EXPR <cju0>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                   CYV0=(CJV0*DCOS(PV0)-CJU0)/DSIN(PV0)
     ^
    scipy/special/specfun/specfun.f:3427:0: warning: ‘REALPART_EXPR <cju0>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f:3406:0: warning: ‘IMAGPART_EXPR <cyv1>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                      CYV1=CA0*(CPZ*CSK+CQZ*CCK)
     ^
    scipy/special/specfun/specfun.f:3406:0: warning: ‘REALPART_EXPR <cyv1>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f:3403:0: warning: ‘IMAGPART_EXPR <cyv0>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                      CYV0=CA0*(CPZ*CSK+CQZ*CCK)
     ^
    scipy/special/specfun/specfun.f:3403:0: warning: ‘REALPART_EXPR <cyv0>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f:3506:0: warning: ‘cyv0’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                YA1=CDABS(CG0)
     ^
    scipy/special/specfun/specfun.f:3458:0: warning: ‘IMAGPART_EXPR <cjv0>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                   CYV0=CYV0/CFAC0+2.0D0*CI*DCOS(PV0)*CJV0
     ^
    scipy/special/specfun/specfun.f:3458:0: warning: ‘REALPART_EXPR <cjv0>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f:3488:0: warning: ‘cjv0’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                IF (CDABS(CJV0).GT.CDABS(CJV1)) CS=CJV0/CF
     ^
    scipy/special/specfun/specfun.f: In function ‘cjyvb’:
    scipy/special/specfun/specfun.f:3693:0: warning: ‘IMAGPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             CS=CJV0/CF
     ^
    scipy/special/specfun/specfun.f:3693:0: warning: ‘REALPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f:3651:0: warning: ‘IMAGPART_EXPR <cyv0>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                CYV0=CA0*(CPZ*CSK+CQZ*CCK)
     ^
    scipy/special/specfun/specfun.f:3651:0: warning: ‘REALPART_EXPR <cyv0>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘ciklv’:
    scipy/special/specfun/specfun.f:5419:0: warning: ‘IMAGPART_EXPR <cfk>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             CDKV=-CFK-V/Z*CBKV
     ^
    scipy/special/specfun/specfun.f:5419:0: warning: ‘REALPART_EXPR <cfk>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘hygfx’:
    scipy/special/specfun/specfun.f:5827:0: warning: ‘nm’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                IF (L2) NM=INT(ABS(A))
     ^
    scipy/special/specfun/specfun.f: In function ‘cchg’:
    scipy/special/specfun/specfun.f:6079:0: warning: ‘IMAGPART_EXPR <cy1>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                      CHG=((2.0D0*A-B+Z)*CY1+(B-A)*CY0)/A
     ^
    scipy/special/specfun/specfun.f:6079:0: warning: ‘REALPART_EXPR <cy1>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f:6079:0: warning: ‘IMAGPART_EXPR <cy0>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f:6079:0: warning: ‘REALPART_EXPR <cy0>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f:6040:0: warning: ‘IMAGPART_EXPR <chw>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                         IF (CDABS((CHG-CHW)/CHG).LT.1.D-15) GO TO 25
     ^
    scipy/special/specfun/specfun.f:6040:0: warning: ‘REALPART_EXPR <chw>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘hygfz’:
    scipy/special/specfun/specfun.f:6377:0: warning: ‘k’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             IF (K.GT.150) WRITE(*,160)
     ^
    scipy/special/specfun/specfun.f:6311:0: warning: ‘IMAGPART_EXPR <zw>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                      IF (CDABS((ZHF-ZW)/ZHF).LE.EPS) GO TO 115
     ^
    scipy/special/specfun/specfun.f:6311:0: warning: ‘REALPART_EXPR <zw>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘cjyna’:
    scipy/special/specfun/specfun.f:6637:0: warning: ‘IMAGPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                   CF=2.0D0*(K+1.0D0)/Z*CF1-CF2
     ^
    scipy/special/specfun/specfun.f:6637:0: warning: ‘REALPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘csphik’:
    scipy/special/specfun/specfun.f:10213:0: warning: ‘IMAGPART_EXPR <cs>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     20            CSI(K)=CS*CSI(K)
     ^
    scipy/special/specfun/specfun.f:10213:0: warning: ‘REALPART_EXPR <cs>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f:10206:0: warning: ‘IMAGPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                   CF=(2.0D0*K+3.0D0)*CF1/Z+CF0
     ^
    scipy/special/specfun/specfun.f:10206:0: warning: ‘REALPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘cikvb’:
    scipy/special/specfun/specfun.f:11242:0: warning: ‘IMAGPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             CS=CBI0/CF
     ^
    scipy/special/specfun/specfun.f:11242:0: warning: ‘REALPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘cikva’:
    scipy/special/specfun/specfun.f:11403:0: warning: ‘IMAGPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             CS=CBI0/CF
     ^
    scipy/special/specfun/specfun.f:11403:0: warning: ‘REALPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘clqmn’:
    scipy/special/specfun/specfun.f:12163:0: warning: ‘IMAGPART_EXPR <cqf0>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                   CQF0=((2*K+3.0D0)*Z*CQF1-(K+2.0D0)*CQF2)/(K+1.0D0)
     ^
    scipy/special/specfun/specfun.f:12163:0: warning: ‘REALPART_EXPR <cqf0>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘cikna’:
    scipy/special/specfun/specfun.f:12466:0: warning: ‘IMAGPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             CS=CBI0/CF
     ^
    scipy/special/specfun/specfun.f:12466:0: warning: ‘REALPART_EXPR <cf>’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/special/specfun/specfun.f: In function ‘stvhv’:
    scipy/special/specfun/specfun.f:13026:0: warning: ‘bjv’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                      BYV = DCOS(V*PI)*BYV + DSIN(-V*PI)*BJV
     ^
    ar: adding 1 object files to build/temp.linux-x86_64-2.7/libsc_specfun.a
    building 'statlib' library
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -c'
    gfortran:f77: scipy/stats/statlib/ansari.f
    gfortran:f77: scipy/stats/statlib/swilk.f
    gfortran:f77: scipy/stats/statlib/spearman.f
    scipy/stats/statlib/spearman.f:12.49:
    
          double precision zero, one, two, b, x, y, z, u, six,
                                                     1
    Warning: Unused variable 'z' declared at (1)
    ar: adding 3 object files to build/temp.linux-x86_64-2.7/libstatlib.a
    customize UnixCCompiler
    customize UnixCCompiler using build_ext
    resetting extension 'scipy.integrate._odepack' language from 'c' to 'f77'.
    resetting extension 'scipy.integrate.vode' language from 'c' to 'f77'.
    resetting extension 'scipy.integrate.lsoda' language from 'c' to 'f77'.
    resetting extension 'scipy.odr.__odrpack' language from 'c' to 'f77'.
    extending extension 'scipy.sparse.linalg.dsolve._superlu' defined_macros with [('USE_VENDOR_BLAS', 1)]
    customize UnixCCompiler
    customize UnixCCompiler using build_ext
    customize Gnu95FCompiler
    customize Gnu95FCompiler
    customize Gnu95FCompiler using build_ext
    building 'scipy.cluster._vq' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/cluster/src/vq_module.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/cluster/src/vq_module.c:7:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    scipy/cluster/src/vq_module.c: In function ‘compute_vq’:
    scipy/cluster/src/vq_module.c:54:18: warning: variable ‘nd’ set but not used [-Wunused-but-set-variable]
         npy_intp nc, nd;
                      ^
    x86_64-linux-gnu-gcc: scipy/cluster/src/vq.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/cluster/src/vq.h:6,
                     from scipy/cluster/src/vq.c:16:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/cluster/src/vq.h:6,
                     from scipy/cluster/src/vq.c:16:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1629:1: warning: ‘_import_array’ defined but not used [-Wunused-function]
     _import_array(void)
     ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/cluster/src/vq_module.o build/temp.linux-x86_64-2.7/scipy/cluster/src/vq.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/cluster/_vq.so
    building 'scipy.cluster._hierarchy_wrap' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/cluster/src/hierarchy.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from scipy/cluster/src/hierarchy.c:37:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    scipy/cluster/src/hierarchy.c: In function ‘dist_ward’:
    scipy/cluster/src/hierarchy.c:257:10: warning: variable ‘m’ set but not used [-Wunused-but-set-variable]
       int i, m, xi, rind, sind;
              ^
    scipy/cluster/src/hierarchy.c:256:17: warning: variable ‘centroid_tq’ set but not used [-Wunused-but-set-variable]
       const double *centroid_tq;
                     ^
    scipy/cluster/src/hierarchy.c: In function ‘print_dm’:
    scipy/cluster/src/hierarchy.c:320:17: warning: variable ‘row’ set but not used [-Wunused-but-set-variable]
       const double *row;
                     ^
    scipy/cluster/src/hierarchy.c: In function ‘linkage’:
    scipy/cluster/src/hierarchy.c:371:40: warning: variable ‘npc2’ set but not used [-Wunused-but-set-variable]
       int i, j, k, t, np, nid, mini, minj, npc2;
                                            ^
    scipy/cluster/src/hierarchy.c: In function ‘linkage_alt’:
    scipy/cluster/src/hierarchy.c:648:40: warning: variable ‘npc2’ set but not used [-Wunused-but-set-variable]
       int i, j, k, t, np, nid, mini, minj, npc2;
                                            ^
    scipy/cluster/src/hierarchy.c: In function ‘form_member_list’:
    scipy/cluster/src/hierarchy.c:1130:30: warning: variable ‘rn’ set but not used [-Wunused-but-set-variable]
       int ndid, lid, rid, k, ln, rn;
                                  ^
    scipy/cluster/src/hierarchy.c: In function ‘form_flat_clusters_maxclust_monocrit’:
    scipy/cluster/src/hierarchy.c:1310:7: warning: variable ‘min_legal_nc’ set but not used [-Wunused-but-set-variable]
       int min_legal_nc = 1;
           ^
    scipy/cluster/src/hierarchy.c:1303:33: warning: variable ‘ms’ set but not used [-Wunused-but-set-variable]
       int ndid, lid, rid, k, nc, g, ms;
                                     ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from scipy/cluster/src/hierarchy.c:37:
    scipy/cluster/src/hierarchy.c: At top level:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1629:1: warning: ‘_import_array’ defined but not used [-Wunused-function]
     _import_array(void)
     ^
    x86_64-linux-gnu-gcc: scipy/cluster/src/hierarchy_wrap.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/cluster/src/hierarchy_wrap.c:39:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    scipy/cluster/src/hierarchy_wrap.c: In function ‘linkage_euclid_wrap’:
    scipy/cluster/src/hierarchy_wrap.c:83:21: warning: variable ‘ml’ set but not used [-Wunused-but-set-variable]
       int method, m, n, ml;
                         ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/cluster/src/hierarchy_wrap.o build/temp.linux-x86_64-2.7/scipy/cluster/src/hierarchy.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/cluster/_hierarchy_wrap.so
    building 'scipy.fftpack._fftpack' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Iscipy/fftpack/src -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/fftpack/src/zrfft.c
    x86_64-linux-gnu-gcc: scipy/fftpack/src/zfftnd.c
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/fftpack/src/dct.c
    scipy/fftpack/src/dct.c.src: In function ‘dct1’:
    scipy/fftpack/src/dct.c.src:46:29: warning: unused variable ‘n2’ [-Wunused-variable]
         @type@ *ptr = inout, n1, n2;
                                 ^
    scipy/fftpack/src/dct.c.src:46:25: warning: unused variable ‘n1’ [-Wunused-variable]
         @type@ *ptr = inout, n1, n2;
                             ^
    scipy/fftpack/src/dct.c.src:45:12: warning: unused variable ‘j’ [-Wunused-variable]
         int i, j;
                ^
    scipy/fftpack/src/dct.c.src: In function ‘ddct1’:
    scipy/fftpack/src/dct.c.src:46:30: warning: unused variable ‘n2’ [-Wunused-variable]
         @type@ *ptr = inout, n1, n2;
                                  ^
    scipy/fftpack/src/dct.c.src:46:26: warning: unused variable ‘n1’ [-Wunused-variable]
         @type@ *ptr = inout, n1, n2;
                              ^
    scipy/fftpack/src/dct.c.src:45:12: warning: unused variable ‘j’ [-Wunused-variable]
         int i, j;
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/fftpack/src/dst.c
    scipy/fftpack/src/dst.c.src: In function ‘dst1’:
    scipy/fftpack/src/dst.c.src:46:29: warning: unused variable ‘n2’ [-Wunused-variable]
         @type@ *ptr = inout, n1, n2;
                                 ^
    scipy/fftpack/src/dst.c.src:46:25: warning: unused variable ‘n1’ [-Wunused-variable]
         @type@ *ptr = inout, n1, n2;
                             ^
    scipy/fftpack/src/dst.c.src:45:12: warning: unused variable ‘j’ [-Wunused-variable]
         int i, j;
                ^
    scipy/fftpack/src/dst.c.src: In function ‘ddst1’:
    scipy/fftpack/src/dst.c.src:46:30: warning: unused variable ‘n2’ [-Wunused-variable]
         @type@ *ptr = inout, n1, n2;
                                  ^
    scipy/fftpack/src/dst.c.src:46:26: warning: unused variable ‘n1’ [-Wunused-variable]
         @type@ *ptr = inout, n1, n2;
                              ^
    scipy/fftpack/src/dst.c.src:45:12: warning: unused variable ‘j’ [-Wunused-variable]
         int i, j;
                ^
    x86_64-linux-gnu-gcc: scipy/fftpack/src/zfft.c
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: scipy/fftpack/src/drfft.c
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/fftpack/_fftpackmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/fftpack/_fftpackmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/fftpack/_fftpackmodule.o build/temp.linux-x86_64-2.7/scipy/fftpack/src/zfft.o build/temp.linux-x86_64-2.7/scipy/fftpack/src/drfft.o build/temp.linux-x86_64-2.7/scipy/fftpack/src/zrfft.o build/temp.linux-x86_64-2.7/scipy/fftpack/src/zfftnd.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/fftpack/src/dct.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/fftpack/src/dst.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o -Lbuild/temp.linux-x86_64-2.7 -ldfftpack -lfftpack -lgfortran -o build/lib.linux-x86_64-2.7/scipy/fftpack/_fftpack.so
    building 'scipy.fftpack.convolve' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/fftpack/convolvemodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/fftpack/convolvemodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/fftpack/convolvemodule.c:130:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: scipy/fftpack/src/convolve.c
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/fftpack/convolvemodule.o build/temp.linux-x86_64-2.7/scipy/fftpack/src/convolve.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o -Lbuild/temp.linux-x86_64-2.7 -ldfftpack -lgfortran -o build/lib.linux-x86_64-2.7/scipy/fftpack/convolve.so
    building 'scipy.integrate._quadpack' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/integrate/_quadpackmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:23,
                     from scipy/integrate/quadpack.h:32,
                     from scipy/integrate/_quadpackmodule.c:4:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from scipy/integrate/_quadpackmodule.c:5:0:
    scipy/integrate/__quadpack.h:54:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     void DQAGSE();
     ^
    scipy/integrate/__quadpack.h:55:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     void DQAGIE();
     ^
    scipy/integrate/__quadpack.h:56:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     void DQAGPE();
     ^
    scipy/integrate/__quadpack.h:57:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     void DQAWOE();
     ^
    scipy/integrate/__quadpack.h:58:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     void DQAWFE();
     ^
    scipy/integrate/__quadpack.h:59:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     void DQAWSE();
     ^
    scipy/integrate/__quadpack.h:60:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     void DQAWCE();
     ^
    scipy/integrate/__quadpack.h: In function ‘quad_function’:
    scipy/integrate/__quadpack.h:190:20: warning: unused variable ‘nb’ [-Wunused-variable]
       PyNumberMethods *nb;
                        ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/scipy/integrate/_quadpackmodule.o -Lbuild/temp.linux-x86_64-2.7 -lquadpack -llinpack_lite -lmach -lgfortran -o build/lib.linux-x86_64-2.7/scipy/integrate/_quadpack.so
    building 'scipy.integrate._odepack' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/integrate/_odepackmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:23,
                     from scipy/integrate/multipack.h:32,
                     from scipy/integrate/_odepackmodule.c:4:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from scipy/integrate/_odepackmodule.c:4:0:
    scipy/integrate/multipack.h: In function ‘call_python_function’:
    scipy/integrate/multipack.h:151:27: warning: unused variable ‘str1’ [-Wunused-variable]
       PyObject *arg1 = NULL, *str1 = NULL;
                               ^
    In file included from scipy/integrate/_odepackmodule.c:6:0:
    scipy/integrate/__odepack.h: At top level:
    scipy/integrate/__odepack.h:31:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     void LSODA();
     ^
    In file included from scipy/integrate/_odepackmodule.c:4:0:
    scipy/integrate/multipack.h:110:24: warning: ‘my_make_numpy_array’ defined but not used [-Wunused-function]
     static PyArrayObject * my_make_numpy_array(PyObject *y0, int type, int mindim, int maxdim)
                            ^
    In file included from scipy/integrate/_odepackmodule.c:6:0:
    scipy/integrate/__odepack.h: In function ‘odepack_odeint’:
    scipy/integrate/__odepack.h:325:87: warning: ‘tcrit’ may be used uninitialized in this function [-Wmaybe-uninitialized]
         if (itask == 4 && *tout_ptr > *(tcrit + crit_ind)) {crit_ind++; rwork[0] = *(tcrit+crit_ind);}
                                                                                           ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/scipy/integrate/_odepackmodule.o -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -lodepack -llinpack_lite -lmach -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/integrate/_odepack.so
    building 'scipy.integrate.vode' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:347:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern void F_FUNC(dvode,DVODE)();
     ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:348:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern void F_FUNC(zvode,ZVODE)();
     ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c: In function ‘cb_f_in_dvode__user__routines’:
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:372:7: warning: unused variable ‘ipar’ [-Wunused-variable]
       int ipar=(*ipar_cb_capi);
           ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:371:10: warning: unused variable ‘rpar’ [-Wunused-variable]
       double rpar=(*rpar_cb_capi);
              ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c: In function ‘cb_jac_in_dvode__user__routines’:
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:504:7: warning: unused variable ‘ipar’ [-Wunused-variable]
       int ipar=(*ipar_cb_capi);
           ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:503:10: warning: unused variable ‘rpar’ [-Wunused-variable]
       double rpar=(*rpar_cb_capi);
              ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:501:7: warning: unused variable ‘mu’ [-Wunused-variable]
       int mu=(*mu_cb_capi);
           ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:500:7: warning: unused variable ‘ml’ [-Wunused-variable]
       int ml=(*ml_cb_capi);
           ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c: In function ‘cb_f_in_zvode__user__routines’:
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:632:7: warning: unused variable ‘ipar’ [-Wunused-variable]
       int ipar=(*ipar_cb_capi);
           ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:631:10: warning: unused variable ‘rpar’ [-Wunused-variable]
       double rpar=(*rpar_cb_capi);
              ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c: In function ‘cb_jac_in_zvode__user__routines’:
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:764:7: warning: unused variable ‘ipar’ [-Wunused-variable]
       int ipar=(*ipar_cb_capi);
           ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:763:10: warning: unused variable ‘rpar’ [-Wunused-variable]
       double rpar=(*rpar_cb_capi);
              ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:761:7: warning: unused variable ‘mu’ [-Wunused-variable]
       int mu=(*mu_cb_capi);
           ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:760:7: warning: unused variable ‘ml’ [-Wunused-variable]
       int ml=(*ml_cb_capi);
           ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c: At top level:
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:919:28: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
                                void (*f2py_func)()) {
                                ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c: In function ‘f2py_rout_vode_dvode’:
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:933:43: warning: variable ‘jac_cptr’ set but not used [-Wunused-but-set-variable]
       cb_jac_in_dvode__user__routines_typedef jac_cptr;
                                               ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:928:41: warning: variable ‘f_cptr’ set but not used [-Wunused-but-set-variable]
       cb_f_in_dvode__user__routines_typedef f_cptr;
                                             ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c: At top level:
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:1247:28: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
                                void (*f2py_func)()) {
                                ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c: In function ‘f2py_rout_vode_zvode’:
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:1261:43: warning: variable ‘jac_cptr’ set but not used [-Wunused-but-set-variable]
       cb_jac_in_zvode__user__routines_typedef jac_cptr;
                                               ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:1256:41: warning: variable ‘f_cptr’ set but not used [-Wunused-but-set-variable]
       cb_f_in_zvode__user__routines_typedef f_cptr;
                                             ^
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c: At top level:
    build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.c:144:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/integrate/vodemodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -lodepack -llinpack_lite -lmach -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/integrate/vode.so
    building 'scipy.integrate.lsoda' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.c:345:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern void F_FUNC(lsoda,LSODA)();
     ^
    build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.c: In function ‘cb_jac_in_lsoda__user__routines’:
    build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.c:496:7: warning: unused variable ‘mu’ [-Wunused-variable]
       int mu=(*mu_cb_capi);
           ^
    build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.c:495:7: warning: unused variable ‘ml’ [-Wunused-variable]
       int ml=(*ml_cb_capi);
           ^
    build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.c: At top level:
    build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.c:652:28: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
                                void (*f2py_func)()) {
                                ^
    build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.c: In function ‘f2py_rout_lsoda_lsoda’:
    build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.c:710:43: warning: variable ‘jac_cptr’ set but not used [-Wunused-but-set-variable]
       cb_jac_in_lsoda__user__routines_typedef jac_cptr;
                                               ^
    build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.c:661:41: warning: variable ‘f_cptr’ set but not used [-Wunused-but-set-variable]
       cb_f_in_lsoda__user__routines_typedef f_cptr;
                                             ^
    build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.c: At top level:
    build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.c:142:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/integrate/lsodamodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -lodepack -llinpack_lite -lmach -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/integrate/lsoda.so
    building 'scipy.integrate._dop' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/integrate/_dopmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/integrate/_dopmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/integrate/_dopmodule.c: In function ‘cb_fcn_in___user__routines’:
    build/src.linux-x86_64-2.7/scipy/integrate/_dopmodule.c:370:7: warning: unused variable ‘ipar’ [-Wunused-variable]
       int ipar=(*ipar_cb_capi);
           ^
    build/src.linux-x86_64-2.7/scipy/integrate/_dopmodule.c:369:10: warning: unused variable ‘rpar’ [-Wunused-variable]
       double rpar=(*rpar_cb_capi);
              ^
    build/src.linux-x86_64-2.7/scipy/integrate/_dopmodule.c: In function ‘cb_solout_in___user__routines’:
    build/src.linux-x86_64-2.7/scipy/integrate/_dopmodule.c:503:7: warning: unused variable ‘irtn’ [-Wunused-variable]
       int irtn=(*irtn_cb_capi);
           ^
    build/src.linux-x86_64-2.7/scipy/integrate/_dopmodule.c:502:7: warning: unused variable ‘ipar’ [-Wunused-variable]
       int ipar=(*ipar_cb_capi);
           ^
    build/src.linux-x86_64-2.7/scipy/integrate/_dopmodule.c:501:10: warning: unused variable ‘rpar’ [-Wunused-variable]
       double rpar=(*rpar_cb_capi);
              ^
    build/src.linux-x86_64-2.7/scipy/integrate/_dopmodule.c: At top level:
    build/src.linux-x86_64-2.7/scipy/integrate/_dopmodule.c:142:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/integrate/_dopmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o -Lbuild/temp.linux-x86_64-2.7 -ldop -lgfortran -o build/lib.linux-x86_64-2.7/scipy/integrate/_dop.so
    building 'scipy.interpolate.interpnd' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/interpolate/interpnd.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from scipy/interpolate/interpnd.c:311:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from scipy/interpolate/interpnd.c:311:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1629:1: warning: ‘_import_array’ defined but not used [-Wunused-function]
     _import_array(void)
     ^
    scipy/interpolate/interpnd.c: In function ‘__pyx_pf_5scipy_11interpolate_8interpnd_26CloughTocher2DInterpolator_10_do_evaluate.isra.37’:
    scipy/interpolate/interpnd.c:25379:25: warning: ‘__pyx_v_g3’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             z.imag = a.real * b.imag + a.imag * b.real;
                             ^
    scipy/interpolate/interpnd.c:7772:10: note: ‘__pyx_v_g3’ was declared here
       double __pyx_v_g3;
              ^
    scipy/interpolate/interpnd.c:25379:25: warning: ‘__pyx_v_g2’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             z.imag = a.real * b.imag + a.imag * b.real;
                             ^
    scipy/interpolate/interpnd.c:7771:10: note: ‘__pyx_v_g2’ was declared here
       double __pyx_v_g2;
              ^
    scipy/interpolate/interpnd.c:25379:25: warning: ‘__pyx_v_g1’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             z.imag = a.real * b.imag + a.imag * b.real;
                             ^
    scipy/interpolate/interpnd.c:7770:10: note: ‘__pyx_v_g1’ was declared here
       double __pyx_v_g1;
              ^
    scipy/interpolate/interpnd.c: In function ‘__pyx_pf_5scipy_11interpolate_8interpnd_26CloughTocher2DInterpolator_8_do_evaluate.isra.38’:
    scipy/interpolate/interpnd.c:7593:33: warning: ‘__pyx_v_g3’ may be used uninitialized in this function [-Wmaybe-uninitialized]
       __pyx_v_c1101 = (((__pyx_v_g3 * ((((-__pyx_v_c3000) + (3.0 * __pyx_v_c2100)) - (3.0 * __pyx_v_c1200)) + __pyx_v_c0300)) + (((((-__pyx_v_c3000) + (2.0 * __pyx_v_c2100)) - __pyx_v_c1200) + __pyx_v_c2001) + __pyx_v_c0201)) / 2.0);
                                     ^
    scipy/interpolate/interpnd.c:7040:10: note: ‘__pyx_v_g3’ was declared here
       double __pyx_v_g3;
              ^
    scipy/interpolate/interpnd.c:7584:33: warning: ‘__pyx_v_g2’ may be used uninitialized in this function [-Wmaybe-uninitialized]
       __pyx_v_c1011 = (((__pyx_v_g2 * ((((-__pyx_v_c0030) + (3.0 * __pyx_v_c1020)) - (3.0 * __pyx_v_c2010)) + __pyx_v_c3000)) + (((((-__pyx_v_c0030) + (2.0 * __pyx_v_c1020)) - __pyx_v_c2010) + __pyx_v_c2001) + __pyx_v_c0021)) / 2.0);
                                     ^
    scipy/interpolate/interpnd.c:7039:10: note: ‘__pyx_v_g2’ was declared here
       double __pyx_v_g2;
              ^
    scipy/interpolate/interpnd.c:7575:33: warning: ‘__pyx_v_g1’ may be used uninitialized in this function [-Wmaybe-uninitialized]
       __pyx_v_c0111 = (((__pyx_v_g1 * ((((-__pyx_v_c0300) + (3.0 * __pyx_v_c0210)) - (3.0 * __pyx_v_c0120)) + __pyx_v_c0030)) + (((((-__pyx_v_c0300) + (2.0 * __pyx_v_c0210)) - __pyx_v_c0120) + __pyx_v_c0021) + __pyx_v_c0201)) / 2.0);
                                     ^
    scipy/interpolate/interpnd.c:7038:10: note: ‘__pyx_v_g1’ was declared here
       double __pyx_v_g1;
              ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/interpolate/interpnd.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/interpolate/interpnd.so
    building 'scipy.interpolate._fitpack' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/interpolate/src/_fitpackmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/interpolate/src/multipack.h:32,
                     from scipy/interpolate/src/_fitpackmodule.c:5:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from scipy/interpolate/src/_fitpackmodule.c:5:0:
    scipy/interpolate/src/multipack.h: In function ‘call_python_function’:
    scipy/interpolate/src/multipack.h:164:27: warning: unused variable ‘str1’ [-Wunused-variable]
       PyObject *arg1 = NULL, *str1 = NULL;
                               ^
    scipy/interpolate/src/_fitpackmodule.c: At top level:
    scipy/interpolate/src/multipack.h:123:24: warning: ‘my_make_numpy_array’ defined but not used [-Wunused-function]
     static PyArrayObject * my_make_numpy_array(PyObject *y0, int type, int mindim, int maxdim)
                            ^
    scipy/interpolate/src/multipack.h:147:18: warning: ‘call_python_function’ defined but not used [-Wunused-function]
     static PyObject *call_python_function(PyObject *func, npy_intp n, double *x, PyObject *args, int dim, PyObject *error_obj)
                      ^
    In file included from scipy/interpolate/src/_fitpackmodule.c:7:0:
    scipy/interpolate/src/__fitpack.h: In function ‘_bspldismat’:
    scipy/interpolate/src/__fitpack.h:1379:20: warning: ‘dx’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                 factor = pow(dx, (double)k);
                        ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/scipy/interpolate/src/_fitpackmodule.o -Lbuild/temp.linux-x86_64-2.7 -lfitpack -lgfortran -o build/lib.linux-x86_64-2.7/scipy/interpolate/_fitpack.so
    building 'scipy.interpolate.dfitpack' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/interpolate/src/dfitpackmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/interpolate/src/dfitpackmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/interpolate/src/dfitpackmodule.c:151:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: build/src.linux-x86_64-2.7/scipy/interpolate/src/dfitpack-f2pywrappers.f
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/interpolate/src/dfitpackmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/interpolate/src/dfitpack-f2pywrappers.o -Lbuild/temp.linux-x86_64-2.7 -lfitpack -lgfortran -o build/lib.linux-x86_64-2.7/scipy/interpolate/dfitpack.so
    building 'scipy.interpolate._interpolate' extension
    compiling C++ sources
    C compiler: c++ -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -fPIC
    
    compile options: '-Iscipy/interpolate/src -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    c++: scipy/interpolate/src/_interpolate.cpp
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/interpolate/src/_interpolate.cpp:5:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    scipy/interpolate/src/_interpolate.cpp: In function ‘PyObject* linear_method(PyObject*, PyObject*, PyObject*)’:
    scipy/interpolate/src/_interpolate.cpp:13:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         static char *kwlist[] = {"x","y","new_x","new_y", NULL};
                                                               ^
    scipy/interpolate/src/_interpolate.cpp:13:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
    scipy/interpolate/src/_interpolate.cpp:13:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
    scipy/interpolate/src/_interpolate.cpp:13:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
    scipy/interpolate/src/_interpolate.cpp: In function ‘PyObject* loginterp_method(PyObject*, PyObject*, PyObject*)’:
    scipy/interpolate/src/_interpolate.cpp:63:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         static char *kwlist[] = {"x","y","new_x","new_y", NULL};
                                                               ^
    scipy/interpolate/src/_interpolate.cpp:63:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
    scipy/interpolate/src/_interpolate.cpp:63:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
    scipy/interpolate/src/_interpolate.cpp:63:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
    scipy/interpolate/src/_interpolate.cpp: In function ‘PyObject* window_average_method(PyObject*, PyObject*, PyObject*)’:
    scipy/interpolate/src/_interpolate.cpp:113:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         static char *kwlist[] = {"x","y","new_x","new_y", NULL};
                                                               ^
    scipy/interpolate/src/_interpolate.cpp:113:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
    scipy/interpolate/src/_interpolate.cpp:113:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
    scipy/interpolate/src/_interpolate.cpp:113:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
    scipy/interpolate/src/_interpolate.cpp: In function ‘PyObject* block_average_above_method(PyObject*, PyObject*, PyObject*)’:
    scipy/interpolate/src/_interpolate.cpp:164:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         static char *kwlist[] = {"x","y","new_x","new_y", NULL};
                                                               ^
    scipy/interpolate/src/_interpolate.cpp:164:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
    scipy/interpolate/src/_interpolate.cpp:164:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
    scipy/interpolate/src/_interpolate.cpp:164:59: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
    c++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/interpolate/src/_interpolate.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/interpolate/_interpolate.so
    building 'scipy.io.matlab.streams' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/io/matlab/streams.c
    scipy/io/matlab/streams.c:1044:13: warning: ‘__pyx_k__rb’ defined but not used [-Wunused-variable]
     static char __pyx_k__rb[] = "rb";
                 ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/io/matlab/streams.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/io/matlab/streams.so
    building 'scipy.io.matlab.mio_utils' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/io/matlab/mio_utils.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/io/matlab/mio_utils.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/io/matlab/mio_utils.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1629:1: warning: ‘_import_array’ defined but not used [-Wunused-function]
     _import_array(void)
     ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ufuncobject.h:327:0,
                     from scipy/io/matlab/mio_utils.c:315:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__ufunc_api.h:241:1: warning: ‘_import_umath’ defined but not used [-Wunused-function]
     _import_umath(void)
     ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/io/matlab/mio_utils.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/io/matlab/mio_utils.so
    building 'scipy.io.matlab.mio5_utils' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/io/matlab/mio5_utils.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/io/matlab/mio5_utils.c:315:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ufuncobject.h:327:0,
                     from scipy/io/matlab/mio5_utils.c:316:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__ufunc_api.h:241:1: warning: ‘_import_umath’ defined but not used [-Wunused-function]
     _import_umath(void)
     ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/io/matlab/mio5_utils.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/io/matlab/mio5_utils.so
    building 'scipy.lib.blas.fblas' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/blas/fblasmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/blas/fblasmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/blas/fblasmodule.c:154:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: /tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_g77_abi.f
    gfortran:f77: /tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_accelerate.f
    gfortran:f77: build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/blas/fblas-f2pywrappers.f
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/blas/fblasmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_g77_abi.o build/temp.linux-x86_64-2.7/tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_accelerate.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/blas/fblas-f2pywrappers.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/lib/blas/fblas.so
    building 'scipy.lib.blas.cblas' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/blas/cblasmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/blas/cblasmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/blas/cblasmodule.c:233:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/blas/cblasmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/lib/blas/cblas.so
    building 'scipy.lib.lapack.flapack' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/flapackmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/flapackmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/flapackmodule.c: In function ‘f2py_rout_flapack_sgees’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/flapackmodule.c:6162:46: warning: variable ‘sselect_cptr’ set but not used [-Wunused-but-set-variable]
       cb_sselect_in_gees__user__routines_typedef sselect_cptr;
                                                  ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/flapackmodule.c: In function ‘f2py_rout_flapack_dgees’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/flapackmodule.c:6434:46: warning: variable ‘dselect_cptr’ set but not used [-Wunused-but-set-variable]
       cb_dselect_in_gees__user__routines_typedef dselect_cptr;
                                                  ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/flapackmodule.c: In function ‘f2py_rout_flapack_cgees’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/flapackmodule.c:6704:46: warning: variable ‘cselect_cptr’ set but not used [-Wunused-but-set-variable]
       cb_cselect_in_gees__user__routines_typedef cselect_cptr;
                                                  ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/flapackmodule.c: In function ‘f2py_rout_flapack_zgees’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/flapackmodule.c:6975:46: warning: variable ‘zselect_cptr’ set but not used [-Wunused-but-set-variable]
       cb_zselect_in_gees__user__routines_typedef zselect_cptr;
                                                  ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/flapackmodule.c: At top level:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/flapackmodule.c:181:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/flapackmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/lib/lapack/flapack.so
    building 'scipy.lib.lapack.clapack' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_sgesv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:243:7: warning: variable ‘sgesv_return_value’ set but not used [-Wunused-but-set-variable]
       int sgesv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_dgesv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:400:7: warning: variable ‘dgesv_return_value’ set but not used [-Wunused-but-set-variable]
       int dgesv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_cgesv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:557:7: warning: variable ‘cgesv_return_value’ set but not used [-Wunused-but-set-variable]
       int cgesv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_zgesv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:714:7: warning: variable ‘zgesv_return_value’ set but not used [-Wunused-but-set-variable]
       int zgesv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_sposv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:871:7: warning: variable ‘sposv_return_value’ set but not used [-Wunused-but-set-variable]
       int sposv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_dposv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:1021:7: warning: variable ‘dposv_return_value’ set but not used [-Wunused-but-set-variable]
       int dposv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_cposv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:1171:7: warning: variable ‘cposv_return_value’ set but not used [-Wunused-but-set-variable]
       int cposv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_zposv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:1321:7: warning: variable ‘zposv_return_value’ set but not used [-Wunused-but-set-variable]
       int zposv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_spotrf’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:1469:7: warning: variable ‘spotrf_return_value’ set but not used [-Wunused-but-set-variable]
       int spotrf_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_dpotrf’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:1601:7: warning: variable ‘dpotrf_return_value’ set but not used [-Wunused-but-set-variable]
       int dpotrf_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_cpotrf’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:1733:7: warning: variable ‘cpotrf_return_value’ set but not used [-Wunused-but-set-variable]
       int cpotrf_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_zpotrf’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:1865:7: warning: variable ‘zpotrf_return_value’ set but not used [-Wunused-but-set-variable]
       int zpotrf_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_spotrs’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:1997:7: warning: variable ‘spotrs_return_value’ set but not used [-Wunused-but-set-variable]
       int spotrs_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_dpotrs’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:2145:7: warning: variable ‘dpotrs_return_value’ set but not used [-Wunused-but-set-variable]
       int dpotrs_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_cpotrs’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:2293:7: warning: variable ‘cpotrs_return_value’ set but not used [-Wunused-but-set-variable]
       int cpotrs_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_zpotrs’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:2441:7: warning: variable ‘zpotrs_return_value’ set but not used [-Wunused-but-set-variable]
       int zpotrs_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_spotri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:2588:7: warning: variable ‘spotri_return_value’ set but not used [-Wunused-but-set-variable]
       int spotri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_dpotri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:2709:7: warning: variable ‘dpotri_return_value’ set but not used [-Wunused-but-set-variable]
       int dpotri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_cpotri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:2830:7: warning: variable ‘cpotri_return_value’ set but not used [-Wunused-but-set-variable]
       int cpotri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_zpotri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:2951:7: warning: variable ‘zpotri_return_value’ set but not used [-Wunused-but-set-variable]
       int zpotri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_slauum’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:3072:7: warning: variable ‘slauum_return_value’ set but not used [-Wunused-but-set-variable]
       int slauum_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_dlauum’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:3193:7: warning: variable ‘dlauum_return_value’ set but not used [-Wunused-but-set-variable]
       int dlauum_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_clauum’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:3314:7: warning: variable ‘clauum_return_value’ set but not used [-Wunused-but-set-variable]
       int clauum_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_zlauum’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:3435:7: warning: variable ‘zlauum_return_value’ set but not used [-Wunused-but-set-variable]
       int zlauum_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_strtri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:3557:7: warning: variable ‘strtri_return_value’ set but not used [-Wunused-but-set-variable]
       int strtri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_dtrtri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:3689:7: warning: variable ‘dtrtri_return_value’ set but not used [-Wunused-but-set-variable]
       int dtrtri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_ctrtri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:3821:7: warning: variable ‘ctrtri_return_value’ set but not used [-Wunused-but-set-variable]
       int ctrtri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: In function ‘f2py_rout_clapack_ztrtri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:3953:7: warning: variable ‘ztrtri_return_value’ set but not used [-Wunused-but-set-variable]
       int ztrtri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c: At top level:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.c:117:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/clapackmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/lib/lapack/clapack.so
    building 'scipy.lib.lapack.calc_lwork' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/lib/lapack/calc_lworkmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/lib/lapack/calc_lworkmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/lib/lapack/calc_lworkmodule.c:137:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: scipy/lib/lapack/calc_lwork.f
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/lib/lapack/calc_lworkmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/scipy/lib/lapack/calc_lwork.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/lib/lapack/calc_lwork.so
    building 'scipy.linalg._fblas' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_fblasmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_fblasmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_fblasmodule.c:154:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: /tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_g77_abi.f
    gfortran:f77: /tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_accelerate.f
    gfortran:f77: build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_fblas-f2pywrappers.f
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_fblasmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_g77_abi.o build/temp.linux-x86_64-2.7/tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_accelerate.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_fblas-f2pywrappers.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/linalg/_fblas.so
    building 'scipy.linalg._flapack' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:645:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern void F_FUNC(chbevx,CHBEVX)();
     ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:646:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern void F_FUNC(zhbevx,ZHBEVX)();
     ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:653:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern void F_FUNC(sgbtrs,SGBTRS)();
     ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:654:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern void F_FUNC(dgbtrs,DGBTRS)();
     ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:655:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern void F_FUNC(cgbtrs,CGBTRS)();
     ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:656:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern void F_FUNC(zgbtrs,ZGBTRS)();
     ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c: In function ‘f2py_rout__flapack_sgges’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:1577:46: warning: variable ‘sselect_cptr’ set but not used [-Wunused-but-set-variable]
       cb_sselect_in_gges__user__routines_typedef sselect_cptr;
                                                  ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c: In function ‘f2py_rout__flapack_dgges’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:1936:46: warning: variable ‘dselect_cptr’ set but not used [-Wunused-but-set-variable]
       cb_dselect_in_gges__user__routines_typedef dselect_cptr;
                                                  ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c: In function ‘f2py_rout__flapack_cgges’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:2293:46: warning: variable ‘cselect_cptr’ set but not used [-Wunused-but-set-variable]
       cb_cselect_in_gges__user__routines_typedef cselect_cptr;
                                                  ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c: In function ‘f2py_rout__flapack_zgges’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:2651:46: warning: variable ‘zselect_cptr’ set but not used [-Wunused-but-set-variable]
       cb_zselect_in_gges__user__routines_typedef zselect_cptr;
                                                  ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c: In function ‘f2py_rout__flapack_cgeqp3’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:11695:9: warning: passing argument 9 of ‘f2py_func’ from incompatible pointer type [enabled by default]
             (*f2py_func)(&m,&n,a,&m,jpvt,tau,work,&lwork,rwork,&info);
             ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:11695:9: note: expected ‘float *’ but argument is of type ‘struct complex_float *’
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c: In function ‘f2py_rout__flapack_zgeqp3’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:11878:9: warning: passing argument 9 of ‘f2py_func’ from incompatible pointer type [enabled by default]
             (*f2py_func)(&m,&n,a,&m,jpvt,tau,work,&lwork,rwork,&info);
             ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:11878:9: note: expected ‘double *’ but argument is of type ‘struct complex_double *’
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c: In function ‘f2py_rout__flapack_cheev’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:17619:9: warning: passing argument 6 of ‘f2py_func’ from incompatible pointer type [enabled by default]
             (*f2py_func)((compute_v?"V":"N"),(lower?"L":"U"),&n,a,&n,w,work,&lwork,rwork,&info);
             ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:17619:9: note: expected ‘struct complex_float *’ but argument is of type ‘float *’
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c: In function ‘f2py_rout__flapack_zheev’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:17802:9: warning: passing argument 6 of ‘f2py_func’ from incompatible pointer type [enabled by default]
             (*f2py_func)((compute_v?"V":"N"),(lower?"L":"U"),&n,a,&n,w,work,&lwork,rwork,&info);
             ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:17802:9: note: expected ‘struct complex_double *’ but argument is of type ‘double *’
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c: In function ‘f2py_rout__flapack_cgees’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:23026:46: warning: variable ‘cselect_cptr’ set but not used [-Wunused-but-set-variable]
       cb_cselect_in_gees__user__routines_typedef cselect_cptr;
                                                  ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c: In function ‘f2py_rout__flapack_zgees’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:23297:46: warning: variable ‘zselect_cptr’ set but not used [-Wunused-but-set-variable]
       cb_zselect_in_gees__user__routines_typedef zselect_cptr;
                                                  ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c: In function ‘f2py_rout__flapack_sgees’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:23570:46: warning: variable ‘sselect_cptr’ set but not used [-Wunused-but-set-variable]
       cb_sselect_in_gees__user__routines_typedef sselect_cptr;
                                                  ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c: In function ‘f2py_rout__flapack_dgees’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:23842:46: warning: variable ‘dselect_cptr’ set but not used [-Wunused-but-set-variable]
       cb_dselect_in_gees__user__routines_typedef dselect_cptr;
                                                  ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c: At top level:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:27153:28: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
                                void (*f2py_func)()) {
                                ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:27488:28: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
                                void (*f2py_func)()) {
                                ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:28582:28: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
                                void (*f2py_func)()) {
                                ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:28774:28: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
                                void (*f2py_func)()) {
                                ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:28966:28: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
                                void (*f2py_func)()) {
                                ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:29158:28: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
                                void (*f2py_func)()) {
                                ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.c:261:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: /tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_g77_abi.f
    gfortran:f77: /tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_accelerate.f
    gfortran:f77: build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapack-f2pywrappers.f
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapackmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_g77_abi.o build/temp.linux-x86_64-2.7/tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_accelerate.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flapack-f2pywrappers.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/linalg/_flapack.so
    building 'scipy.linalg._cblas' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_cblasmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_cblasmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_cblasmodule.c:233:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_cblasmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/linalg/_cblas.so
    building 'scipy.linalg._clapack' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_sgesv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:243:7: warning: variable ‘sgesv_return_value’ set but not used [-Wunused-but-set-variable]
       int sgesv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_dgesv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:400:7: warning: variable ‘dgesv_return_value’ set but not used [-Wunused-but-set-variable]
       int dgesv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_cgesv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:557:7: warning: variable ‘cgesv_return_value’ set but not used [-Wunused-but-set-variable]
       int cgesv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_zgesv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:714:7: warning: variable ‘zgesv_return_value’ set but not used [-Wunused-but-set-variable]
       int zgesv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_sposv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:871:7: warning: variable ‘sposv_return_value’ set but not used [-Wunused-but-set-variable]
       int sposv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_dposv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:1021:7: warning: variable ‘dposv_return_value’ set but not used [-Wunused-but-set-variable]
       int dposv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_cposv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:1171:7: warning: variable ‘cposv_return_value’ set but not used [-Wunused-but-set-variable]
       int cposv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_zposv’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:1321:7: warning: variable ‘zposv_return_value’ set but not used [-Wunused-but-set-variable]
       int zposv_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_spotrf’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:1469:7: warning: variable ‘spotrf_return_value’ set but not used [-Wunused-but-set-variable]
       int spotrf_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_dpotrf’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:1601:7: warning: variable ‘dpotrf_return_value’ set but not used [-Wunused-but-set-variable]
       int dpotrf_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_cpotrf’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:1733:7: warning: variable ‘cpotrf_return_value’ set but not used [-Wunused-but-set-variable]
       int cpotrf_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_zpotrf’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:1865:7: warning: variable ‘zpotrf_return_value’ set but not used [-Wunused-but-set-variable]
       int zpotrf_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_spotrs’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:1997:7: warning: variable ‘spotrs_return_value’ set but not used [-Wunused-but-set-variable]
       int spotrs_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_dpotrs’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:2145:7: warning: variable ‘dpotrs_return_value’ set but not used [-Wunused-but-set-variable]
       int dpotrs_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_cpotrs’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:2293:7: warning: variable ‘cpotrs_return_value’ set but not used [-Wunused-but-set-variable]
       int cpotrs_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_zpotrs’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:2441:7: warning: variable ‘zpotrs_return_value’ set but not used [-Wunused-but-set-variable]
       int zpotrs_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_spotri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:2588:7: warning: variable ‘spotri_return_value’ set but not used [-Wunused-but-set-variable]
       int spotri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_dpotri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:2709:7: warning: variable ‘dpotri_return_value’ set but not used [-Wunused-but-set-variable]
       int dpotri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_cpotri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:2830:7: warning: variable ‘cpotri_return_value’ set but not used [-Wunused-but-set-variable]
       int cpotri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_zpotri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:2951:7: warning: variable ‘zpotri_return_value’ set but not used [-Wunused-but-set-variable]
       int zpotri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_slauum’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:3072:7: warning: variable ‘slauum_return_value’ set but not used [-Wunused-but-set-variable]
       int slauum_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_dlauum’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:3193:7: warning: variable ‘dlauum_return_value’ set but not used [-Wunused-but-set-variable]
       int dlauum_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_clauum’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:3314:7: warning: variable ‘clauum_return_value’ set but not used [-Wunused-but-set-variable]
       int clauum_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_zlauum’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:3435:7: warning: variable ‘zlauum_return_value’ set but not used [-Wunused-but-set-variable]
       int zlauum_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_strtri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:3557:7: warning: variable ‘strtri_return_value’ set but not used [-Wunused-but-set-variable]
       int strtri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_dtrtri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:3689:7: warning: variable ‘dtrtri_return_value’ set but not used [-Wunused-but-set-variable]
       int dtrtri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_ctrtri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:3821:7: warning: variable ‘ctrtri_return_value’ set but not used [-Wunused-but-set-variable]
       int ctrtri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: In function ‘f2py_rout__clapack_ztrtri’:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:3953:7: warning: variable ‘ztrtri_return_value’ set but not used [-Wunused-but-set-variable]
       int ztrtri_return_value=0;
           ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c: At top level:
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.c:117:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_clapackmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/linalg/_clapack.so
    building 'scipy.linalg._flinalg' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/linalg/_flinalgmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/linalg/_flinalgmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/linalg/_flinalgmodule.c:112:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: scipy/linalg/src/det.f
    gfortran:f77: scipy/linalg/src/lu.f
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_flinalgmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/scipy/linalg/src/det.o build/temp.linux-x86_64-2.7/scipy/linalg/src/lu.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/linalg/_flinalg.so
    building 'scipy.linalg.calc_lwork' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/linalg/calc_lworkmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/linalg/calc_lworkmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/linalg/calc_lworkmodule.c:137:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: scipy/linalg/src/calc_lwork.f
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/calc_lworkmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/scipy/linalg/src/calc_lwork.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/linalg/calc_lwork.so
    building 'scipy.linalg._interpolative' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/linalg/_interpolativemodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/linalg/_interpolativemodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/linalg/_interpolativemodule.c:146:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_sfft.f
    gfortran:f77: scipy/linalg/src/id_dist/src/iddp_aid.f
    scipy/linalg/src/id_dist/src/iddp_aid.f:78.13:
    
            n2 = work(2)
                 1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/iddp_aid.f:239.13:
    
            n2 = w(2)
                 1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/idzp_rid.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_snorm.f
    scipy/linalg/src/id_dist/src/idz_snorm.f:166.18:
    
              enorm = enorm+v(k)*conjg(v(k))
                      1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/iddr_asvd_subr_0.f
    gfortran:f77: scipy/linalg/src/id_dist/src/iddr_asvd_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_0.f
    Warning: Nonconforming tab character in column 1 of line 2
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_1.f
    Warning: Nonconforming tab character in column 1 of line 2
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_2.f
    Warning: Nonconforming tab character in column 1 of line 2
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_3.f
    Warning: Nonconforming tab character in column 1 of line 2
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_4.f
    Warning: Nonconforming tab character in column 1 of line 2
    scipy/linalg/src/id_dist/src/dfft_subr_4.f: In function ‘zffti1’:
    scipy/linalg/src/id_dist/src/dfft_subr_4.f:12:0: warning: ‘ntry’ may be used uninitialized in this function [-Wmaybe-uninitialized]
       103 NTRY = NTRY+2
     ^
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_5.f
    Warning: Nonconforming tab character in column 1 of line 2
    Warning: Nonconforming tab character in column 1 of line 12
    Warning: Nonconforming tab character in column 1 of line 41
    Warning: Nonconforming tab character in column 1 of line 67
    Warning: Nonconforming tab character in column 1 of line 80
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_6.f
    Warning: Nonconforming tab character in column 1 of line 2
    scipy/linalg/src/id_dist/src/dfft_subr_6.f: In function ‘dzfft1’:
    scipy/linalg/src/id_dist/src/dfft_subr_6.f:13:0: warning: ‘ntry’ may be used uninitialized in this function [-Wmaybe-uninitialized]
       103 NTRY = NTRY+2
     ^
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_7.f
    Warning: Nonconforming tab character in column 1 of line 2
    Warning: Nonconforming tab character in column 1 of line 16
    Warning: Nonconforming tab character in column 1 of line 28
    Warning: Nonconforming tab character in column 1 of line 48
    Warning: Nonconforming tab character in column 1 of line 70
    Warning: Nonconforming tab character in column 1 of line 97
    Warning: Nonconforming tab character in column 1 of line 104
    Warning: Nonconforming tab character in column 1 of line 221
    Warning: Nonconforming tab character in column 1 of line 245
    Warning: Nonconforming tab character in column 1 of line 288
    Warning: Nonconforming tab character in column 1 of line 340
    Warning: Nonconforming tab character in column 1 of line 419
    Warning: Nonconforming tab character in column 1 of line 536
    Warning: Nonconforming tab character in column 1 of line 560
    Warning: Nonconforming tab character in column 1 of line 603
    Warning: Nonconforming tab character in column 1 of line 655
    Warning: Nonconforming tab character in column 1 of line 734
    Warning: Nonconforming tab character in column 1 of line 762
    Warning: Nonconforming tab character in column 1 of line 800
    Warning: Nonconforming tab character in column 1 of line 858
    Warning: Nonconforming tab character in column 1 of line 925
    Warning: Nonconforming tab character in column 1 of line 1086
    Warning: Nonconforming tab character in column 1 of line 1114
    Warning: Nonconforming tab character in column 1 of line 1150
    Warning: Nonconforming tab character in column 1 of line 1204
    Warning: Nonconforming tab character in column 1 of line 1267
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_8.f
    Warning: Nonconforming tab character in column 1 of line 2
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_9.f
    Warning: Nonconforming tab character in column 1 of line 2
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_10.f
    Warning: Nonconforming tab character in column 1 of line 2
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_11.f
    Warning: Nonconforming tab character in column 1 of line 2
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_12.f
    Warning: Nonconforming tab character in column 1 of line 2
    scipy/linalg/src/id_dist/src/dfft_subr_12.f: In function ‘dffti1’:
    scipy/linalg/src/id_dist/src/dfft_subr_12.f:12:0: warning: ‘ntry’ may be used uninitialized in this function [-Wmaybe-uninitialized]
       103 NTRY = NTRY+2
     ^
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_13.f
    Warning: Nonconforming tab character in column 1 of line 2
    Warning: Nonconforming tab character in column 1 of line 9
    Warning: Nonconforming tab character in column 1 of line 28
    Warning: Nonconforming tab character in column 1 of line 45
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_14.f
    Warning: Nonconforming tab character in column 1 of line 2
    gfortran:f77: scipy/linalg/src/id_dist/src/dfft_subr_15.f
    Warning: Nonconforming tab character in column 1 of line 2
    Warning: Nonconforming tab character in column 1 of line 13
    gfortran:f77: scipy/linalg/src/id_dist/src/iddr_rid.f
    gfortran:f77: scipy/linalg/src/id_dist/src/iddr_rsvd_subr_0.f
    gfortran:f77: scipy/linalg/src/id_dist/src/iddr_rsvd_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_id.f
    scipy/linalg/src/id_dist/src/idd_id.f:106.20:
    
                iswap = rnorms(k)
                        1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/idd_id.f:114.20:
    
              list(k) = rnorms(k)
                        1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/idd_id.f:218.20:
    
                iswap = rnorms(k)
                        1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/idd_id.f:226.20:
    
              list(k) = rnorms(k)
                        1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/idzr_aid.f
    scipy/linalg/src/id_dist/src/idzr_aid.f:105.12:
    
            l = w(1)
                1
    Warning: Possible change of value in conversion from COMPLEX(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/idzr_aid.f:106.13:
    
            n2 = w(2)
                 1
    Warning: Possible change of value in conversion from COMPLEX(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/iddp_asvd_subr_0.f
    gfortran:f77: scipy/linalg/src/id_dist/src/iddp_asvd_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idzp_rsvd_subr_0.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idzp_rsvd_subr_1.f
    scipy/linalg/src/id_dist/src/idzp_rsvd_subr_1.f:1.41:
    
            subroutine idzp_rsvd0(m,n,matveca,p1t,p2t,p3t,p4t,
                                             1
    Warning: Unused dummy argument 'matveca' at (1)
    scipy/linalg/src/id_dist/src/idzp_rsvd_subr_1.f:1.45:
    
            subroutine idzp_rsvd0(m,n,matveca,p1t,p2t,p3t,p4t,
                                                 1
    Warning: Unused dummy argument 'p1t' at (1)
    scipy/linalg/src/id_dist/src/idzp_rsvd_subr_1.f:1.49:
    
            subroutine idzp_rsvd0(m,n,matveca,p1t,p2t,p3t,p4t,
                                                     1
    Warning: Unused dummy argument 'p2t' at (1)
    scipy/linalg/src/id_dist/src/idzp_rsvd_subr_1.f:1.53:
    
            subroutine idzp_rsvd0(m,n,matveca,p1t,p2t,p3t,p4t,
                                                         1
    Warning: Unused dummy argument 'p3t' at (1)
    scipy/linalg/src/id_dist/src/idzp_rsvd_subr_1.f:1.57:
    
            subroutine idzp_rsvd0(m,n,matveca,p1t,p2t,p3t,p4t,
                                                             1
    Warning: Unused dummy argument 'p4t' at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/idzp_rsvd_subr_2.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_svd_subr_0.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_svd_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_svd_subr_2.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_id.f
    scipy/linalg/src/id_dist/src/idz_id.f:107.20:
    
                iswap = rnorms(k)
                        1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/idz_id.f:115.20:
    
              list(k) = rnorms(k)
                        1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/idz_id.f:124.24:
    
                rnorms(k) = a(k,k)
                            1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idz_id.f:373.21:
    
                rnumer = a(j,krank+k)*conjg(a(j,krank+k))
                         1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idz_id.f:374.21:
    
                rdenom = a(j,j)*conjg(a(j,j))
                         1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idz_id.f:220.20:
    
                iswap = rnorms(k)
                        1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/idz_id.f:228.20:
    
              list(k) = rnorms(k)
                        1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/idz_id.f:237.22:
    
              rnorms(k) = a(k,k)
                          1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/idzp_aid_subr_0.f
    scipy/linalg/src/id_dist/src/idzp_aid_subr_0.f:63.13:
    
            n2 = work(2)
                 1
    Warning: Possible change of value in conversion from COMPLEX(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/idzp_aid_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idzp_aid_subr_2.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idzp_aid_subr_3.f
    scipy/linalg/src/id_dist/src/idzp_aid_subr_3.f:47.13:
    
            n2 = w(2)
                 1
    Warning: Possible change of value in conversion from COMPLEX(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/idzp_aid_subr_4.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idzp_aid_subr_5.f
    gfortran:f77: scipy/linalg/src/id_dist/src/iddp_rsvd_subr_0.f
    gfortran:f77: scipy/linalg/src/id_dist/src/iddp_rsvd_subr_1.f
    scipy/linalg/src/id_dist/src/iddp_rsvd_subr_1.f:1.41:
    
            subroutine iddp_rsvd0(m,n,matvect,p1t,p2t,p3t,p4t,
                                             1
    Warning: Unused dummy argument 'matvect' at (1)
    scipy/linalg/src/id_dist/src/iddp_rsvd_subr_1.f:1.45:
    
            subroutine iddp_rsvd0(m,n,matvect,p1t,p2t,p3t,p4t,
                                                 1
    Warning: Unused dummy argument 'p1t' at (1)
    scipy/linalg/src/id_dist/src/iddp_rsvd_subr_1.f:1.49:
    
            subroutine iddp_rsvd0(m,n,matvect,p1t,p2t,p3t,p4t,
                                                     1
    Warning: Unused dummy argument 'p2t' at (1)
    scipy/linalg/src/id_dist/src/iddp_rsvd_subr_1.f:1.53:
    
            subroutine iddp_rsvd0(m,n,matvect,p1t,p2t,p3t,p4t,
                                                         1
    Warning: Unused dummy argument 'p3t' at (1)
    scipy/linalg/src/id_dist/src/iddp_rsvd_subr_1.f:1.57:
    
            subroutine iddp_rsvd0(m,n,matvect,p1t,p2t,p3t,p4t,
                                                             1
    Warning: Unused dummy argument 'p4t' at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/iddr_aid.f
    scipy/linalg/src/id_dist/src/iddr_aid.f:104.12:
    
            l = w(1)
                1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/iddr_aid.f:105.13:
    
            n2 = w(2)
                 1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_qrpiv.f
    gfortran:f77: scipy/linalg/src/id_dist/src/iddp_rid.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_id2svd_subr_0.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_id2svd_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_id2svd_subr_2.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_house.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idzr_asvd_subr_0.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idzr_asvd_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_svd_subr_0.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_svd_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_svd_subr_2.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_svd_subr_3.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_sfft.f
    scipy/linalg/src/id_dist/src/idd_sfft.f:75.37:
    
            if(l .eq. 1) call idd_sffti1(ind,n,wsave)
                                         1
    Warning: Rank mismatch in argument 'ind' at (1) (scalar and rank-1)
    scipy/linalg/src/id_dist/src/idd_sfft.f:233.36:
    
            if(l .eq. 1) call idd_sfft1(ind,n,v,wsave)
                                        1
    Warning: Rank mismatch in argument 'ind' at (1) (scalar and rank-1)
    scipy/linalg/src/id_dist/src/idd_sfft.f:386.23:
    
                v(2*i-1) = sum
                           1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idd_sfft.f:387.21:
    
                v(2*i) = -ci*sum
                         1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idd_sfft.f:406.25:
    
                  v(2*i-1) = sum
                             1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idd_sfft.f:407.23:
    
                  v(2*i) = -ci*sum
                           1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idd_sfft.f:419.23:
    
                    rsum = rsum + wsave(iii+m*(nblock/2)+k)
                           1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idd_sfft.f:428.23:
    
                    rsum = rsum + wsave(iii+m*(nblock/2)+2*k-1)
                           1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idd_sfft.f:429.23:
    
                    rsum = rsum - wsave(iii+m*(nblock/2)+2*k)
                           1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idd_sfft.f: In function ‘idd_sfft1’:
    scipy/linalg/src/id_dist/src/idd_sfft.f:305:0: warning: ‘sumr’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             v(2*ind-1) = sumr
     ^
    scipy/linalg/src/id_dist/src/idd_sfft.f:306:0: warning: ‘sumi’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             v(2*ind) = sumi
     ^
    gfortran:f77: scipy/linalg/src/id_dist/src/prini.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idzr_rid.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_id2svd_subr_0.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_id2svd_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_id2svd_subr_2.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_snorm.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_house.f
    scipy/linalg/src/id_dist/src/idz_house.f:73.18:
    
                sum = sum+vn(k)*conjg(vn(k))
                      1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idz_house.f:169.16:
    
              sum = sum+x(k)*conjg(x(k))
                    1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idz_house.f:189.14:
    
            rss = x1*conjg(x1) + sum
                  1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idz_house.f:199.15:
    
            test = conjg(phase) * x1
                   1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idz_house.f:234.15:
    
            scal = 2*v1*conjg(v1) / (v1*conjg(v1)+sum)
                   1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idz_house.f:237.14:
    
            rss = phase*rss
                  1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/idzp_asvd_subr_0.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idzp_asvd_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idzp_asvd_subr_2.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_qrpiv.f
    scipy/linalg/src/id_dist/src/idz_qrpiv.f:491.20:
    
                ss(k) = ss(k)+a(j,k)*conjg(a(j,k))
                        1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idz_qrpiv.f:573.22:
    
                  ss(k) = ss(k)-a(krank,k)*conjg(a(krank,k))
                          1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idz_qrpiv.f:621.28:
    
                        ss(k) = ss(k)+a(j,k)*conjg(a(j,k))
                                1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idz_qrpiv.f:749.20:
    
                ss(k) = ss(k)+a(j,k)*conjg(a(j,k))
                        1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idz_qrpiv.f:828.22:
    
                  ss(k) = ss(k)-a(loop,k)*conjg(a(loop,k))
                          1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    scipy/linalg/src/id_dist/src/idz_qrpiv.f:876.28:
    
                        ss(k) = ss(k)+a(j,k)*conjg(a(j,k))
                                1
    Warning: Possible change of value in conversion from COMPLEX(8) to REAL(8) at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rand_subr_0.f
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rand_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rand_subr_2.f
    scipy/linalg/src/id_dist/src/id_rand_subr_2.f:31.14:
    
              j = m*r+1
                  1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_frm_subr_0.f
    scipy/linalg/src/id_dist/src/idd_frm_subr_0.f:37.13:
    
            iw = w(3+m+n)
                 1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/idd_frm_subr_0.f:104.13:
    
            l2 = w(3)
                 1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/idd_frm_subr_0.f:110.13:
    
            iw = w(4+m+l+l2)
                 1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_frm_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_frm_subr_2.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_frm_subr_3.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_frm_subr_4.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_frm_subr_5.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idd_frm_subr_6.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idzr_rsvd_subr_0.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idzr_rsvd_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_frm_subr_0.f
    scipy/linalg/src/id_dist/src/idz_frm_subr_0.f:37.13:
    
            iw = w(3+m+n)
                 1
    Warning: Possible change of value in conversion from COMPLEX(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/idz_frm_subr_0.f:104.13:
    
            iw = w(4+m+l)
                 1
    Warning: Possible change of value in conversion from COMPLEX(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_frm_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_frm_subr_2.f
    gfortran:f77: scipy/linalg/src/id_dist/src/idz_frm_subr_3.f
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:145.17:
    
            ialbetas=w(1)
                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:146.13:
    
            iixs=w(2)
                 1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:147.15:
    
            nsteps=w(3)
                   1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:148.12:
    
            iww=w(4)
                1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:149.10:
    
            n=w(5)
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:185.17:
    
            ialbetas=w(1)
                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:186.13:
    
            iixs=w(2)
                 1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:187.15:
    
            nsteps=w(3)
                   1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:188.12:
    
            iww=w(4)
                1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:189.10:
    
            n=w(5)
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:223.17:
    
            ialbetas=w(1)
                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:224.13:
    
            iixs=w(2)
                 1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:225.15:
    
            nsteps=w(3)
                   1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:226.12:
    
            iww=w(4)
                1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:227.10:
    
            n=w(5)
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:228.16:
    
            igammas=w(6)
                    1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:265.17:
    
            ialbetas=w(1)
                     1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:266.13:
    
            iixs=w(2)
                 1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:267.15:
    
            nsteps=w(3)
                   1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:268.12:
    
            iww=w(4)
                1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:269.10:
    
            n=w(5)
              1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/linalg/src/id_dist/src/id_rtrans_subr_0.f:270.16:
    
            igammas=w(6)
                    1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rtrans_subr_1.f
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rtrans_subr_2.f
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rtrans_subr_3.f
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rtrans_subr_4.f
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rtrans_subr_5.f
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rtrans_subr_6.f
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rtrans_subr_7.f
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rtrans_subr_8.f
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rtrans_subr_9.f
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rtrans_subr_10.f
    gfortran:f77: scipy/linalg/src/id_dist/src/id_rtrans_subr_11.f
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/linalg/_interpolativemodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_sfft.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/iddp_aid.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzp_rid.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_snorm.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/iddr_asvd_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/iddr_asvd_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_2.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_3.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_4.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_5.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_6.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_7.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_8.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_9.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_10.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_11.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_12.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_13.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_14.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/dfft_subr_15.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/iddr_rid.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/iddr_rsvd_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/iddr_rsvd_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_id.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzr_aid.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/iddp_asvd_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/iddp_asvd_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzp_rsvd_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzp_rsvd_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzp_rsvd_subr_2.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_svd_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_svd_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_svd_subr_2.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_id.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzp_aid_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzp_aid_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzp_aid_subr_2.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzp_aid_subr_3.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzp_aid_subr_4.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzp_aid_subr_5.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/iddp_rsvd_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/iddp_rsvd_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/iddr_aid.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_qrpiv.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/iddp_rid.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_id2svd_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_id2svd_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_id2svd_subr_2.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_house.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzr_asvd_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzr_asvd_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_svd_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_svd_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_svd_subr_2.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_svd_subr_3.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_sfft.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/prini.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzr_rid.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_id2svd_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_id2svd_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_id2svd_subr_2.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_snorm.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_house.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzp_asvd_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzp_asvd_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzp_asvd_subr_2.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_qrpiv.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rand_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rand_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rand_subr_2.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_frm_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_frm_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_frm_subr_2.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_frm_subr_3.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_frm_subr_4.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_frm_subr_5.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idd_frm_subr_6.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzr_rsvd_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idzr_rsvd_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_frm_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_frm_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_frm_subr_2.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/idz_frm_subr_3.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rtrans_subr_0.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rtrans_subr_1.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rtrans_subr_2.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rtrans_subr_3.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rtrans_subr_4.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rtrans_subr_5.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rtrans_subr_6.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rtrans_subr_7.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rtrans_subr_8.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rtrans_subr_9.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rtrans_subr_10.o build/temp.linux-x86_64-2.7/scipy/linalg/src/id_dist/src/id_rtrans_subr_11.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/linalg/_interpolative.so
    building 'scipy.odr.__odrpack' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -Iscipy/odr -I/usr/include/atlas -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/odr/__odrpack.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/odr/odrpack.h:2,
                     from scipy/odr/__odrpack.c:12:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    scipy/odr/__odrpack.c: In function ‘odr’:
    scipy/odr/__odrpack.c:1115:13: warning: format ‘%d’ expects argument of type ‘int’, but argument 2 has type ‘npy_intp’ [-Wformat=]
                 printf("%d %d\n", work->dimensions[0], lwork);
                 ^
    scipy/odr/__odrpack.c: In function ‘init__odrpack’:
    scipy/odr/__odrpack.c:1342:19: warning: unused variable ‘d’ [-Wunused-variable]
         PyObject *m, *d;
                       ^
    scipy/odr/__odrpack.c:1342:15: warning: variable ‘m’ set but not used [-Wunused-but-set-variable]
         PyObject *m, *d;
                   ^
    scipy/odr/__odrpack.c: At top level:
    scipy/odr/__odrpack.c:1256:13: warning: ‘check_args’ defined but not used [-Wunused-function]
     static void check_args(int n, int m, int np, int nq,
                 ^
    In file included from /usr/include/python2.7/Python.h:80:0,
                     from scipy/odr/odrpack.h:1,
                     from scipy/odr/__odrpack.c:12:
    scipy/odr/__odrpack.c: In function ‘fcn_callback’:
    /usr/include/python2.7/object.h:823:32: warning: ‘result’ may be used uninitialized in this function [-Wmaybe-uninitialized]
     #define Py_XDECREF(op) do { if ((op) == NULL) ; else Py_DECREF(op); } while (0)
                                    ^
    scipy/odr/__odrpack.c:47:13: note: ‘result’ was declared here
       PyObject *result;
                 ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/scipy/odr/__odrpack.o -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -lodrpack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/odr/__odrpack.so
    building 'scipy.optimize._minpack' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/optimize/_minpackmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/optimize/minpack.h:38,
                     from scipy/optimize/_minpackmodule.c:5:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from scipy/optimize/_minpackmodule.c:5:0:
    scipy/optimize/__minpack.h: In function ‘minpack_hybrd’:
    scipy/optimize/minpack.h:44:64: warning: unused variable ‘store_multipack_globals3’ [-Wunused-variable]
     #define STORE_VARS() PyObject *store_multipack_globals[4]; int store_multipack_globals3;
                                                                    ^
    scipy/optimize/__minpack.h:245:3: note: in expansion of macro ‘STORE_VARS’
       STORE_VARS();    /* Define storage variables for global variables. */
       ^
    scipy/optimize/__minpack.h: In function ‘minpack_lmdif’:
    scipy/optimize/minpack.h:44:64: warning: unused variable ‘store_multipack_globals3’ [-Wunused-variable]
     #define STORE_VARS() PyObject *store_multipack_globals[4]; int store_multipack_globals3;
                                                                    ^
    scipy/optimize/__minpack.h:452:3: note: in expansion of macro ‘STORE_VARS’
       STORE_VARS();
       ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/scipy/optimize/_minpackmodule.o -Lbuild/temp.linux-x86_64-2.7 -lminpack -lgfortran -o build/lib.linux-x86_64-2.7/scipy/optimize/_minpack.so
    building 'scipy.optimize._zeros' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/optimize/zeros.c
    In file included from scipy/optimize/zeros.c:24:0:
    scipy/optimize/Zeros/zeros.h:16:15: warning: ‘dminarg1’ defined but not used [-Wunused-variable]
     static double dminarg1,dminarg2;
                   ^
    scipy/optimize/Zeros/zeros.h:16:24: warning: ‘dminarg2’ defined but not used [-Wunused-variable]
     static double dminarg1,dminarg2;
                            ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/optimize/zeros.o -Lbuild/temp.linux-x86_64-2.7 -lrootfind -o build/lib.linux-x86_64-2.7/scipy/optimize/_zeros.so
    building 'scipy.optimize._lbfgsb' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/optimize/lbfgsb/_lbfgsbmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/optimize/lbfgsb/_lbfgsbmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/optimize/lbfgsb/_lbfgsbmodule.c:184:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: scipy/optimize/lbfgsb/lbfgsb.f
    scipy/optimize/lbfgsb/lbfgsb.f: In function ‘cauchy’:
    scipy/optimize/lbfgsb/lbfgsb.f:1455:0: warning: ‘tu’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                 xupper = nbd(i) .ge. 2 .and. tu .le. zero
     ^
    scipy/optimize/lbfgsb/lbfgsb.f:1484:0: warning: ‘tl’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                    t(nbreak) = tl/(-neggi)
     ^
    gfortran:f77: scipy/optimize/lbfgsb/linpack.f
    gfortran:f77: scipy/optimize/lbfgsb/timer.f
    scipy/optimize/lbfgsb/timer.f:4.15:
    
          real temp
                   1
    Warning: Unused variable 'temp' declared at (1)
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/optimize/lbfgsb/_lbfgsbmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/scipy/optimize/lbfgsb/lbfgsb.o build/temp.linux-x86_64-2.7/scipy/optimize/lbfgsb/linpack.o build/temp.linux-x86_64-2.7/scipy/optimize/lbfgsb/timer.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/optimize/_lbfgsb.so
    building 'scipy.optimize.moduleTNC' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/optimize/tnc/tnc.c
    x86_64-linux-gnu-gcc: scipy/optimize/tnc/moduleTNC.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/optimize/tnc/moduleTNC.c:30:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/optimize/tnc/moduleTNC.o build/temp.linux-x86_64-2.7/scipy/optimize/tnc/tnc.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/optimize/moduleTNC.so
    building 'scipy.optimize._cobyla' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/optimize/cobyla/_cobylamodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/optimize/cobyla/_cobylamodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/optimize/cobyla/_cobylamodule.c: In function ‘cb_calcfc_in__cobyla__user__routines’:
    build/src.linux-x86_64-2.7/scipy/optimize/cobyla/_cobylamodule.c:355:10: warning: unused variable ‘f’ [-Wunused-variable]
       double f=(*f_cb_capi);
              ^
    build/src.linux-x86_64-2.7/scipy/optimize/cobyla/_cobylamodule.c: At top level:
    build/src.linux-x86_64-2.7/scipy/optimize/cobyla/_cobylamodule.c:129:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: scipy/optimize/cobyla/cobyla2.f
    gfortran:f77: scipy/optimize/cobyla/trstlp.f
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/optimize/cobyla/_cobylamodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/scipy/optimize/cobyla/cobyla2.o build/temp.linux-x86_64-2.7/scipy/optimize/cobyla/trstlp.o -Lbuild/temp.linux-x86_64-2.7 -lgfortran -o build/lib.linux-x86_64-2.7/scipy/optimize/_cobyla.so
    building 'scipy.optimize.minpack2' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/optimize/minpack2/minpack2module.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/optimize/minpack2/minpack2module.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/optimize/minpack2/minpack2module.c:136:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: scipy/optimize/minpack2/dcsrch.f
    gfortran:f77: scipy/optimize/minpack2/dcstep.f
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/optimize/minpack2/minpack2module.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/scipy/optimize/minpack2/dcsrch.o build/temp.linux-x86_64-2.7/scipy/optimize/minpack2/dcstep.o -Lbuild/temp.linux-x86_64-2.7 -lgfortran -o build/lib.linux-x86_64-2.7/scipy/optimize/minpack2.so
    building 'scipy.optimize._slsqp' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/optimize/slsqp/_slsqpmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/optimize/slsqp/_slsqpmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/optimize/slsqp/_slsqpmodule.c:152:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: scipy/optimize/slsqp/slsqp_optmz.f
    scipy/optimize/slsqp/slsqp_optmz.f:1933.72:
    
       10 assign 30 to next
                                                                            1
    Warning: Deleted feature: ASSIGN statement at (1)
    scipy/optimize/slsqp/slsqp_optmz.f:1938.19:
    
       20    GO TO next,(30, 50, 70, 110)
                       1
    Warning: Deleted feature: Assigned GOTO statement at (1)
    scipy/optimize/slsqp/slsqp_optmz.f:1940.72:
    
          assign 50 to next
                                                                            1
    Warning: Deleted feature: ASSIGN statement at (1)
    scipy/optimize/slsqp/slsqp_optmz.f:1950.72:
    
          assign 70 to next
                                                                            1
    Warning: Deleted feature: ASSIGN statement at (1)
    scipy/optimize/slsqp/slsqp_optmz.f:1956.72:
    
          assign 110 to next
                                                                            1
    Warning: Deleted feature: ASSIGN statement at (1)
    scipy/optimize/slsqp/slsqp_optmz.f:1969.5:
    
      110 IF( ABS(dx(i)) .LE. xmax ) GO TO 115
         1
    Warning: Label 110 at (1) defined but not used
    scipy/optimize/slsqp/slsqp_optmz.f:1964.5:
    
       70 IF( ABS(dx(i)) .GT. cutlo ) GO TO 75
         1
    Warning: Label 70 at (1) defined but not used
    scipy/optimize/slsqp/slsqp_optmz.f:1945.5:
    
       50 IF( dx(i) .EQ. ZERO) GO TO 200
         1
    Warning: Label 50 at (1) defined but not used
    scipy/optimize/slsqp/slsqp_optmz.f:834.49:
    
          CALL hfti (w(ie),me,me,l,w(IF),k,1,t,krank,xnrm,w,w(l+1),jw)
                                                     1
    Warning: Rank mismatch in argument 'rnorm' at (1) (rank-1 and scalar)
    scipy/optimize/slsqp/slsqp_optmz.f: In function ‘ldl’:
    scipy/optimize/slsqp/slsqp_optmz.f:1519:0: warning: ‘tp’ may be used uninitialized in this function [-Wmaybe-uninitialized]
               alpha=tp/t
     ^
    scipy/optimize/slsqp/slsqp_optmz.f: In function ‘dnrm2_’:
    scipy/optimize/slsqp/slsqp_optmz.f:1969:0: warning: ‘xmax’ may be used uninitialized in this function [-Wmaybe-uninitialized]
       110 IF( ABS(dx(i)) .LE. xmax ) GO TO 115
     ^
    scipy/optimize/slsqp/slsqp_optmz.f: In function ‘linmin’:
    scipy/optimize/slsqp/slsqp_optmz.f:1624:0: warning: ‘e’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           IF (ABS(e) .LE. tol1) GOTO 30
     ^
    scipy/optimize/slsqp/slsqp_optmz.f:1649:0: warning: ‘u’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           IF (u - a .LT. tol2) d = SIGN(tol1, m - x)
     ^
    scipy/optimize/slsqp/slsqp_optmz.f:1670:0: warning: ‘fx’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           IF (fu .GT. fx) GOTO 60
     ^
    scipy/optimize/slsqp/slsqp_optmz.f:1671:0: warning: ‘x’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           IF (u .GE. x) a = x
     ^
    scipy/optimize/slsqp/slsqp_optmz.f:1682:0: warning: ‘fw’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           IF (fu .LE. fw .OR. w .EQ. x) GOTO 70
     ^
    scipy/optimize/slsqp/slsqp_optmz.f:1682:0: warning: ‘w’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/optimize/slsqp/slsqp_optmz.f:1683:0: warning: ‘fv’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           IF (fu .LE. fv .OR. v .EQ. x .OR. v .EQ. w) GOTO 80
     ^
    scipy/optimize/slsqp/slsqp_optmz.f:1683:0: warning: ‘v’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    scipy/optimize/slsqp/slsqp_optmz.f:1650:0: warning: ‘b’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           IF (b - u .LT. tol2) d = SIGN(tol1, m - x)
     ^
    scipy/optimize/slsqp/slsqp_optmz.f:1614:0: warning: ‘a’ may be used uninitialized in this function [-Wmaybe-uninitialized]
        20 m = 0.5d0*(a + b)
     ^
    scipy/optimize/slsqp/slsqp_optmz.f: In function ‘nnls’:
    scipy/optimize/slsqp/slsqp_optmz.f:1090:0: warning: ‘izmax’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           INTEGER          i,ii,ip,iter,itmax,iz,izmax,iz1,iz2,j,jj,jz,
     ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/optimize/slsqp/_slsqpmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/scipy/optimize/slsqp/slsqp_optmz.o -Lbuild/temp.linux-x86_64-2.7 -lgfortran -o build/lib.linux-x86_64-2.7/scipy/optimize/_slsqp.so
    building 'scipy.optimize._nnls' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/optimize/nnls/_nnlsmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/optimize/nnls/_nnlsmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/optimize/nnls/_nnlsmodule.c:111:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: scipy/optimize/nnls/nnls.f
    scipy/optimize/nnls/nnls.f:121.44:
    
          CALL H12 (1,NPP1,NPP1+1,M,A(1,J),1,UP,DUMMY,1,1,0)
                                                1
    Warning: Rank mismatch in argument 'c' at (1) (rank-1 and scalar)
    scipy/optimize/nnls/nnls.f: In function ‘nnls’:
    scipy/optimize/nnls/nnls.f:52:0: warning: ‘izmax’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           integer I, II, IP, ITER, ITMAX, IZ, IZ1, IZ2, IZMAX, J, JJ, JZ, L
     ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/optimize/nnls/_nnlsmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/scipy/optimize/nnls/nnls.o -Lbuild/temp.linux-x86_64-2.7 -lgfortran -o build/lib.linux-x86_64-2.7/scipy/optimize/_nnls.so
    building 'scipy.signal.sigtools' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Iscipy/signal -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/signal/sigtoolsmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h:9,
                     from scipy/signal/sigtoolsmodule.c:10:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/signal/lfilter.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h:9,
                     from scipy/signal/lfilter.c.src:8:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h:9,
                     from scipy/signal/lfilter.c.src:8:
    scipy/signal/lfilter.c.src: In function ‘scipy_signal_sigtools_linear_filter’:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1125:23: warning: ‘itzf’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             if (_PyAIT(it)->nd_m1 == 0) { \
                           ^
    scipy/signal/lfilter.c.src:259:43: note: ‘itzf’ was declared here
         PyArrayIterObject *itx, *ity, *itzi, *itzf;
                                               ^
    In file included from /usr/include/python2.7/Python.h:80:0,
                     from scipy/signal/lfilter.c.src:5:
    /usr/include/python2.7/object.h:772:28: warning: ‘itzi’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             --((PyObject*)(op))->ob_refcnt != 0)            \
                                ^
    scipy/signal/lfilter.c.src:259:36: note: ‘itzi’ was declared here
         PyArrayIterObject *itx, *ity, *itzi, *itzf;
                                        ^
    x86_64-linux-gnu-gcc: scipy/signal/medianfilter.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h:9,
                     from scipy/signal/medianfilter.c:6:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: scipy/signal/firfilter.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h:9,
                     from scipy/signal/sigtools.h:11,
                     from scipy/signal/firfilter.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/signal/correlate_nd.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h:9,
                     from scipy/signal/correlate_nd.c.src:8:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/signal/sigtoolsmodule.o build/temp.linux-x86_64-2.7/scipy/signal/firfilter.o build/temp.linux-x86_64-2.7/scipy/signal/medianfilter.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/signal/lfilter.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/signal/correlate_nd.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/signal/sigtools.so
    building 'scipy.signal._spectral' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/signal/_spectral.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/signal/_spectral.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/signal/_spectral.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1629:1: warning: ‘_import_array’ defined but not used [-Wunused-function]
     _import_array(void)
     ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ufuncobject.h:327:0,
                     from scipy/signal/_spectral.c:315:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__ufunc_api.h:241:1: warning: ‘_import_umath’ defined but not used [-Wunused-function]
     _import_umath(void)
     ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/signal/_spectral.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/signal/_spectral.so
    building 'scipy.signal.spline' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/signal/C_bspline_util.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/signal/C_bspline_util.c:8:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: scipy/signal/splinemodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/signal/splinemodule.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: scipy/signal/S_bspline_util.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h:9,
                     from scipy/signal/S_bspline_util.c:8:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: scipy/signal/Z_bspline_util.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/signal/Z_bspline_util.c:8:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: scipy/signal/D_bspline_util.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/signal/D_bspline_util.c:8:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: scipy/signal/bspline_util.c
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/signal/splinemodule.o build/temp.linux-x86_64-2.7/scipy/signal/S_bspline_util.o build/temp.linux-x86_64-2.7/scipy/signal/D_bspline_util.o build/temp.linux-x86_64-2.7/scipy/signal/C_bspline_util.o build/temp.linux-x86_64-2.7/scipy/signal/Z_bspline_util.o build/temp.linux-x86_64-2.7/scipy/signal/bspline_util.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/signal/spline.so
    building 'scipy.sparse.linalg.isolve._iterative' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/_iterativemodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/_iterativemodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/_iterativemodule.c:153:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/STOPTEST2.f
    gfortran:f77: build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/getbreak.f
    gfortran:f77: build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/BiCGREVCOM.f
    gfortran:f77: build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/BiCGSTABREVCOM.f
    gfortran:f77: build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/CGREVCOM.f
    gfortran:f77: build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/CGSREVCOM.f
    gfortran:f77: build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/GMRESREVCOM.f
    build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/GMRESREVCOM.f:2260.37:
    
         $     FUNCTION dzAPPROXRES( I, H, S, GIVENS, LDG )
                                         1
    Warning: Unused dummy argument 'h' at (1)
    build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/GMRESREVCOM.f:1666.38:
    
         $     FUNCTION wscAPPROXRES( I, H, S, GIVENS, LDG )
                                          1
    Warning: Unused dummy argument 'h' at (1)
    build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/GMRESREVCOM.f:1072.36:
    
         $     FUNCTION dAPPROXRES( I, H, S, GIVENS, LDG )
                                        1
    Warning: Unused dummy argument 'h' at (1)
    build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/GMRESREVCOM.f:478.37:
    
         $     FUNCTION wsAPPROXRES( I, H, S, GIVENS, LDG )
                                         1
    Warning: Unused dummy argument 'h' at (1)
    gfortran:f77: build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/QMRREVCOM.f
    gfortran:f77: /tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_g77_abi.f
    gfortran:f77: /tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_accelerate.f
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/_iterativemodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/STOPTEST2.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/getbreak.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/BiCGREVCOM.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/BiCGSTABREVCOM.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/CGREVCOM.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/CGSREVCOM.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/GMRESREVCOM.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/isolve/iterative/QMRREVCOM.o build/temp.linux-x86_64-2.7/tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_g77_abi.o build/temp.linux-x86_64-2.7/tmp/pip_build_root/scipy/scipy/_build_utils/src/wrap_dummy_accelerate.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/sparse/linalg/isolve/_iterative.so
    building 'scipy.sparse.linalg.dsolve._superlu' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -DUSE_VENDOR_BLAS=1 -I/usr/include/atlas -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/_superlu_utils.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/_superluobject.h:18,
                     from scipy/sparse/linalg/dsolve/_superlu_utils.c:8:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/sparse/linalg/dsolve/_superluobject.h:19,
                     from scipy/sparse/linalg/dsolve/_superlu_utils.c:8:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from scipy/sparse/linalg/dsolve/_superlu_utils.c:8:0:
    scipy/sparse/linalg/dsolve/_superluobject.h:130:19: warning: ‘gstrf’ defined but not used [-Wunused-function]
     TYPE_GENERIC_FUNC(gstrf, void);
                       ^
    scipy/sparse/linalg/dsolve/_superluobject.h:71:23: note: in definition of macro ‘TYPE_GENERIC_FUNC’
         static returntype name(int type, name##_ARGS)          \
                           ^
    scipy/sparse/linalg/dsolve/_superluobject.h:131:19: warning: ‘gsitrf’ defined but not used [-Wunused-function]
     TYPE_GENERIC_FUNC(gsitrf, void);
                       ^
    scipy/sparse/linalg/dsolve/_superluobject.h:71:23: note: in definition of macro ‘TYPE_GENERIC_FUNC’
         static returntype name(int type, name##_ARGS)          \
                           ^
    scipy/sparse/linalg/dsolve/_superluobject.h:132:19: warning: ‘gstrs’ defined but not used [-Wunused-function]
     TYPE_GENERIC_FUNC(gstrs, void);
                       ^
    scipy/sparse/linalg/dsolve/_superluobject.h:71:23: note: in definition of macro ‘TYPE_GENERIC_FUNC’
         static returntype name(int type, name##_ARGS)          \
                           ^
    scipy/sparse/linalg/dsolve/_superluobject.h:133:19: warning: ‘gssv’ defined but not used [-Wunused-function]
     TYPE_GENERIC_FUNC(gssv, void);
                       ^
    scipy/sparse/linalg/dsolve/_superluobject.h:71:23: note: in definition of macro ‘TYPE_GENERIC_FUNC’
         static returntype name(int type, name##_ARGS)          \
                           ^
    scipy/sparse/linalg/dsolve/_superluobject.h:134:19: warning: ‘Create_Dense_Matrix’ defined but not used [-Wunused-function]
     TYPE_GENERIC_FUNC(Create_Dense_Matrix, void);
                       ^
    scipy/sparse/linalg/dsolve/_superluobject.h:71:23: note: in definition of macro ‘TYPE_GENERIC_FUNC’
         static returntype name(int type, name##_ARGS)          \
                           ^
    scipy/sparse/linalg/dsolve/_superluobject.h:135:19: warning: ‘Create_CompRow_Matrix’ defined but not used [-Wunused-function]
     TYPE_GENERIC_FUNC(Create_CompRow_Matrix, void);
                       ^
    scipy/sparse/linalg/dsolve/_superluobject.h:71:23: note: in definition of macro ‘TYPE_GENERIC_FUNC’
         static returntype name(int type, name##_ARGS)          \
                           ^
    scipy/sparse/linalg/dsolve/_superluobject.h:136:19: warning: ‘Create_CompCol_Matrix’ defined but not used [-Wunused-function]
     TYPE_GENERIC_FUNC(Create_CompCol_Matrix, void);
                       ^
    scipy/sparse/linalg/dsolve/_superluobject.h:71:23: note: in definition of macro ‘TYPE_GENERIC_FUNC’
         static returntype name(int type, name##_ARGS)          \
                           ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/_superluobject.c
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/_superluobject.h:18,
                     from scipy/sparse/linalg/dsolve/_superluobject.c:13:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/sparse/linalg/dsolve/_superluobject.h:19,
                     from scipy/sparse/linalg/dsolve/_superluobject.c:13:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/sparse/linalg/dsolve/_superluobject.h:19,
                     from scipy/sparse/linalg/dsolve/_superluobject.c:13:
    scipy/sparse/linalg/dsolve/_superluobject.c: In function ‘SciPyLU_getattr’:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1018:9: warning: initialization from incompatible pointer type [enabled by default]
             (*(PyObject * (*)(PyTypeObject *, int, npy_intp *, int, npy_intp *, void *, int, int, PyObject *)) \
             ^
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:129:9: note: in expansion of macro ‘PyArray_New’
             PyArray_New(&PyArray_Type, nd, dims, typenum, NULL, \
             ^
    scipy/sparse/linalg/dsolve/_superluobject.c:149:29: note: in expansion of macro ‘PyArray_SimpleNewFromData’
         PyArrayObject* perm_r = PyArray_SimpleNewFromData(1, (npy_intp*) (&self->n), NPY_INT, (void*)self->perm_r);
                                 ^
    scipy/sparse/linalg/dsolve/_superluobject.c:151:26: warning: assignment from incompatible pointer type [enabled by default]
         PyArray_BASE(perm_r) = self;
                              ^
    scipy/sparse/linalg/dsolve/_superluobject.c:153:5: warning: return from incompatible pointer type [enabled by default]
         return perm_r ;
         ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/sparse/linalg/dsolve/_superluobject.h:19,
                     from scipy/sparse/linalg/dsolve/_superluobject.c:13:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1018:9: warning: initialization from incompatible pointer type [enabled by default]
             (*(PyObject * (*)(PyTypeObject *, int, npy_intp *, int, npy_intp *, void *, int, int, PyObject *)) \
             ^
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:129:9: note: in expansion of macro ‘PyArray_New’
             PyArray_New(&PyArray_Type, nd, dims, typenum, NULL, \
             ^
    scipy/sparse/linalg/dsolve/_superluobject.c:156:29: note: in expansion of macro ‘PyArray_SimpleNewFromData’
         PyArrayObject* perm_c = PyArray_SimpleNewFromData(1, (npy_intp*) (&self->n), NPY_INT, (void*)self->perm_c);
                                 ^
    scipy/sparse/linalg/dsolve/_superluobject.c:158:26: warning: assignment from incompatible pointer type [enabled by default]
         PyArray_BASE(perm_c) = self;
                              ^
    scipy/sparse/linalg/dsolve/_superluobject.c:160:5: warning: return from incompatible pointer type [enabled by default]
         return perm_c ;
         ^
    scipy/sparse/linalg/dsolve/_superluobject.c: In function ‘droprule_one_cvt’:
    scipy/sparse/linalg/dsolve/_superluobject.c:475:10: warning: variable ‘i’ set but not used [-Wunused-but-set-variable]
         long i = -1;                                \
              ^
    scipy/sparse/linalg/dsolve/_superluobject.c:616:5: note: in expansion of macro ‘ENUM_CHECK_INIT’
         ENUM_CHECK_INIT;
         ^
    In file included from scipy/sparse/linalg/dsolve/_superluobject.c:13:0:
    scipy/sparse/linalg/dsolve/_superluobject.c: At top level:
    scipy/sparse/linalg/dsolve/_superluobject.h:133:19: warning: ‘gssv’ defined but not used [-Wunused-function]
     TYPE_GENERIC_FUNC(gssv, void);
                       ^
    scipy/sparse/linalg/dsolve/_superluobject.h:71:23: note: in definition of macro ‘TYPE_GENERIC_FUNC’
         static returntype name(int type, name##_ARGS)          \
                           ^
    scipy/sparse/linalg/dsolve/_superluobject.c: In function ‘droprule_cvt’:
    scipy/sparse/linalg/dsolve/_superluobject.c:682:14: warning: ‘one_value’ may be used uninitialized in this function [-Wmaybe-uninitialized]
             rule |= one_value;
                  ^
    x86_64-linux-gnu-gcc: scipy/sparse/linalg/dsolve/_superlumodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/sparse/linalg/dsolve/_superlumodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_zdefs.h:84:0,
                     from scipy/sparse/linalg/dsolve/_superluobject.h:18,
                     from scipy/sparse/linalg/dsolve/_superlumodule.c:20:
    scipy/sparse/linalg/dsolve/SuperLU/SRC/slu_util.h:349:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern double  SuperLU_timer_ ();
     ^
    scipy/sparse/linalg/dsolve/_superlumodule.c: In function ‘Py_gssv’:
    scipy/sparse/linalg/dsolve/_superlumodule.c:89:9: warning: variable ‘ssv_finished’ set but not used [-Wunused-but-set-variable]
         int ssv_finished = 0;
             ^
    In file included from scipy/sparse/linalg/dsolve/_superlumodule.c:20:0:
    scipy/sparse/linalg/dsolve/_superlumodule.c: At top level:
    scipy/sparse/linalg/dsolve/_superluobject.h:130:19: warning: ‘gstrf’ defined but not used [-Wunused-function]
     TYPE_GENERIC_FUNC(gstrf, void);
                       ^
    scipy/sparse/linalg/dsolve/_superluobject.h:71:23: note: in definition of macro ‘TYPE_GENERIC_FUNC’
         static returntype name(int type, name##_ARGS)          \
                           ^
    scipy/sparse/linalg/dsolve/_superluobject.h:131:19: warning: ‘gsitrf’ defined but not used [-Wunused-function]
     TYPE_GENERIC_FUNC(gsitrf, void);
                       ^
    scipy/sparse/linalg/dsolve/_superluobject.h:71:23: note: in definition of macro ‘TYPE_GENERIC_FUNC’
         static returntype name(int type, name##_ARGS)          \
                           ^
    scipy/sparse/linalg/dsolve/_superluobject.h:132:19: warning: ‘gstrs’ defined but not used [-Wunused-function]
     TYPE_GENERIC_FUNC(gstrs, void);
                       ^
    scipy/sparse/linalg/dsolve/_superluobject.h:71:23: note: in definition of macro ‘TYPE_GENERIC_FUNC’
         static returntype name(int type, name##_ARGS)          \
                           ^
    scipy/sparse/linalg/dsolve/_superluobject.h:134:19: warning: ‘Create_Dense_Matrix’ defined but not used [-Wunused-function]
     TYPE_GENERIC_FUNC(Create_Dense_Matrix, void);
                       ^
    scipy/sparse/linalg/dsolve/_superluobject.h:71:23: note: in definition of macro ‘TYPE_GENERIC_FUNC’
         static returntype name(int type, name##_ARGS)          \
                           ^
    scipy/sparse/linalg/dsolve/_superluobject.h:135:19: warning: ‘Create_CompRow_Matrix’ defined but not used [-Wunused-function]
     TYPE_GENERIC_FUNC(Create_CompRow_Matrix, void);
                       ^
    scipy/sparse/linalg/dsolve/_superluobject.h:71:23: note: in definition of macro ‘TYPE_GENERIC_FUNC’
         static returntype name(int type, name##_ARGS)          \
                           ^
    scipy/sparse/linalg/dsolve/_superluobject.h:136:19: warning: ‘Create_CompCol_Matrix’ defined but not used [-Wunused-function]
     TYPE_GENERIC_FUNC(Create_CompCol_Matrix, void);
                       ^
    scipy/sparse/linalg/dsolve/_superluobject.h:71:23: note: in definition of macro ‘TYPE_GENERIC_FUNC’
         static returntype name(int type, name##_ARGS)          \
                           ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/scipy/sparse/linalg/dsolve/_superlumodule.o build/temp.linux-x86_64-2.7/scipy/sparse/linalg/dsolve/_superlu_utils.o build/temp.linux-x86_64-2.7/scipy/sparse/linalg/dsolve/_superluobject.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -lsuperlu_src -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/sparse/linalg/dsolve/_superlu.so
    building 'scipy.sparse.linalg.dsolve.umfpack.__umfpack' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DSCIPY_UMFPACK_H -DSCIPY_AMD_H -DATLAS_INFO="\"3.10.1\"" -I/usr/include/suitesparse -I/usr/include/atlas -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/sparse/linalg/dsolve/umfpack/_umfpack_wrap.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/scipy/sparse/linalg/dsolve/umfpack/_umfpack_wrap.c:2968:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/dsolve/umfpack/_umfpack_wrap.o -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -lumfpack -lamd -lf77blas -lcblas -latlas -o build/lib.linux-x86_64-2.7/scipy/sparse/linalg/dsolve/umfpack/__umfpack.so
    building 'scipy.sparse.linalg.eigen.arpack._arpack' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/_arpackmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/_arpackmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/_arpackmodule.c:262:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-DATLAS_INFO="\"3.10.1\"" -I/usr/include/atlas -Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/_arpack-f2pywrappers.f
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/_arpackmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/_arpack-f2pywrappers.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -larpack_scipy -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/sparse/linalg/eigen/arpack/_arpack.so
    building 'scipy.sparse.sparsetools._csr' extension
    compiling C++ sources
    C compiler: c++ -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -fPIC
    
    compile options: '-D__STDC_FORMAT_MACROS=1 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    c++: scipy/sparse/sparsetools/csr_wrap.cxx
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:23,
                     from scipy/sparse/sparsetools/py3k.h:23,
                     from scipy/sparse/sparsetools/csr_wrap.cxx:3069:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from scipy/sparse/sparsetools/py3k.h:23:0,
                     from scipy/sparse/sparsetools/csr_wrap.cxx:3069:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h: In function ‘PyObject* npy_PyFile_OpenFile(PyObject*, const char*)’:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:247:60: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         return PyObject_CallFunction(open, "Os", filename, mode);
                                                                ^
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h: In function ‘int npy_PyFile_CloseFile(PyObject*)’:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:255:50: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         ret = PyObject_CallMethod(file, "close", NULL);
                                                      ^
    scipy/sparse/sparsetools/csr_wrap.cxx: In function ‘void init_csr()’:
    scipy/sparse/sparsetools/csr_wrap.cxx:73303:21: warning: variable ‘md’ set but not used [-Wunused-but-set-variable]
       PyObject *m, *d, *md;
                         ^
    scipy/sparse/sparsetools/csr_wrap.cxx: At global scope:
    scipy/sparse/sparsetools/csr_wrap.cxx:3143:12: warning: ‘int type_match(int, int)’ defined but not used [-Wunused-function]
     static int type_match(int actual_type, int desired_type) {
                ^
    scipy/sparse/sparsetools/csr_wrap.cxx:3292:12: warning: ‘int require_dimensions_n(PyArrayObject*, int*, int)’ defined but not used [-Wunused-function]
     static int require_dimensions_n(PyArrayObject* ary, int* exact_dimensions, int n) {
                ^
    c++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/sparse/sparsetools/csr_wrap.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/sparse/sparsetools/_csr.so
    building 'scipy.sparse.sparsetools._csc' extension
    compiling C++ sources
    C compiler: c++ -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -fPIC
    
    compile options: '-D__STDC_FORMAT_MACROS=1 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    c++: scipy/sparse/sparsetools/csc_wrap.cxx
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:23,
                     from scipy/sparse/sparsetools/py3k.h:23,
                     from scipy/sparse/sparsetools/csc_wrap.cxx:3054:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from scipy/sparse/sparsetools/py3k.h:23:0,
                     from scipy/sparse/sparsetools/csc_wrap.cxx:3054:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h: In function ‘PyObject* npy_PyFile_OpenFile(PyObject*, const char*)’:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:247:60: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         return PyObject_CallFunction(open, "Os", filename, mode);
                                                                ^
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h: In function ‘int npy_PyFile_CloseFile(PyObject*)’:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:255:50: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         ret = PyObject_CallMethod(file, "close", NULL);
                                                      ^
    scipy/sparse/sparsetools/csc_wrap.cxx: In function ‘void init_csc()’:
    scipy/sparse/sparsetools/csc_wrap.cxx:54411:21: warning: variable ‘md’ set but not used [-Wunused-but-set-variable]
       PyObject *m, *d, *md;
                         ^
    scipy/sparse/sparsetools/csc_wrap.cxx: At global scope:
    scipy/sparse/sparsetools/csc_wrap.cxx:3128:12: warning: ‘int type_match(int, int)’ defined but not used [-Wunused-function]
     static int type_match(int actual_type, int desired_type) {
                ^
    scipy/sparse/sparsetools/csc_wrap.cxx:3277:12: warning: ‘int require_dimensions_n(PyArrayObject*, int*, int)’ defined but not used [-Wunused-function]
     static int require_dimensions_n(PyArrayObject* ary, int* exact_dimensions, int n) {
                ^
    scipy/sparse/sparsetools/csc_wrap.cxx:3355:18: warning: ‘PyObject* helper_appendToTuple(PyObject*, PyObject*)’ defined but not used [-Wunused-function]
     static PyObject *helper_appendToTuple( PyObject *where, PyObject *what ) {
                      ^
    c++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/sparse/sparsetools/csc_wrap.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/sparse/sparsetools/_csc.so
    building 'scipy.sparse.sparsetools._coo' extension
    compiling C++ sources
    C compiler: c++ -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -fPIC
    
    compile options: '-D__STDC_FORMAT_MACROS=1 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    c++: scipy/sparse/sparsetools/coo_wrap.cxx
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:23,
                     from scipy/sparse/sparsetools/py3k.h:23,
                     from scipy/sparse/sparsetools/coo_wrap.cxx:3054:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from scipy/sparse/sparsetools/py3k.h:23:0,
                     from scipy/sparse/sparsetools/coo_wrap.cxx:3054:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h: In function ‘PyObject* npy_PyFile_OpenFile(PyObject*, const char*)’:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:247:60: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         return PyObject_CallFunction(open, "Os", filename, mode);
                                                                ^
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h: In function ‘int npy_PyFile_CloseFile(PyObject*)’:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:255:50: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         ret = PyObject_CallMethod(file, "close", NULL);
                                                      ^
    scipy/sparse/sparsetools/coo_wrap.cxx: In function ‘void init_coo()’:
    scipy/sparse/sparsetools/coo_wrap.cxx:15105:21: warning: variable ‘md’ set but not used [-Wunused-but-set-variable]
       PyObject *m, *d, *md;
                         ^
    scipy/sparse/sparsetools/coo_wrap.cxx: At global scope:
    scipy/sparse/sparsetools/coo_wrap.cxx:3128:12: warning: ‘int type_match(int, int)’ defined but not used [-Wunused-function]
     static int type_match(int actual_type, int desired_type) {
                ^
    scipy/sparse/sparsetools/coo_wrap.cxx:3277:12: warning: ‘int require_dimensions_n(PyArrayObject*, int*, int)’ defined but not used [-Wunused-function]
     static int require_dimensions_n(PyArrayObject* ary, int* exact_dimensions, int n) {
                ^
    scipy/sparse/sparsetools/coo_wrap.cxx:3355:18: warning: ‘PyObject* helper_appendToTuple(PyObject*, PyObject*)’ defined but not used [-Wunused-function]
     static PyObject *helper_appendToTuple( PyObject *where, PyObject *what ) {
                      ^
    c++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/sparse/sparsetools/coo_wrap.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/sparse/sparsetools/_coo.so
    building 'scipy.sparse.sparsetools._bsr' extension
    compiling C++ sources
    C compiler: c++ -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -fPIC
    
    compile options: '-D__STDC_FORMAT_MACROS=1 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    c++: scipy/sparse/sparsetools/bsr_wrap.cxx
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:23,
                     from scipy/sparse/sparsetools/py3k.h:23,
                     from scipy/sparse/sparsetools/bsr_wrap.cxx:3054:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from scipy/sparse/sparsetools/py3k.h:23:0,
                     from scipy/sparse/sparsetools/bsr_wrap.cxx:3054:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h: In function ‘PyObject* npy_PyFile_OpenFile(PyObject*, const char*)’:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:247:60: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         return PyObject_CallFunction(open, "Os", filename, mode);
                                                                ^
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h: In function ‘int npy_PyFile_CloseFile(PyObject*)’:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:255:50: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         ret = PyObject_CallMethod(file, "close", NULL);
                                                      ^
    scipy/sparse/sparsetools/bsr_wrap.cxx: In function ‘void init_bsr()’:
    scipy/sparse/sparsetools/bsr_wrap.cxx:68015:21: warning: variable ‘md’ set but not used [-Wunused-but-set-variable]
       PyObject *m, *d, *md;
                         ^
    scipy/sparse/sparsetools/bsr_wrap.cxx: At global scope:
    scipy/sparse/sparsetools/bsr_wrap.cxx:3128:12: warning: ‘int type_match(int, int)’ defined but not used [-Wunused-function]
     static int type_match(int actual_type, int desired_type) {
                ^
    scipy/sparse/sparsetools/bsr_wrap.cxx:3277:12: warning: ‘int require_dimensions_n(PyArrayObject*, int*, int)’ defined but not used [-Wunused-function]
     static int require_dimensions_n(PyArrayObject* ary, int* exact_dimensions, int n) {
                ^
    scipy/sparse/sparsetools/bsr_wrap.cxx:3355:18: warning: ‘PyObject* helper_appendToTuple(PyObject*, PyObject*)’ defined but not used [-Wunused-function]
     static PyObject *helper_appendToTuple( PyObject *where, PyObject *what ) {
                      ^
    c++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/sparse/sparsetools/bsr_wrap.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/sparse/sparsetools/_bsr.so
    building 'scipy.sparse.sparsetools._dia' extension
    compiling C++ sources
    C compiler: c++ -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -fPIC
    
    compile options: '-D__STDC_FORMAT_MACROS=1 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    c++: scipy/sparse/sparsetools/dia_wrap.cxx
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:23,
                     from scipy/sparse/sparsetools/py3k.h:23,
                     from scipy/sparse/sparsetools/dia_wrap.cxx:3054:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from scipy/sparse/sparsetools/py3k.h:23:0,
                     from scipy/sparse/sparsetools/dia_wrap.cxx:3054:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h: In function ‘PyObject* npy_PyFile_OpenFile(PyObject*, const char*)’:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:247:60: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         return PyObject_CallFunction(open, "Os", filename, mode);
                                                                ^
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h: In function ‘int npy_PyFile_CloseFile(PyObject*)’:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:255:50: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         ret = PyObject_CallMethod(file, "close", NULL);
                                                      ^
    scipy/sparse/sparsetools/dia_wrap.cxx: In function ‘void init_dia()’:
    scipy/sparse/sparsetools/dia_wrap.cxx:6809:21: warning: variable ‘md’ set but not used [-Wunused-but-set-variable]
       PyObject *m, *d, *md;
                         ^
    scipy/sparse/sparsetools/dia_wrap.cxx: At global scope:
    scipy/sparse/sparsetools/dia_wrap.cxx:3128:12: warning: ‘int type_match(int, int)’ defined but not used [-Wunused-function]
     static int type_match(int actual_type, int desired_type) {
                ^
    scipy/sparse/sparsetools/dia_wrap.cxx:3277:12: warning: ‘int require_dimensions_n(PyArrayObject*, int*, int)’ defined but not used [-Wunused-function]
     static int require_dimensions_n(PyArrayObject* ary, int* exact_dimensions, int n) {
                ^
    scipy/sparse/sparsetools/dia_wrap.cxx:3355:18: warning: ‘PyObject* helper_appendToTuple(PyObject*, PyObject*)’ defined but not used [-Wunused-function]
     static PyObject *helper_appendToTuple( PyObject *where, PyObject *what ) {
                      ^
    c++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/sparse/sparsetools/dia_wrap.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/sparse/sparsetools/_dia.so
    building 'scipy.sparse.sparsetools._csgraph' extension
    compiling C++ sources
    C compiler: c++ -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -fPIC
    
    compile options: '-D__STDC_FORMAT_MACROS=1 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    c++: scipy/sparse/sparsetools/csgraph_wrap.cxx
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:23,
                     from scipy/sparse/sparsetools/py3k.h:23,
                     from scipy/sparse/sparsetools/csgraph_wrap.cxx:3054:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from scipy/sparse/sparsetools/py3k.h:23:0,
                     from scipy/sparse/sparsetools/csgraph_wrap.cxx:3054:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h: In function ‘PyObject* npy_PyFile_OpenFile(PyObject*, const char*)’:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:247:60: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         return PyObject_CallFunction(open, "Os", filename, mode);
                                                                ^
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h: In function ‘int npy_PyFile_CloseFile(PyObject*)’:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_3kcompat.h:255:50: warning: deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]
         ret = PyObject_CallMethod(file, "close", NULL);
                                                      ^
    scipy/sparse/sparsetools/csgraph_wrap.cxx: In function ‘void init_csgraph()’:
    scipy/sparse/sparsetools/csgraph_wrap.cxx:4206:21: warning: variable ‘md’ set but not used [-Wunused-but-set-variable]
       PyObject *m, *d, *md;
                         ^
    scipy/sparse/sparsetools/csgraph_wrap.cxx: At global scope:
    scipy/sparse/sparsetools/csgraph_wrap.cxx:3128:12: warning: ‘int type_match(int, int)’ defined but not used [-Wunused-function]
     static int type_match(int actual_type, int desired_type) {
                ^
    scipy/sparse/sparsetools/csgraph_wrap.cxx:3277:12: warning: ‘int require_dimensions_n(PyArrayObject*, int*, int)’ defined but not used [-Wunused-function]
     static int require_dimensions_n(PyArrayObject* ary, int* exact_dimensions, int n) {
                ^
    scipy/sparse/sparsetools/csgraph_wrap.cxx:3355:18: warning: ‘PyObject* helper_appendToTuple(PyObject*, PyObject*)’ defined but not used [-Wunused-function]
     static PyObject *helper_appendToTuple( PyObject *where, PyObject *what ) {
                      ^
    c++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/sparse/sparsetools/csgraph_wrap.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/sparse/sparsetools/_csgraph.so
    building 'scipy.sparse.csgraph._shortest_path' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/sparse/csgraph/_shortest_path.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/sparse/csgraph/_shortest_path.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/sparse/csgraph/_shortest_path.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1629:1: warning: ‘_import_array’ defined but not used [-Wunused-function]
     _import_array(void)
     ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ufuncobject.h:327:0,
                     from scipy/sparse/csgraph/_shortest_path.c:315:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__ufunc_api.h:241:1: warning: ‘_import_umath’ defined but not used [-Wunused-function]
     _import_umath(void)
     ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/sparse/csgraph/_shortest_path.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/sparse/csgraph/_shortest_path.so
    building 'scipy.sparse.csgraph._traversal' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/sparse/csgraph/_traversal.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/sparse/csgraph/_traversal.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/sparse/csgraph/_traversal.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1629:1: warning: ‘_import_array’ defined but not used [-Wunused-function]
     _import_array(void)
     ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ufuncobject.h:327:0,
                     from scipy/sparse/csgraph/_traversal.c:315:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__ufunc_api.h:241:1: warning: ‘_import_umath’ defined but not used [-Wunused-function]
     _import_umath(void)
     ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/sparse/csgraph/_traversal.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/sparse/csgraph/_traversal.so
    building 'scipy.sparse.csgraph._min_spanning_tree' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/sparse/csgraph/_min_spanning_tree.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/sparse/csgraph/_min_spanning_tree.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/sparse/csgraph/_min_spanning_tree.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1629:1: warning: ‘_import_array’ defined but not used [-Wunused-function]
     _import_array(void)
     ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ufuncobject.h:327:0,
                     from scipy/sparse/csgraph/_min_spanning_tree.c:315:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__ufunc_api.h:241:1: warning: ‘_import_umath’ defined but not used [-Wunused-function]
     _import_umath(void)
     ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/sparse/csgraph/_min_spanning_tree.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/sparse/csgraph/_min_spanning_tree.so
    building 'scipy.sparse.csgraph._tools' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/sparse/csgraph/_tools.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/sparse/csgraph/_tools.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/sparse/csgraph/_tools.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1629:1: warning: ‘_import_array’ defined but not used [-Wunused-function]
     _import_array(void)
     ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ufuncobject.h:327:0,
                     from scipy/sparse/csgraph/_tools.c:315:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__ufunc_api.h:241:1: warning: ‘_import_umath’ defined but not used [-Wunused-function]
     _import_umath(void)
     ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/sparse/csgraph/_tools.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/sparse/csgraph/_tools.so
    building 'scipy.spatial.qhull' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-DATLAS_INFO="\"3.10.1\"" -Dqh_QHpointer=1 -I/usr/include/atlas -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/user.c
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/stat.c
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/qset.c
    scipy/spatial/qhull/src/qset.c: In function ‘qh_setfree’:
    scipy/spatial/qhull/src/qset.c:717:10: warning: variable ‘freelistp’ set but not used [-Wunused-but-set-variable]
       void **freelistp;  /* used !qh_NOmem */
              ^
    scipy/spatial/qhull/src/qset.c: In function ‘qh_setnew’:
    scipy/spatial/qhull/src/qset.c:927:10: warning: variable ‘freelistp’ set but not used [-Wunused-but-set-variable]
       void **freelistp; /* used !qh_NOmem */
              ^
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/mem.c
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/random.c
    x86_64-linux-gnu-gcc: scipy/spatial/qhull.c
    In file included from /usr/include/python2.7/numpy/ndarraytypes.h:1728:0,
                     from /usr/include/python2.7/numpy/ndarrayobject.h:17,
                     from scipy/spatial/qhull.c:311:
    /usr/include/python2.7/numpy/npy_deprecated_api.h:11:2: warning: #warning "Using deprecated NumPy API, disable it by #defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by #defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
      ^
    scipy/spatial/qhull.c: In function ‘__pyx_f_5scipy_7spatial_5qhull_6_Qhull__get_voronoi_diagram’:
    scipy/spatial/qhull.c:7405:3: warning: passing argument 2 of ‘qh_eachvoronoi_all’ from incompatible pointer type [enabled by default]
       qh_eachvoronoi_all(((void *)__pyx_v_self), (&__pyx_f_5scipy_7spatial_5qhull__visit_voronoi), qh_qh->UPPERdelaunay, qh_RIDGEall, 1);
       ^
    In file included from scipy/spatial/qhull.c:321:0:
    scipy/spatial/qhull/src/io.h:94:9: note: expected ‘printvridgeT’ but argument is of type ‘void (*)(void *, struct vertexT *, struct vertexT *, struct setT *, unsigned int)’
     int     qh_eachvoronoi_all(FILE *fp, printvridgeT printvridge, boolT isUpper, qh_RIDGE innerouter, boolT inorder);
             ^
    scipy/spatial/qhull.c: In function ‘__pyx_pf_5scipy_7spatial_5qhull__get_barycentric_transforms’:
    scipy/spatial/qhull.c:10268:11: warning: implicit declaration of function ‘dgetrf_’ [-Wimplicit-function-declaration]
               qh_dgetrf((&__pyx_v_n), (&__pyx_v_n), ((double *)__pyx_v_T->data), (&__pyx_v_lda), __pyx_v_ipiv, (&__pyx_v_info));
               ^
    scipy/spatial/qhull.c:10287:13: warning: implicit declaration of function ‘dgecon_’ [-Wimplicit-function-declaration]
                 qh_dgecon(__pyx_k__1, (&__pyx_v_n), ((double *)__pyx_v_T->data), (&__pyx_v_lda), (&__pyx_v_anorm), (&__pyx_v_rcond), __pyx_v_work, __pyx_v_iwork, (&__pyx_v_info));
                 ^
    scipy/spatial/qhull.c:10331:13: warning: implicit declaration of function ‘dgetrs_’ [-Wimplicit-function-declaration]
                 qh_dgetrs(__pyx_k__N, (&__pyx_v_n), (&__pyx_v_nrhs), ((double *)__pyx_v_T->data), (&__pyx_v_lda), __pyx_v_ipiv, (((double *)__pyx_v_Tinvs->data) + ((__pyx_v_ndim * (__pyx_v_ndim + 1)) * __pyx_v_isimplex)), (&__pyx_v_ldb), (&__pyx_v_info));
                 ^
    In file included from /usr/include/python2.7/numpy/ndarrayobject.h:26:0,
                     from scipy/spatial/qhull.c:311:
    scipy/spatial/qhull.c: At top level:
    /usr/include/python2.7/numpy/__multiarray_api.h:1594:1: warning: ‘_import_array’ defined but not used [-Wunused-function]
     _import_array(void)
     ^
    In file included from /usr/include/python2.7/numpy/ufuncobject.h:311:0,
                     from scipy/spatial/qhull.c:316:
    /usr/include/python2.7/numpy/__ufunc_api.h:236:1: warning: ‘_import_umath’ defined but not used [-Wunused-function]
     _import_umath(void)
     ^
    scipy/spatial/qhull.c: In function ‘__pyx_pw_5scipy_7spatial_5qhull_6_Qhull_9get_paraboloid_shift_scale’:
    scipy/spatial/qhull.c:5418:13: warning: ‘__pyx_v_paraboloid_shift’ may be used uninitialized in this function [-Wmaybe-uninitialized]
       __pyx_t_2 = PyFloat_FromDouble(__pyx_v_paraboloid_shift); if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 508; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
                 ^
    scipy/spatial/qhull.c:5244:10: note: ‘__pyx_v_paraboloid_shift’ was declared here
       double __pyx_v_paraboloid_shift;
              ^
    scipy/spatial/qhull.c:5416:13: warning: ‘__pyx_v_paraboloid_scale’ may be used uninitialized in this function [-Wmaybe-uninitialized]
       __pyx_t_1 = PyFloat_FromDouble(__pyx_v_paraboloid_scale); if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 508; __pyx_clineno = __LINE__; goto __pyx_L1_error;}
                 ^
    scipy/spatial/qhull.c:5243:10: note: ‘__pyx_v_paraboloid_scale’ was declared here
       double __pyx_v_paraboloid_scale;
              ^
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/io.c
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/merge.c
    scipy/spatial/qhull/src/merge.c: In function ‘qh_all_merges’:
    scipy/spatial/qhull/src/merge.c:219:10: warning: variable ‘freelistp’ set but not used [-Wunused-but-set-variable]
       void **freelistp;  /* used !qh_NOmem */
              ^
    scipy/spatial/qhull/src/merge.c: In function ‘qh_appendmergeset’:
    scipy/spatial/qhull/src/merge.c:322:10: warning: variable ‘freelistp’ set but not used [-Wunused-but-set-variable]
       void **freelistp; /* used !qh_NOmem */
              ^
    scipy/spatial/qhull/src/merge.c: In function ‘qh_mergecycle_ridges’:
    scipy/spatial/qhull/src/merge.c:2086:10: warning: variable ‘freelistp’ set but not used [-Wunused-but-set-variable]
       void **freelistp; /* used !qh_NOmem */
              ^
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/libqhull.c
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/poly.c
    scipy/spatial/qhull/src/poly.c: In function ‘qh_delfacet’:
    scipy/spatial/qhull/src/poly.c:248:10: warning: variable ‘freelistp’ set but not used [-Wunused-but-set-variable]
       void **freelistp; /* used !qh_NOmem */
              ^
    scipy/spatial/qhull/src/poly.c: In function ‘qh_makenew_nonsimplicial’:
    scipy/spatial/qhull/src/poly.c:564:10: warning: variable ‘freelistp’ set but not used [-Wunused-but-set-variable]
       void **freelistp; /* used !qh_NOmem */
              ^
    scipy/spatial/qhull/src/poly.c: In function ‘qh_newfacet’:
    scipy/spatial/qhull/src/poly.c:981:10: warning: variable ‘freelistp’ set but not used [-Wunused-but-set-variable]
       void **freelistp; /* used !qh_NOmem */
              ^
    scipy/spatial/qhull/src/poly.c: In function ‘qh_newridge’:
    scipy/spatial/qhull/src/poly.c:1014:10: warning: variable ‘freelistp’ set but not used [-Wunused-but-set-variable]
       void **freelistp;   /* used !qh_NOmem */
              ^
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/usermem.c
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/userprintf.c
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/global.c
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/poly2.c
    scipy/spatial/qhull/src/poly2.c: In function ‘qh_delridge’:
    scipy/spatial/qhull/src/poly2.c:1076:10: warning: variable ‘freelistp’ set but not used [-Wunused-but-set-variable]
       void **freelistp; /* used !qh_NOmem */
              ^
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/rboxlib.c
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/geom2.c
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/userprintf_rbox.c
    x86_64-linux-gnu-gcc: scipy/spatial/qhull/src/geom.c
    scipy/spatial/qhull/src/geom.c: In function ‘qh_projectpoint’:
    scipy/spatial/qhull/src/geom.c:897:10: warning: variable ‘freelistp’ set but not used [-Wunused-but-set-variable]
       void **freelistp; /* used !qh_NOmem */
              ^
    scipy/spatial/qhull/src/geom.c: In function ‘qh_setfacetplane’:
    scipy/spatial/qhull/src/geom.c:935:10: warning: variable ‘freelistp’ set but not used [-Wunused-but-set-variable]
       void **freelistp; /* used !qh_NOmem */
              ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/scipy/spatial/qhull.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/geom2.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/geom.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/global.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/io.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/libqhull.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/mem.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/merge.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/poly2.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/poly.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/qset.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/random.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/rboxlib.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/stat.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/user.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/usermem.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/userprintf.o build/temp.linux-x86_64-2.7/scipy/spatial/qhull/src/userprintf_rbox.o -L/usr/lib/atlas-base/atlas -L/usr/lib/atlas-base -Lbuild/temp.linux-x86_64-2.7 -llapack -lf77blas -lcblas -latlas -lgfortran -o build/lib.linux-x86_64-2.7/scipy/spatial/qhull.so
    building 'scipy.spatial.ckdtree' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/spatial/ckdtree.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/spatial/ckdtree.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/spatial/ckdtree.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1629:1: warning: ‘_import_array’ defined but not used [-Wunused-function]
     _import_array(void)
     ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ufuncobject.h:327:0,
                     from scipy/spatial/ckdtree.c:315:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__ufunc_api.h:241:1: warning: ‘_import_umath’ defined but not used [-Wunused-function]
     _import_umath(void)
     ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/spatial/ckdtree.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/spatial/ckdtree.so
    building 'scipy.spatial._distance_wrap' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/spatial/src/distance_wrap.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/spatial/src/distance_wrap.c:40:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    scipy/spatial/src/distance_wrap.c: In function ‘pdist_weighted_minkowski_wrap’:
    scipy/spatial/src/distance_wrap.c:866:7: warning: assignment discards ‘const’ qualifier from pointer target type [enabled by default]
         w = (const double*)w_->data;
           ^
    x86_64-linux-gnu-gcc: scipy/spatial/src/distance.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from scipy/spatial/src/distance.c:37:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from scipy/spatial/src/distance.c:37:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1629:1: warning: ‘_import_array’ defined but not used [-Wunused-function]
     _import_array(void)
     ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/spatial/src/distance_wrap.o build/temp.linux-x86_64-2.7/scipy/spatial/src/distance.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/spatial/_distance_wrap.so
    building 'scipy.special.specfun' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/special/specfunmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/special/specfunmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/special/specfunmodule.c:112:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/special/specfunmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o -Lbuild/temp.linux-x86_64-2.7 -lsc_specfun -lgfortran -o build/lib.linux-x86_64-2.7/scipy/special/specfun.so
    building 'scipy.special._ufuncs' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/tmp/pip_build_root/scipy/scipy/special -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -I/tmp/pip_build_root/scipy/scipy/special/c_misc -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/special/amos_wrappers.c
    In file included from scipy/special/amos_wrappers.h:11:0,
                     from scipy/special/amos_wrappers.c:8:
    scipy/special/sf_error.h:26:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     int sf_error_get_print();
     ^
    x86_64-linux-gnu-gcc: scipy/special/specfun_wrappers.c
    In file included from scipy/special/specfun_wrappers.h:16:0,
                     from scipy/special/specfun_wrappers.c:5:
    scipy/special/sf_error.h:26:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     int sf_error_get_print();
     ^
    scipy/special/specfun_wrappers.c: In function ‘cexpi_wrap’:
    scipy/special/specfun_wrappers.c:216:3: warning: implicit declaration of function ‘eixz_’ [-Wimplicit-function-declaration]
       F_FUNC(eixz,EIXZ)(&z, &outz);
       ^
    x86_64-linux-gnu-gcc: scipy/special/sf_error.c
    In file included from scipy/special/sf_error.c:6:0:
    scipy/special/sf_error.h:26:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     int sf_error_get_print();
     ^
    scipy/special/sf_error.c:28:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern int wrap_PyUFunc_getfperr();
     ^
    scipy/special/sf_error.c:37:5: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     int sf_error_get_print()
         ^
    x86_64-linux-gnu-gcc: scipy/special/cdf_wrappers.c
    In file included from scipy/special/cdf_wrappers.h:14:0,
                     from scipy/special/cdf_wrappers.c:6:
    scipy/special/sf_error.h:26:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     int sf_error_get_print();
     ^
    x86_64-linux-gnu-gcc: scipy/special/_ufuncs.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/special/_ufuncs.c:316:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from scipy/special/_ufuncs.c:318:0:
    scipy/special/sf_error.h:26:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     int sf_error_get_print();
     ^
    scipy/special/_ufuncs.c:3622:12: warning: ‘__pyx_f_5scipy_7special_7_ufuncs__set_errprint’ defined but not used [-Wunused-function]
     static int __pyx_f_5scipy_7special_7_ufuncs__set_errprint(int __pyx_v_flag) {
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/special/_logit.c
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/scipy/special/_ufuncs.o build/temp.linux-x86_64-2.7/scipy/special/sf_error.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/special/_logit.o build/temp.linux-x86_64-2.7/scipy/special/amos_wrappers.o build/temp.linux-x86_64-2.7/scipy/special/cdf_wrappers.o build/temp.linux-x86_64-2.7/scipy/special/specfun_wrappers.o -L/usr/local/lib/python2.7/dist-packages/numpy/core/lib -Lbuild/temp.linux-x86_64-2.7 -lsc_amos -lsc_c_misc -lsc_cephes -lsc_mach -lsc_cdf -lsc_specfun -lnpymath -lm -lgfortran -o build/lib.linux-x86_64-2.7/scipy/special/_ufuncs.so
    building 'scipy.special._ufuncs_cxx' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/tmp/pip_build_root/scipy/scipy/special -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/special/sf_error.c
    In file included from scipy/special/sf_error.c:6:0:
    scipy/special/sf_error.h:26:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     int sf_error_get_print();
     ^
    scipy/special/sf_error.c:28:1: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     extern int wrap_PyUFunc_getfperr();
     ^
    scipy/special/sf_error.c:37:5: warning: function declaration isn’t a prototype [-Wstrict-prototypes]
     int sf_error_get_print()
         ^
    compiling C++ sources
    C compiler: c++ -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -fPIC
    
    compile options: '-I/tmp/pip_build_root/scipy/scipy/special -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    c++: scipy/special/_ufuncs_cxx.cxx
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/special/_ufuncs_cxx.cxx:316:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    c++: scipy/special/_faddeeva.cxx
    c++: scipy/special/Faddeeva.cc
    c++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/special/sf_error.o build/temp.linux-x86_64-2.7/scipy/special/_ufuncs_cxx.o build/temp.linux-x86_64-2.7/scipy/special/_faddeeva.o build/temp.linux-x86_64-2.7/scipy/special/Faddeeva.o -L/usr/local/lib/python2.7/dist-packages/numpy/core/lib -Lbuild/temp.linux-x86_64-2.7 -lnpymath -lm -o build/lib.linux-x86_64-2.7/scipy/special/_ufuncs_cxx.so
    building 'scipy.stats.statlib' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/stats/statlibmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/stats/statlibmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/stats/statlibmodule.c: In function ‘f2py_rout_statlib_prho’:
    build/src.linux-x86_64-2.7/scipy/stats/statlibmodule.c:636:10: warning: variable ‘prho_return_value’ set but not used [-Wunused-but-set-variable]
       double prho_return_value=0;
              ^
    build/src.linux-x86_64-2.7/scipy/stats/statlibmodule.c: At top level:
    build/src.linux-x86_64-2.7/scipy/stats/statlibmodule.c:111:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/stats/statlibmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o -Lbuild/temp.linux-x86_64-2.7 -lstatlib -lgfortran -o build/lib.linux-x86_64-2.7/scipy/stats/statlib.so
    building 'scipy.stats.vonmises_cython' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/stats/vonmises_cython.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/stats/vonmises_cython.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/stats/vonmises_cython.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1629:1: warning: ‘_import_array’ defined but not used [-Wunused-function]
     _import_array(void)
     ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ufuncobject.h:327:0,
                     from scipy/stats/vonmises_cython.c:315:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__ufunc_api.h:241:1: warning: ‘_import_umath’ defined but not used [-Wunused-function]
     _import_umath(void)
     ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/stats/vonmises_cython.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/stats/vonmises_cython.so
    building 'scipy.stats._rank' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/stats/_rank.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/stats/_rank.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:26:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/stats/_rank.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h:1629:1: warning: ‘_import_array’ defined but not used [-Wunused-function]
     _import_array(void)
     ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ufuncobject.h:327:0,
                     from scipy/stats/_rank.c:315:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__ufunc_api.h:241:1: warning: ‘_import_umath’ defined but not used [-Wunused-function]
     _import_umath(void)
     ^
    scipy/stats/_rank.c: In function ‘__pyx_fuse_2__pyx_f_5scipy_5stats_5_rank__rankdata_fused’:
    scipy/stats/_rank.c:2999:170: warning: ‘__pyx_v_tie_rank’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                     *__Pyx_BufPtrStrided1d(__pyx_t_5numpy_float64_t *, __pyx_pybuffernd_ranks.rcbuffer->pybuffer.buf, __pyx_t_25, __pyx_pybuffernd_ranks.diminfo[0].strides) = __pyx_v_tie_rank;
                                                                                                                                                                              ^
    scipy/stats/_rank.c: In function ‘__pyx_fuse_1__pyx_f_5scipy_5stats_5_rank__rankdata_fused’:
    scipy/stats/_rank.c:2413:170: warning: ‘__pyx_v_tie_rank’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                     *__Pyx_BufPtrStrided1d(__pyx_t_5numpy_float64_t *, __pyx_pybuffernd_ranks.rcbuffer->pybuffer.buf, __pyx_t_25, __pyx_pybuffernd_ranks.diminfo[0].strides) = __pyx_v_tie_rank;
                                                                                                                                                                              ^
    scipy/stats/_rank.c: In function ‘__pyx_fuse_0__pyx_f_5scipy_5stats_5_rank__rankdata_fused’:
    scipy/stats/_rank.c:1827:170: warning: ‘__pyx_v_tie_rank’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                     *__Pyx_BufPtrStrided1d(__pyx_t_5numpy_float64_t *, __pyx_pybuffernd_ranks.rcbuffer->pybuffer.buf, __pyx_t_25, __pyx_pybuffernd_ranks.diminfo[0].strides) = __pyx_v_tie_rank;
                                                                                                                                                                              ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/stats/_rank.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/stats/_rank.so
    building 'scipy.stats.futil' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/stats/futilmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/stats/futilmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/stats/futilmodule.c:104:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: scipy/stats/futil.f
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/stats/futilmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/scipy/stats/futil.o -Lbuild/temp.linux-x86_64-2.7 -lgfortran -o build/lib.linux-x86_64-2.7/scipy/stats/futil.so
    building 'scipy.stats.mvn' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/fortranobject.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/fortranobject.c:2:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: build/src.linux-x86_64-2.7/scipy/stats/mvnmodule.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from build/src.linux-x86_64-2.7/fortranobject.h:13,
                     from build/src.linux-x86_64-2.7/scipy/stats/mvnmodule.c:18:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    build/src.linux-x86_64-2.7/scipy/stats/mvnmodule.c:104:12: warning: ‘f2py_size’ defined but not used [-Wunused-function]
     static int f2py_size(PyArrayObject* var, ...)
                ^
    compiling Fortran sources
    Fortran f77 compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran f90 compiler: /usr/bin/gfortran -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    Fortran fix compiler: /usr/bin/gfortran -Wall -ffixed-form -fno-second-underscore -Wall -fno-second-underscore -fPIC -O3 -funroll-loops
    compile options: '-Ibuild/src.linux-x86_64-2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    gfortran:f77: scipy/stats/mvndst.f
    scipy/stats/mvndst.f:130.18:
    
             INFORM = MVNDNT(N, CORREL, LOWER, UPPER, INFIN, INFIS, D, E)
                      1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/stats/mvndst.f:715.14:
    
             JP = J + MVNUNI()*( NK + 1 - J )
                  1
    Warning: Possible change of value in conversion from REAL(8) to INTEGER(4) at (1)
    scipy/stats/mvndst.f: In function ‘bvnmvn’:
    scipy/stats/mvndst.f:951:0: warning: ‘__result_bvnmvn’ may be used uninitialized in this function [-Wmaybe-uninitialized]
           END
     ^
    scipy/stats/mvndst.f:909:0: note: ‘__result_bvnmvn’ was declared here
           DOUBLE PRECISION FUNCTION BVNMVN( LOWER, UPPER, INFIN, CORREL )
     ^
    scipy/stats/mvndst.f: In function ‘covsrt’:
    scipy/stats/mvndst.f:356:0: warning: ‘bmin’ may be used uninitialized in this function [-Wmaybe-uninitialized]
                       IF ( INFI(I) .EQ. 2 ) Y(I) = ( AMIN + BMIN )/2
     ^
    scipy/stats/mvndst.f:356:0: warning: ‘amin’ may be used uninitialized in this function [-Wmaybe-uninitialized]
    gfortran:f77: build/src.linux-x86_64-2.7/scipy/stats/mvn-f2pywrappers.f
    /usr/bin/gfortran -Wall -Wall -shared build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/stats/mvnmodule.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/fortranobject.o build/temp.linux-x86_64-2.7/scipy/stats/mvndst.o build/temp.linux-x86_64-2.7/build/src.linux-x86_64-2.7/scipy/stats/mvn-f2pywrappers.o -Lbuild/temp.linux-x86_64-2.7 -lgfortran -o build/lib.linux-x86_64-2.7/scipy/stats/mvn.so
    building 'scipy.ndimage._nd_image' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Iscipy/ndimage/src -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/ndimage/src/ni_filters.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h:9,
                     from scipy/ndimage/src/nd_image.h:41,
                     from scipy/ndimage/src/ni_support.h:35,
                     from scipy/ndimage/src/ni_filters.c:32:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: scipy/ndimage/src/ni_support.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h:9,
                     from scipy/ndimage/src/nd_image.h:41,
                     from scipy/ndimage/src/ni_support.h:35,
                     from scipy/ndimage/src/ni_support.c:32:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: scipy/ndimage/src/ni_morphology.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h:9,
                     from scipy/ndimage/src/nd_image.h:41,
                     from scipy/ndimage/src/ni_support.h:35,
                     from scipy/ndimage/src/ni_morphology.c:32:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: scipy/ndimage/src/ni_interpolation.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h:9,
                     from scipy/ndimage/src/nd_image.h:41,
                     from scipy/ndimage/src/ni_support.h:35,
                     from scipy/ndimage/src/ni_interpolation.c:32:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: scipy/ndimage/src/ni_fourier.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h:9,
                     from scipy/ndimage/src/nd_image.h:41,
                     from scipy/ndimage/src/ni_support.h:35,
                     from scipy/ndimage/src/ni_fourier.c:32:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: scipy/ndimage/src/ni_measure.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h:9,
                     from scipy/ndimage/src/nd_image.h:41,
                     from scipy/ndimage/src/ni_support.h:35,
                     from scipy/ndimage/src/ni_measure.c:32:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc: scipy/ndimage/src/nd_image.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/noprefix.h:9,
                     from scipy/ndimage/src/nd_image.h:41,
                     from scipy/ndimage/src/nd_image.c:32:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/ndimage/src/nd_image.o build/temp.linux-x86_64-2.7/scipy/ndimage/src/ni_filters.o build/temp.linux-x86_64-2.7/scipy/ndimage/src/ni_fourier.o build/temp.linux-x86_64-2.7/scipy/ndimage/src/ni_interpolation.o build/temp.linux-x86_64-2.7/scipy/ndimage/src/ni_measure.o build/temp.linux-x86_64-2.7/scipy/ndimage/src/ni_morphology.o build/temp.linux-x86_64-2.7/scipy/ndimage/src/ni_support.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/ndimage/_nd_image.so
    building 'scipy.ndimage._ni_label' extension
    compiling C sources
    C compiler: x86_64-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC
    
    compile options: '-Iscipy/ndimage/src -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c'
    x86_64-linux-gnu-gcc: scipy/ndimage/src/_ni_label.c
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1760:0,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h:17,
                     from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h:4,
                     from scipy/ndimage/src/_ni_label.c:314:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
     #warning "Using deprecated NumPy API, disable it by " \
      ^
    In file included from /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ufuncobject.h:327:0,
                     from scipy/ndimage/src/_ni_label.c:315:
    /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/__ufunc_api.h:241:1: warning: ‘_import_umath’ defined but not used [-Wunused-function]
     _import_umath(void)
     ^
    x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security build/temp.linux-x86_64-2.7/scipy/ndimage/src/_ni_label.o -Lbuild/temp.linux-x86_64-2.7 -o build/lib.linux-x86_64-2.7/scipy/ndimage/_ni_label.so
    
    no previously-included directories found matching 'scipy/special/tests/data/boost'
    no previously-included directories found matching 'scipy/special/tests/data/gsl'
    no previously-included directories found matching 'doc/build'
    no previously-included directories found matching 'doc/source/generated'
    no previously-included directories found matching '*/__pycache__'
    warning: no previously-included files matching '*~' found anywhere in distribution
    warning: no previously-included files matching '*.bak' found anywhere in distribution
    warning: no previously-included files matching '*.swp' found anywhere in distribution
    warning: no previously-included files matching '*.pyo' found anywhere in distribution
Successfully installed scipy
Cleaning up...
Requirement already up-to-date: numpy in /usr/local/lib/python2.7/dist-packages
Cleaning up...
Downloading/unpacking ipython from https://pypi.python.org/packages/source/i/ipython/ipython-1.1.0.tar.gz#md5=70d69c78122923879232567ac3c47cef
  Running setup.py egg_info for package ipython
    
Installing collected packages: ipython
  Found existing installation: ipython 0.13.2
    Uninstalling ipython:
      Successfully uninstalled ipython
  Running setup.py install for ipython
    
    Installing ipcontroller script to /usr/local/bin
    Installing iptest script to /usr/local/bin
    Installing ipcluster script to /usr/local/bin
    Installing ipython script to /usr/local/bin
    Installing pycolor script to /usr/local/bin
    Installing iplogger script to /usr/local/bin
    Installing irunner script to /usr/local/bin
    Installing ipengine script to /usr/local/bin
Successfully installed ipython
Cleaning up...
Requirement already up-to-date: scipy in /usr/local/lib/python2.7/dist-packages
Cleaning up...
Requirement already up-to-date: numpy in /usr/local/lib/python2.7/dist-packages
Cleaning up...
Requirement already up-to-date: ipython in /usr/local/lib/python2.7/dist-packages
Cleaning up...
