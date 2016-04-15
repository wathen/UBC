python Maxwell.py    -help \
	 -ksp_view \
	 -ksp_monitor \
	 -fieldsplit_field1_ksp_type cg \
	 -fieldsplit_field1_pc_type hypre \
	 -fieldsplit_field2_ksp_type cg \
	 -fieldsplit_field2_pc_type hypre \
	 -fieldsplit_field2_ksp_rtol 1e-8 \
	 -fieldsplit_field1_ksp_rtol 1e-8 \
	 -fieldsplit_field1_pc_hypre_boomeramg_strong_threshold 0.5 \
	 -fieldsplit_field2_pc_hypre_boomeramg_strong_threshold 0.5
--------------------------------------------------------------------------
Petsc Release Version 3.4.3, Oct, 15, 2013 
       The PETSc Team
    petsc-maint@mcs.anl.gov
 http://www.mcs.anl.gov/petsc/
See docs/changes/index.html for recent updates.
See docs/faq.html for problems.
See docs/manualpages/index.html for help. 
Libraries linked from /home/mwathen/programs4/petsc/petsc-3.4.3/arch-linux2-c-opt/lib
--------------------------------------------------------------------------
Options for all PETSc programs:
 -help: prints help method for each option
 -on_error_abort: cause an abort when an error is detected. Useful 
        only when run in the debugger
 -on_error_attach_debugger [gdb,dbx,xxgdb,ups,noxterm]
       start the debugger in new xterm
       unless noxterm is given
 -start_in_debugger [gdb,dbx,xxgdb,ups,noxterm]
       start all processes in the debugger
 -on_error_emacs <machinename>
    emacs jumps to error file
 -debugger_nodes [n1,n2,..] Nodes to start in debugger
 -debugger_pause [m] : delay (in seconds) to attach debugger
 -stop_for_debugger : prints message on how to attach debugger manually
                      waits the delay for you to attach
 -display display: Location where graphics and debuggers are displayed
 -no_signal_handler: do not trap error signals
 -mpi_return_on_error: MPI returns error code, rather than abort on internal error
 -fp_trap: stop on floating point exceptions
           note on IBM RS6000 this slows run greatly
 -malloc_dump <optional filename>: dump list of unfreed memory at conclusion
 -malloc: use our error checking malloc
 -malloc no: don't use error checking malloc
 -malloc_info: prints total memory usage
 -malloc_log: keeps log of all memory allocations
 -malloc_debug: enables extended checking for memory corruption
 -options_table: dump list of options inputted
 -options_left: dump list of unused options
 -options_left no: don't dump list of unused options
 -tmp tmpdir: alternative /tmp directory
 -shared_tmp: tmp directory is shared by all processors
 -not_shared_tmp: each processor has separate tmp directory
 -memory_info: print memory usage at end of run
 -server <port>: Run PETSc webserver (default port is 8080) see PetscWebServe()
 -get_total_flops: total flops over all processors
 -log[_summary _summary_python]: logging objects and events
 -log_trace [filename]: prints trace of all PETSc calls
 -info <optional filename>: print informative messages about the calculations
 -v: prints PETSc version number and release date
 -options_file <file>: reads options from file
 -petsc_sleep n: sleeps n seconds before running program
-----------------------------------------------
------Additional PETSc component options--------
 -log_summary_exclude: <vec,mat,pc.ksp,snes>
 -info_exclude: <null,vec,mat,pc,ksp,snes,ts>
-----------------------------------------------
Options database options -------------------------------------------------
  -options_monitor <stdout>: Monitor options database (PetscOptionsMonitorSet)
  -options_monitor_cancel: <FALSE> Cancel all options database monitors (PetscOptionsMonitorCancel)
1


V:   [ 2624.] Q:   [ 289.] W:   [ 2913.] 


Thread comm - setting number of threads -------------------------------------------------
  -threadcomm_nthreads <1>: number of threads to use in the thread communicator (PetscThreadCommSetNThreads)
Thread comm - setting thread affinities -------------------------------------------------
  -threadcomm_affinities <-434464856>: Set core affinities of threads (PetscThreadCommSetAffinities)
Thread comm - setting number of kernels -------------------------------------------------
  -threadcomm_nkernels <16>: number of kernels that can be launched simultaneously ()
Thread comm - setting threading model -------------------------------------------------
  -threadcomm_type <nothread>: Thread communicator model (one of) nothread (PetscThreadCommSetType)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Options for SEQAIJ matrix -------------------------------------------------
  -mat_no_unroll: <FALSE> Do not optimize for inodes (slower) (None)
  -mat_no_inode: <FALSE> Do not optimize for inodes -slower- (None)
  -mat_inode_limit <5>: Do not use inodes larger then this value (None)
Matrix (Mat) options -------------------------------------------------
  -mat_type <aij>: Matrix type (one of) mffd mpimaij seqmaij maij is shell composite mpiaij
      seqaij mpiaijperm seqaijperm seqaijcrl mpiaijcrl mpibaij seqbaij mpisbaij seqsbaij mpibstrm seqbstrm mpisbstrm seqsbstrm mpidense seqdense mpiadj scatter blockmat nest schurcomplement hyprestruct python (MatSetType)
  -mat_view <>: Display mat with the viewer on MatAssemblyEnd() (MatView)
  -mat_is_symmetric: Checks if mat is symmetric on MatAssemblyEnd() (MatIsSymmetric)
  -mat_is_symmetric <0>: Checks if mat is symmetric on MatAssemblyEnd() (MatIsSymmetric)
  -mat_null_space_test: <FALSE> Checks if provided null space is correct in MatAssemblyEnd() (MatSetNullSpaceTest)
  -mat_new_nonzero_location_err: <FALSE> Generate an error if new nonzeros are created in the matrix structure (useful to test preallocation) (MatSetOption)
  -mat_new_nonzero_allocation_err: <FALSE> Generate an error if new nonzeros are allocated in the matrix structure (useful to test preallocation) (MatSetOption)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
0.0243680477142
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Options for SEQAIJ matrix -------------------------------------------------
  -mat_no_unroll: <FALSE> Do not optimize for inodes (slower) (None)
  -mat_no_inode: <FALSE> Do not optimize for inodes -slower- (None)
  -mat_inode_limit <5>: Do not use inodes larger then this value (None)
Matrix (Mat) options -------------------------------------------------
  -mat_type <aij>: Matrix type (one of) mffd mpimaij seqmaij maij is shell composite mpiaij
      seqaij mpiaijperm seqaijperm seqaijcrl mpiaijcrl mpibaij seqbaij mpisbaij seqsbaij mpibstrm seqbstrm mpisbstrm seqsbstrm mpidense seqdense mpiadj scatter blockmat nest schurcomplement hyprestruct python (MatSetType)
  -mat_view <>: Display mat with the viewer on MatAssemblyEnd() (MatView)
  -mat_is_symmetric: Checks if mat is symmetric on MatAssemblyEnd() (MatIsSymmetric)
  -mat_is_symmetric <0>: Checks if mat is symmetric on MatAssemblyEnd() (MatIsSymmetric)
  -mat_null_space_test: <FALSE> Checks if provided null space is correct in MatAssemblyEnd() (MatSetNullSpaceTest)
  -mat_new_nonzero_location_err: <FALSE> Generate an error if new nonzeros are created in the matrix structure (useful to test preallocation) (MatSetOption)
  -mat_new_nonzero_allocation_err: <FALSE> Generate an error if new nonzeros are allocated in the matrix structure (useful to test preallocation) (MatSetOption)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
0.019770860672
Preconditioner (PC) options -------------------------------------------------
  -pc_type <fieldsplit>: Preconditioner (one of) none jacobi pbjacobi bjacobi sor lu shell mg
      eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic hmpi asa cp lsc redistribute svd gamg ml spai hypre pfmg syspfmg tfs python (PCSetType)
  -pc_use_amat: <FALSE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  FieldSplit options
  -pc_fieldsplit_dm_splits: <TRUE> Whether to use DMCreateFieldDecomposition() for splits (PCFieldSplitSetDMSplits)
  -pc_fieldsplit_block_size <-1>: Blocksize that defines number of fields (PCFieldSplitSetBlockSize)
  -pc_fieldsplit_type <ADDITIVE> (choose one of) ADDITIVE MULTIPLICATIVE SYMMETRIC_MULTIPLICATIVE SPECIAL SCHUR (PCFieldSplitSetType)
Krylov Method (KSP) options -------------------------------------------------
  -ksp_type <minres>: Krylov method (one of) cg groppcg pipecg cgne nash stcg gltr richardson
      chebyshev gmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl cgs tfqmr cr pipecr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd gcr pgmres specest dgmres python (KSPSetType)
  -ksp_max_it <10000>: Maximum number of iterations (KSPSetTolerances)
  -ksp_rtol <1e-06>: Relative decrease in residual norm (KSPSetTolerances)
  -ksp_atol <1e-50>: Absolute value of residual norm (KSPSetTolerances)
  -ksp_divtol <10000>: Residual norm increase cause divergence (KSPSetTolerances)
  -ksp_converged_use_initial_residual_norm: <FALSE> Use initial residual residual norm for computing relative convergence (KSPDefaultConvergedSetUIRNorm)
  -ksp_converged_use_min_initial_residual_norm: <FALSE> Use minimum of initial residual norm and b for computing relative convergence (KSPDefaultConvergedSetUMIRNorm)
  -ksp_initial_guess_nonzero: <FALSE> Use the contents of the solution vector for initial guess (KSPSetInitialNonzero)
  -ksp_knoll: <FALSE> Use preconditioner applied to b for initial guess (KSPSetInitialGuessKnoll)
  -ksp_error_if_not_converged: <FALSE> Generate error if solver does not converge (KSPSetErrorIfNotConverged)
  -ksp_fischer_guess <0>: Use Paul Fischer's algorithm for initial guess (KSPSetUseFischerGuess)
  -ksp_convergence_test <default> (choose one of) default skip (KSPSetConvergenceTest)
  -ksp_norm_type <PRECONDITIONED> (choose one of) NONE PRECONDITIONED UNPRECONDITIONED NATURAL (KSPSetNormType)
  -ksp_check_norm_iteration <-1>: First iteration to compute residual norm (KSPSetCheckNormIteration)
  -ksp_lag_norm: <FALSE> Lag the calculation of the residual norm (KSPSetLagNorm)
  -ksp_diagonal_scale: <FALSE> Diagonal scale matrix before building preconditioner (KSPSetDiagonalScale)
  -ksp_diagonal_scale_fix: <FALSE> Fix diagonally scaled matrix after solve (KSPSetDiagonalScaleFix)
  -ksp_constant_null_space: <FALSE> Add constant null space to Krylov solver (KSPSetNullSpace)
  -ksp_converged_reason: <FALSE> Print reason for converged or diverged (KSPSolve)
  -ksp_monitor_cancel: <FALSE> Remove any hardwired monitor routines (KSPMonitorCancel)
  -ksp_monitor <stdout>: Monitor preconditioned residual norm (KSPMonitorSet)
  -ksp_monitor_range <stdout>: Monitor percent of residual entries more than 10 percent of max (KSPMonitorRange)
  -ksp_monitor_solution: <FALSE> Monitor solution graphically (KSPMonitorSet)
  -ksp_monitor_true_residual <stdout>: Monitor true residual norm (KSPMonitorSet)
  -ksp_monitor_max <stdout>: Monitor true residual max norm (KSPMonitorSet)
  -ksp_monitor_singular_value <stdout>: Monitor singular values (KSPMonitorSet)
  -ksp_monitor_short <stdout>: Monitor preconditioned residual norm with fewer digits (KSPMonitorSet)
  -ksp_monitor_python <(null)>: Use Python function (KSPMonitorSet)
  -ksp_monitor_lg_residualnorm: <FALSE> Monitor graphically preconditioned residual norm (KSPMonitorSet)
  -ksp_monitor_lg_true_residualnorm: <FALSE> Monitor graphically true residual norm (KSPMonitorSet)
  -ksp_monitor_lg_range: <FALSE> Monitor graphically range of preconditioned residual norm (KSPMonitorSet)
  -ksp_pc_side <LEFT> (choose one of) LEFT RIGHT SYMMETRIC (KSPSetPCSide)
  -ksp_compute_singularvalues: <FALSE> Compute singular values of preconditioned operator (KSPSetComputeSingularValues)
  -ksp_compute_eigenvalues: <FALSE> Compute eigenvalues of preconditioned operator (KSPSetComputeSingularValues)
  -ksp_plot_eigenvalues: <FALSE> Scatter plot extreme eigenvalues (KSPSetComputeSingularValues)
  -ksp_view: View linear solver parameters (KSPView)
KSP Object: 1 MPI processes
  type: minres
  maximum iterations=10000, initial guess is zero
  tolerances:  relative=1e-06, absolute=1e-50, divergence=10000
  left preconditioning
  using DEFAULT norm type for convergence test
PC Object: 1 MPI processes
  type: fieldsplit
  PC has not been set up so information may be incomplete
    FieldSplit with ADDITIVE composition: total splits = 2
    Solver info for each split is in the following KSP objects:
    Split number 0 Defined by IS
    KSP Object:    (fieldsplit_field1_)     1 MPI processes
      type: preonly
      maximum iterations=10000, initial guess is zero
      tolerances:  relative=1e-05, absolute=1e-50, divergence=10000
      left preconditioning
      using DEFAULT norm type for convergence test
    PC Object:    (fieldsplit_field1_)     1 MPI processes
      type not yet set
      PC has not been set up so information may be incomplete
    Split number 1 Defined by IS
    KSP Object:    (fieldsplit_field2_)     1 MPI processes
      type: preonly
      maximum iterations=10000, initial guess is zero
      tolerances:  relative=1e-05, absolute=1e-50, divergence=10000
      left preconditioning
      using DEFAULT norm type for convergence test
    PC Object:    (fieldsplit_field2_)     1 MPI processes
      type not yet set
      PC has not been set up so information may be incomplete
  linear system matrix followed by preconditioner matrix:
  Matrix Object:   1 MPI processes
    type: seqaij
    rows=2913, cols=2913
    total: nonzeros=50401, allocated nonzeros=50401
    total number of mallocs used during MatSetValues calls =0
      using I-node routines: found 1600 nodes, limit used is 5
  Matrix Object:   1 MPI processes
    type: seqaij
    rows=2913, cols=2913
    total: nonzeros=50401, allocated nonzeros=50401
    total number of mallocs used during MatSetValues calls =0
      using I-node routines: found 1600 nodes, limit used is 5
Preconditioner (PC) options -------------------------------------------------
  -fieldsplit_field1_pc_type <ilu>: Preconditioner (one of) none jacobi pbjacobi bjacobi sor lu shell mg
      eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic hmpi asa cp lsc redistribute svd gamg ml spai hypre pfmg syspfmg tfs python (PCSetType)
  -fieldsplit_field1_pc_use_amat: <FALSE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  HYPRE preconditioner options
  -fieldsplit_field1_pc_hypre_type <boomeramg> (choose one of) pilut parasails boomeramg euclid (PCHYPRESetType)
  HYPRE BoomerAMG Options
  -fieldsplit_field1_pc_hypre_boomeramg_cycle_type <V> (choose one of) V W (None)
  -fieldsplit_field1_pc_hypre_boomeramg_max_levels <25>: Number of levels (of grids) allowed (None)
  -fieldsplit_field1_pc_hypre_boomeramg_max_iter <1>: Maximum iterations used PER hypre call (None)
  -fieldsplit_field1_pc_hypre_boomeramg_tol <0>: Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations) (None)
  -fieldsplit_field1_pc_hypre_boomeramg_truncfactor <0>: Truncation factor for interpolation (0=no truncation) (None)
  -fieldsplit_field1_pc_hypre_boomeramg_P_max <0>: Max elements per row for interpolation operator (0=unlimited) (None)
  -fieldsplit_field1_pc_hypre_boomeramg_agg_nl <0>: Number of levels of aggressive coarsening (None)
  -fieldsplit_field1_pc_hypre_boomeramg_agg_num_paths <1>: Number of paths for aggressive coarsening (None)
  -fieldsplit_field1_pc_hypre_boomeramg_strong_threshold <0.25>: Threshold for being strongly connected (None)
  -fieldsplit_field1_pc_hypre_boomeramg_max_row_sum <0.9>: Maximum row sum (None)
  -fieldsplit_field1_pc_hypre_boomeramg_grid_sweeps_all <1>: Number of sweeps for the up and down grid levels (None)
  -fieldsplit_field1_pc_hypre_boomeramg_grid_sweeps_down <1>: Number of sweeps for the down cycles (None)
  -fieldsplit_field1_pc_hypre_boomeramg_grid_sweeps_up <1>: Number of sweeps for the up cycles (None)
  -fieldsplit_field1_pc_hypre_boomeramg_grid_sweeps_coarse <1>: Number of sweeps for the coarse level (None)
  -fieldsplit_field1_pc_hypre_boomeramg_relax_type_all <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -fieldsplit_field1_pc_hypre_boomeramg_relax_type_down <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -fieldsplit_field1_pc_hypre_boomeramg_relax_type_up <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -fieldsplit_field1_pc_hypre_boomeramg_relax_type_coarse <Gaussian-elimination> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -fieldsplit_field1_pc_hypre_boomeramg_relax_weight_all <1>: Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps) (None)
  -fieldsplit_field1_pc_hypre_boomeramg_relax_weight_level <1>: Set the relaxation weight for a particular level (weight,level) (None)
  -fieldsplit_field1_pc_hypre_boomeramg_outer_relax_weight_all <1>: Outer relaxation weight for all levels (-k = determined with k CG steps) (None)
  -fieldsplit_field1_pc_hypre_boomeramg_outer_relax_weight_level <1>: Set the outer relaxation weight for a particular level (weight,level) (None)
  -fieldsplit_field1_pc_hypre_boomeramg_no_CF: <FALSE> Do not use CF-relaxation (None)
  -fieldsplit_field1_pc_hypre_boomeramg_measure_type <local> (choose one of) local global (None)
  -fieldsplit_field1_pc_hypre_boomeramg_coarsen_type <Falgout> (choose one of) CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None)
  -fieldsplit_field1_pc_hypre_boomeramg_interp_type <classical> (choose one of) classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1 (None)
  -fieldsplit_field1_pc_hypre_boomeramg_print_statistics: Print statistics (None)
  -fieldsplit_field1_pc_hypre_boomeramg_print_debug: Print debug information (None)
  -fieldsplit_field1_pc_hypre_boomeramg_nodal_coarsen: <FALSE> HYPRE_BoomerAMGSetNodal() (None)
  -fieldsplit_field1_pc_hypre_boomeramg_nodal_relaxation: <FALSE> Nodal relaxation via Schwarz (None)
Krylov Method (KSP) options -------------------------------------------------
  -fieldsplit_field1_ksp_type <preonly>: Krylov method (one of) cg groppcg pipecg cgne nash stcg gltr richardson
      chebyshev gmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl cgs tfqmr cr pipecr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd gcr pgmres specest dgmres python (KSPSetType)
  -fieldsplit_field1_ksp_max_it <10000>: Maximum number of iterations (KSPSetTolerances)
  -fieldsplit_field1_ksp_rtol <1e-05>: Relative decrease in residual norm (KSPSetTolerances)
  -fieldsplit_field1_ksp_atol <1e-50>: Absolute value of residual norm (KSPSetTolerances)
  -fieldsplit_field1_ksp_divtol <10000>: Residual norm increase cause divergence (KSPSetTolerances)
  -fieldsplit_field1_ksp_converged_use_initial_residual_norm: <FALSE> Use initial residual residual norm for computing relative convergence (KSPDefaultConvergedSetUIRNorm)
  -fieldsplit_field1_ksp_converged_use_min_initial_residual_norm: <FALSE> Use minimum of initial residual norm and b for computing relative convergence (KSPDefaultConvergedSetUMIRNorm)
  -fieldsplit_field1_ksp_initial_guess_nonzero: <FALSE> Use the contents of the solution vector for initial guess (KSPSetInitialNonzero)
  -fieldsplit_field1_ksp_knoll: <FALSE> Use preconditioner applied to b for initial guess (KSPSetInitialGuessKnoll)
  -fieldsplit_field1_ksp_error_if_not_converged: <FALSE> Generate error if solver does not converge (KSPSetErrorIfNotConverged)
  -fieldsplit_field1_ksp_fischer_guess <0>: Use Paul Fischer's algorithm for initial guess (KSPSetUseFischerGuess)
  -fieldsplit_field1_ksp_convergence_test <default> (choose one of) default skip (KSPSetConvergenceTest)
  -fieldsplit_field1_ksp_norm_type <PRECONDITIONED> (choose one of) NONE PRECONDITIONED UNPRECONDITIONED NATURAL (KSPSetNormType)
  -fieldsplit_field1_ksp_check_norm_iteration <-1>: First iteration to compute residual norm (KSPSetCheckNormIteration)
  -fieldsplit_field1_ksp_lag_norm: <FALSE> Lag the calculation of the residual norm (KSPSetLagNorm)
  -fieldsplit_field1_ksp_diagonal_scale: <FALSE> Diagonal scale matrix before building preconditioner (KSPSetDiagonalScale)
  -fieldsplit_field1_ksp_diagonal_scale_fix: <FALSE> Fix diagonally scaled matrix after solve (KSPSetDiagonalScaleFix)
  -fieldsplit_field1_ksp_constant_null_space: <FALSE> Add constant null space to Krylov solver (KSPSetNullSpace)
  -fieldsplit_field1_ksp_converged_reason: <FALSE> Print reason for converged or diverged (KSPSolve)
  -fieldsplit_field1_ksp_monitor_cancel: <FALSE> Remove any hardwired monitor routines (KSPMonitorCancel)
  -fieldsplit_field1_ksp_monitor <stdout>: Monitor preconditioned residual norm (KSPMonitorSet)
  -fieldsplit_field1_ksp_monitor_range <stdout>: Monitor percent of residual entries more than 10 percent of max (KSPMonitorRange)
  -fieldsplit_field1_ksp_monitor_solution: <FALSE> Monitor solution graphically (KSPMonitorSet)
  -fieldsplit_field1_ksp_monitor_true_residual <stdout>: Monitor true residual norm (KSPMonitorSet)
  -fieldsplit_field1_ksp_monitor_max <stdout>: Monitor true residual max norm (KSPMonitorSet)
  -fieldsplit_field1_ksp_monitor_singular_value <stdout>: Monitor singular values (KSPMonitorSet)
  -fieldsplit_field1_ksp_monitor_short <stdout>: Monitor preconditioned residual norm with fewer digits (KSPMonitorSet)
  -fieldsplit_field1_ksp_monitor_python <(null)>: Use Python function (KSPMonitorSet)
  -fieldsplit_field1_ksp_monitor_lg_residualnorm: <FALSE> Monitor graphically preconditioned residual norm (KSPMonitorSet)
  -fieldsplit_field1_ksp_monitor_lg_true_residualnorm: <FALSE> Monitor graphically true residual norm (KSPMonitorSet)
  -fieldsplit_field1_ksp_monitor_lg_range: <FALSE> Monitor graphically range of preconditioned residual norm (KSPMonitorSet)
  -fieldsplit_field1_ksp_pc_side <LEFT> (choose one of) LEFT RIGHT SYMMETRIC (KSPSetPCSide)
  -fieldsplit_field1_ksp_compute_singularvalues: <FALSE> Compute singular values of preconditioned operator (KSPSetComputeSingularValues)
  -fieldsplit_field1_ksp_compute_eigenvalues: <FALSE> Compute eigenvalues of preconditioned operator (KSPSetComputeSingularValues)
  -fieldsplit_field1_ksp_plot_eigenvalues: <FALSE> Scatter plot extreme eigenvalues (KSPSetComputeSingularValues)
  KSP CG and CGNE options
  -fieldsplit_field1_ksp_cg_single_reduction: <FALSE> Merge inner products into single MPI_Allreduce() (KSPCGUseSingleReduction)
  -fieldsplit_field1_ksp_view: View linear solver parameters (KSPView)
KSP Object:(fieldsplit_field1_) 1 MPI processes
  type: cg
  maximum iterations=10000, initial guess is zero
  tolerances:  relative=1e-08, absolute=1e-50, divergence=10000
  left preconditioning
  using DEFAULT norm type for convergence test
PC Object:(fieldsplit_field1_) 1 MPI processes
  type: hypre
  PC has not been set up so information may be incomplete
    HYPRE BoomerAMG preconditioning
    HYPRE BoomerAMG: Cycle type V
    HYPRE BoomerAMG: Maximum number of levels 25
    HYPRE BoomerAMG: Maximum number of iterations PER hypre call 1
    HYPRE BoomerAMG: Convergence tolerance PER hypre call 0
    HYPRE BoomerAMG: Threshold for strong coupling 0.5
    HYPRE BoomerAMG: Interpolation truncation factor 0
    HYPRE BoomerAMG: Interpolation: max elements per row 0
    HYPRE BoomerAMG: Number of levels of aggressive coarsening 0
    HYPRE BoomerAMG: Number of paths for aggressive coarsening 1
    HYPRE BoomerAMG: Maximum row sums 0.9
    HYPRE BoomerAMG: Sweeps down         1
    HYPRE BoomerAMG: Sweeps up           1
    HYPRE BoomerAMG: Sweeps on coarse    1
    HYPRE BoomerAMG: Relax down          symmetric-SOR/Jacobi
    HYPRE BoomerAMG: Relax up            symmetric-SOR/Jacobi
    HYPRE BoomerAMG: Relax on coarse     Gaussian-elimination
    HYPRE BoomerAMG: Relax weight  (all)      1
    HYPRE BoomerAMG: Outer relax weight (all) 1
    HYPRE BoomerAMG: Using CF-relaxation
    HYPRE BoomerAMG: Measure type        local
    HYPRE BoomerAMG: Coarsen type        Falgout
    HYPRE BoomerAMG: Interpolation type  classical
Preconditioner (PC) options -------------------------------------------------
  -fieldsplit_field2_pc_type <ilu>: Preconditioner (one of) none jacobi pbjacobi bjacobi sor lu shell mg
      eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic hmpi asa cp lsc redistribute svd gamg ml spai hypre pfmg syspfmg tfs python (PCSetType)
  -fieldsplit_field2_pc_use_amat: <FALSE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  HYPRE preconditioner options
  -fieldsplit_field2_pc_hypre_type <boomeramg> (choose one of) pilut parasails boomeramg euclid (PCHYPRESetType)
  HYPRE BoomerAMG Options
  -fieldsplit_field2_pc_hypre_boomeramg_cycle_type <V> (choose one of) V W (None)
  -fieldsplit_field2_pc_hypre_boomeramg_max_levels <25>: Number of levels (of grids) allowed (None)
  -fieldsplit_field2_pc_hypre_boomeramg_max_iter <1>: Maximum iterations used PER hypre call (None)
  -fieldsplit_field2_pc_hypre_boomeramg_tol <0>: Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations) (None)
  -fieldsplit_field2_pc_hypre_boomeramg_truncfactor <0>: Truncation factor for interpolation (0=no truncation) (None)
  -fieldsplit_field2_pc_hypre_boomeramg_P_max <0>: Max elements per row for interpolation operator (0=unlimited) (None)
  -fieldsplit_field2_pc_hypre_boomeramg_agg_nl <0>: Number of levels of aggressive coarsening (None)
  -fieldsplit_field2_pc_hypre_boomeramg_agg_num_paths <1>: Number of paths for aggressive coarsening (None)
  -fieldsplit_field2_pc_hypre_boomeramg_strong_threshold <0.25>: Threshold for being strongly connected (None)
  -fieldsplit_field2_pc_hypre_boomeramg_max_row_sum <0.9>: Maximum row sum (None)
  -fieldsplit_field2_pc_hypre_boomeramg_grid_sweeps_all <1>: Number of sweeps for the up and down grid levels (None)
  -fieldsplit_field2_pc_hypre_boomeramg_grid_sweeps_down <1>: Number of sweeps for the down cycles (None)
  -fieldsplit_field2_pc_hypre_boomeramg_grid_sweeps_up <1>: Number of sweeps for the up cycles (None)
  -fieldsplit_field2_pc_hypre_boomeramg_grid_sweeps_coarse <1>: Number of sweeps for the coarse level (None)
  -fieldsplit_field2_pc_hypre_boomeramg_relax_type_all <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -fieldsplit_field2_pc_hypre_boomeramg_relax_type_down <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -fieldsplit_field2_pc_hypre_boomeramg_relax_type_up <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -fieldsplit_field2_pc_hypre_boomeramg_relax_type_coarse <Gaussian-elimination> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -fieldsplit_field2_pc_hypre_boomeramg_relax_weight_all <1>: Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps) (None)
  -fieldsplit_field2_pc_hypre_boomeramg_relax_weight_level <1>: Set the relaxation weight for a particular level (weight,level) (None)
  -fieldsplit_field2_pc_hypre_boomeramg_outer_relax_weight_all <1>: Outer relaxation weight for all levels (-k = determined with k CG steps) (None)
  -fieldsplit_field2_pc_hypre_boomeramg_outer_relax_weight_level <1>: Set the outer relaxation weight for a particular level (weight,level) (None)
  -fieldsplit_field2_pc_hypre_boomeramg_no_CF: <FALSE> Do not use CF-relaxation (None)
  -fieldsplit_field2_pc_hypre_boomeramg_measure_type <local> (choose one of) local global (None)
  -fieldsplit_field2_pc_hypre_boomeramg_coarsen_type <Falgout> (choose one of) CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None)
  -fieldsplit_field2_pc_hypre_boomeramg_interp_type <classical> (choose one of) classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1 (None)
  -fieldsplit_field2_pc_hypre_boomeramg_print_statistics: Print statistics (None)
  -fieldsplit_field2_pc_hypre_boomeramg_print_debug: Print debug information (None)
  -fieldsplit_field2_pc_hypre_boomeramg_nodal_coarsen: <FALSE> HYPRE_BoomerAMGSetNodal() (None)
  -fieldsplit_field2_pc_hypre_boomeramg_nodal_relaxation: <FALSE> Nodal relaxation via Schwarz (None)
Krylov Method (KSP) options -------------------------------------------------
  -fieldsplit_field2_ksp_type <preonly>: Krylov method (one of) cg groppcg pipecg cgne nash stcg gltr richardson
      chebyshev gmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl cgs tfqmr cr pipecr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd gcr pgmres specest dgmres python (KSPSetType)
  -fieldsplit_field2_ksp_max_it <10000>: Maximum number of iterations (KSPSetTolerances)
  -fieldsplit_field2_ksp_rtol <1e-05>: Relative decrease in residual norm (KSPSetTolerances)
  -fieldsplit_field2_ksp_atol <1e-50>: Absolute value of residual norm (KSPSetTolerances)
  -fieldsplit_field2_ksp_divtol <10000>: Residual norm increase cause divergence (KSPSetTolerances)
  -fieldsplit_field2_ksp_converged_use_initial_residual_norm: <FALSE> Use initial residual residual norm for computing relative convergence (KSPDefaultConvergedSetUIRNorm)
  -fieldsplit_field2_ksp_converged_use_min_initial_residual_norm: <FALSE> Use minimum of initial residual norm and b for computing relative convergence (KSPDefaultConvergedSetUMIRNorm)
  -fieldsplit_field2_ksp_initial_guess_nonzero: <FALSE> Use the contents of the solution vector for initial guess (KSPSetInitialNonzero)
  -fieldsplit_field2_ksp_knoll: <FALSE> Use preconditioner applied to b for initial guess (KSPSetInitialGuessKnoll)
  -fieldsplit_field2_ksp_error_if_not_converged: <FALSE> Generate error if solver does not converge (KSPSetErrorIfNotConverged)
  -fieldsplit_field2_ksp_fischer_guess <0>: Use Paul Fischer's algorithm for initial guess (KSPSetUseFischerGuess)
  -fieldsplit_field2_ksp_convergence_test <default> (choose one of) default skip (KSPSetConvergenceTest)
  -fieldsplit_field2_ksp_norm_type <PRECONDITIONED> (choose one of) NONE PRECONDITIONED UNPRECONDITIONED NATURAL (KSPSetNormType)
  -fieldsplit_field2_ksp_check_norm_iteration <-1>: First iteration to compute residual norm (KSPSetCheckNormIteration)
  -fieldsplit_field2_ksp_lag_norm: <FALSE> Lag the calculation of the residual norm (KSPSetLagNorm)
  -fieldsplit_field2_ksp_diagonal_scale: <FALSE> Diagonal scale matrix before building preconditioner (KSPSetDiagonalScale)
  -fieldsplit_field2_ksp_diagonal_scale_fix: <FALSE> Fix diagonally scaled matrix after solve (KSPSetDiagonalScaleFix)
  -fieldsplit_field2_ksp_constant_null_space: <FALSE> Add constant null space to Krylov solver (KSPSetNullSpace)
  -fieldsplit_field2_ksp_converged_reason: <FALSE> Print reason for converged or diverged (KSPSolve)
  -fieldsplit_field2_ksp_monitor_cancel: <FALSE> Remove any hardwired monitor routines (KSPMonitorCancel)
  -fieldsplit_field2_ksp_monitor <stdout>: Monitor preconditioned residual norm (KSPMonitorSet)
  -fieldsplit_field2_ksp_monitor_range <stdout>: Monitor percent of residual entries more than 10 percent of max (KSPMonitorRange)
  -fieldsplit_field2_ksp_monitor_solution: <FALSE> Monitor solution graphically (KSPMonitorSet)
  -fieldsplit_field2_ksp_monitor_true_residual <stdout>: Monitor true residual norm (KSPMonitorSet)
  -fieldsplit_field2_ksp_monitor_max <stdout>: Monitor true residual max norm (KSPMonitorSet)
  -fieldsplit_field2_ksp_monitor_singular_value <stdout>: Monitor singular values (KSPMonitorSet)
  -fieldsplit_field2_ksp_monitor_short <stdout>: Monitor preconditioned residual norm with fewer digits (KSPMonitorSet)
  -fieldsplit_field2_ksp_monitor_python <(null)>: Use Python function (KSPMonitorSet)
  -fieldsplit_field2_ksp_monitor_lg_residualnorm: <FALSE> Monitor graphically preconditioned residual norm (KSPMonitorSet)
  -fieldsplit_field2_ksp_monitor_lg_true_residualnorm: <FALSE> Monitor graphically true residual norm (KSPMonitorSet)
  -fieldsplit_field2_ksp_monitor_lg_range: <FALSE> Monitor graphically range of preconditioned residual norm (KSPMonitorSet)
  -fieldsplit_field2_ksp_pc_side <LEFT> (choose one of) LEFT RIGHT SYMMETRIC (KSPSetPCSide)
  -fieldsplit_field2_ksp_compute_singularvalues: <FALSE> Compute singular values of preconditioned operator (KSPSetComputeSingularValues)
  -fieldsplit_field2_ksp_compute_eigenvalues: <FALSE> Compute eigenvalues of preconditioned operator (KSPSetComputeSingularValues)
  -fieldsplit_field2_ksp_plot_eigenvalues: <FALSE> Scatter plot extreme eigenvalues (KSPSetComputeSingularValues)
  KSP CG and CGNE options
  -fieldsplit_field2_ksp_cg_single_reduction: <FALSE> Merge inner products into single MPI_Allreduce() (KSPCGUseSingleReduction)
  -fieldsplit_field2_ksp_view: View linear solver parameters (KSPView)
KSP Object:(fieldsplit_field2_) 1 MPI processes
  type: cg
  maximum iterations=10000, initial guess is zero
  tolerances:  relative=1e-08, absolute=1e-50, divergence=10000
  left preconditioning
  using DEFAULT norm type for convergence test
PC Object:(fieldsplit_field2_) 1 MPI processes
  type: hypre
  PC has not been set up so information may be incomplete
    HYPRE BoomerAMG preconditioning
    HYPRE BoomerAMG: Cycle type V
    HYPRE BoomerAMG: Maximum number of levels 25
    HYPRE BoomerAMG: Maximum number of iterations PER hypre call 1
    HYPRE BoomerAMG: Convergence tolerance PER hypre call 0
    HYPRE BoomerAMG: Threshold for strong coupling 0.5
    HYPRE BoomerAMG: Interpolation truncation factor 0
    HYPRE BoomerAMG: Interpolation: max elements per row 0
    HYPRE BoomerAMG: Number of levels of aggressive coarsening 0
    HYPRE BoomerAMG: Number of paths for aggressive coarsening 1
    HYPRE BoomerAMG: Maximum row sums 0.9
    HYPRE BoomerAMG: Sweeps down         1
    HYPRE BoomerAMG: Sweeps up           1
    HYPRE BoomerAMG: Sweeps on coarse    1
    HYPRE BoomerAMG: Relax down          symmetric-SOR/Jacobi
    HYPRE BoomerAMG: Relax up            symmetric-SOR/Jacobi
    HYPRE BoomerAMG: Relax on coarse     Gaussian-elimination
    HYPRE BoomerAMG: Relax weight  (all)      1
    HYPRE BoomerAMG: Outer relax weight (all) 1
    HYPRE BoomerAMG: Using CF-relaxation
    HYPRE BoomerAMG: Measure type        local
    HYPRE BoomerAMG: Coarsen type        Falgout
    HYPRE BoomerAMG: Interpolation type  classical
0
 time to create petsc field split preconditioner 0.00337815284729 


Options for SEQAIJ matrix -------------------------------------------------
  -mat_no_unroll: <FALSE> Do not optimize for inodes (slower) (None)
  -mat_no_inode: <FALSE> Do not optimize for inodes -slower- (None)
  -mat_inode_limit <5>: Do not use inodes larger then this value (None)
Options for SEQAIJ matrix -------------------------------------------------
  -mat_no_unroll: <FALSE> Do not optimize for inodes (slower) (None)
  -mat_no_inode: <FALSE> Do not optimize for inodes -slower- (None)
  -mat_inode_limit <5>: Do not use inodes larger then this value (None)
  0 KSP Residual norm 1.518006071699e+00 
  1 KSP Residual norm 2.275364759288e-02 
  2 KSP Residual norm 1.023625593561e-05 
  3 KSP Residual norm 3.353007231508e-07 
KSP Object: 1 MPI processes
  type: minres
  maximum iterations=10000, initial guess is zero
  tolerances:  relative=1e-06, absolute=1e-50, divergence=10000
  left preconditioning
  using PRECONDITIONED norm type for convergence test
PC Object: 1 MPI processes
  type: fieldsplit
    FieldSplit with ADDITIVE composition: total splits = 2
    Solver info for each split is in the following KSP objects:
    Split number 0 Defined by IS
    KSP Object:    (fieldsplit_field1_)     1 MPI processes
      type: cg
      maximum iterations=10000, initial guess is zero
      tolerances:  relative=1e-08, absolute=1e-50, divergence=10000
      left preconditioning
      using PRECONDITIONED norm type for convergence test
    PC Object:    (fieldsplit_field1_)     1 MPI processes
      type: hypre
        HYPRE BoomerAMG preconditioning
        HYPRE BoomerAMG: Cycle type V
        HYPRE BoomerAMG: Maximum number of levels 25
        HYPRE BoomerAMG: Maximum number of iterations PER hypre call 1
        HYPRE BoomerAMG: Convergence tolerance PER hypre call 0
        HYPRE BoomerAMG: Threshold for strong coupling 0.5
        HYPRE BoomerAMG: Interpolation truncation factor 0
        HYPRE BoomerAMG: Interpolation: max elements per row 0
        HYPRE BoomerAMG: Number of levels of aggressive coarsening 0
        HYPRE BoomerAMG: Number of paths for aggressive coarsening 1
        HYPRE BoomerAMG: Maximum row sums 0.9
        HYPRE BoomerAMG: Sweeps down         1
        HYPRE BoomerAMG: Sweeps up           1
        HYPRE BoomerAMG: Sweeps on coarse    1
        HYPRE BoomerAMG: Relax down          symmetric-SOR/Jacobi
        HYPRE BoomerAMG: Relax up            symmetric-SOR/Jacobi
        HYPRE BoomerAMG: Relax on coarse     Gaussian-elimination
        HYPRE BoomerAMG: Relax weight  (all)      1
        HYPRE BoomerAMG: Outer relax weight (all) 1
        HYPRE BoomerAMG: Using CF-relaxation
        HYPRE BoomerAMG: Measure type        local
        HYPRE BoomerAMG: Coarsen type        Falgout
        HYPRE BoomerAMG: Interpolation type  classical
      linear system matrix = precond matrix:
      Matrix Object:       1 MPI processes
        type: seqaij
        rows=2624, cols=2624
        total: nonzeros=29824, allocated nonzeros=29824
        total number of mallocs used during MatSetValues calls =0
          using I-node routines: found 1311 nodes, limit used is 5
    Split number 1 Defined by IS
    KSP Object:    (fieldsplit_field2_)     1 MPI processes
      type: cg
      maximum iterations=10000, initial guess is zero
      tolerances:  relative=1e-08, absolute=1e-50, divergence=10000
      left preconditioning
      using PRECONDITIONED norm type for convergence test
    PC Object:    (fieldsplit_field2_)     1 MPI processes
      type: hypre
        HYPRE BoomerAMG preconditioning
        HYPRE BoomerAMG: Cycle type V
        HYPRE BoomerAMG: Maximum number of levels 25
        HYPRE BoomerAMG: Maximum number of iterations PER hypre call 1
        HYPRE BoomerAMG: Convergence tolerance PER hypre call 0
        HYPRE BoomerAMG: Threshold for strong coupling 0.5
        HYPRE BoomerAMG: Interpolation truncation factor 0
        HYPRE BoomerAMG: Interpolation: max elements per row 0
        HYPRE BoomerAMG: Number of levels of aggressive coarsening 0
        HYPRE BoomerAMG: Number of paths for aggressive coarsening 1
        HYPRE BoomerAMG: Maximum row sums 0.9
        HYPRE BoomerAMG: Sweeps down         1
        HYPRE BoomerAMG: Sweeps up           1
        HYPRE BoomerAMG: Sweeps on coarse    1
        HYPRE BoomerAMG: Relax down          symmetric-SOR/Jacobi
        HYPRE BoomerAMG: Relax up            symmetric-SOR/Jacobi
        HYPRE BoomerAMG: Relax on coarse     Gaussian-elimination
        HYPRE BoomerAMG: Relax weight  (all)      1
        HYPRE BoomerAMG: Outer relax weight (all) 1
        HYPRE BoomerAMG: Using CF-relaxation
        HYPRE BoomerAMG: Measure type        local
        HYPRE BoomerAMG: Coarsen type        Falgout
        HYPRE BoomerAMG: Interpolation type  classical
      linear system matrix = precond matrix:
      Matrix Object:       1 MPI processes
        type: seqaij
        rows=289, cols=289
        total: nonzeros=1889, allocated nonzeros=1889
        total number of mallocs used during MatSetValues calls =0
          not using I-node routines
  linear system matrix followed by preconditioner matrix:
  Matrix Object:   1 MPI processes
    type: seqaij
    rows=2913, cols=2913
    total: nonzeros=50401, allocated nonzeros=50401
    total number of mallocs used during MatSetValues calls =0
      using I-node routines: found 1600 nodes, limit used is 5
  Matrix Object:   1 MPI processes
    type: seqaij
    rows=2913, cols=2913
    total: nonzeros=50401, allocated nonzeros=50401
    total number of mallocs used during MatSetValues calls =0
      using I-node routines: found 1600 nodes, limit used is 5
time to solve:  [ 0.96294999]
369


outer iterations =  [ 3.]
Inner itations, field 1 =  369  field 2 =  4
None
4
369
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
*** Warning: Form::coloring does not properly consider form type.
Coloring mesh.
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
*** Warning: Form::coloring does not properly consider form type.
[ 0.06569425]
[  1.58074718e-08]



          ===============================
                  Results Table
          ===============================


   Total DoF  V DoF  Q DoF  # iters  Soln Time  V-L2  V-order      P-L2  P-order
0       2913   2624    289        3       0.96  0.07        0  1.58e-08        0



Velocity Elements rate of convergence  nan
Pressure Elements rate of convergence  nan
\begin{tabular}{lrrrrrrrrr}
\toprule
{} &  Total DoF &  V DoF &  Q DoF &  # iters &  Soln Time &  V-L2 &  V-order &      P-L2 &  P-order \\
\midrule
0 &       2913 &   2624 &    289 &        3 &       0.96 &  0.07 &        0 &  1.58e-08 &        0 \\
\bottomrule
\end{tabular}

Saving plot to file: dolfin_plot_0.png
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
Vector (Vec) options -------------------------------------------------
  -vec_type <seq>: Vector type (one of) seq mpi standard shared (VecSetType)
  -vec_view <>: Display vector with the viewer on VecAssemblyEnd() (VecView)
