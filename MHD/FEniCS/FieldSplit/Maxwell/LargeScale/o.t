1


V:   [ 48.] Q:   [ 25.] W:   [ 73.] 


0.0187969207764
Preconditioner (PC) options -------------------------------------------------
  -pc_type <hypre>: Preconditioner (one of) none jacobi pbjacobi bjacobi sor lu shell mg
      eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic hmpi asa cp lsc redistribute svd gamg ml hypre pfmg syspfmg tfs python (PCSetType)
  -pc_use_amat: <FALSE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  HYPRE preconditioner options
  -pc_hypre_type <boomeramg> (choose one of) pilut parasails boomeramg euclid (PCHYPRESetType)
  HYPRE BoomerAMG Options
  -pc_hypre_boomeramg_cycle_type <V> (choose one of) V W (None)
  -pc_hypre_boomeramg_max_levels <25>: Number of levels (of grids) allowed (None)
  -pc_hypre_boomeramg_max_iter <1>: Maximum iterations used PER hypre call (None)
  -pc_hypre_boomeramg_tol <0>: Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations) (None)
  -pc_hypre_boomeramg_truncfactor <0>: Truncation factor for interpolation (0=no truncation) (None)
  -pc_hypre_boomeramg_P_max <0>: Max elements per row for interpolation operator (0=unlimited) (None)
  -pc_hypre_boomeramg_agg_nl <0>: Number of levels of aggressive coarsening (None)
  -pc_hypre_boomeramg_agg_num_paths <1>: Number of paths for aggressive coarsening (None)
  -pc_hypre_boomeramg_strong_threshold <0.25>: Threshold for being strongly connected (None)
  -pc_hypre_boomeramg_max_row_sum <0.9>: Maximum row sum (None)
  -pc_hypre_boomeramg_grid_sweeps_all <1>: Number of sweeps for the up and down grid levels (None)
  -pc_hypre_boomeramg_grid_sweeps_down <1>: Number of sweeps for the down cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_up <1>: Number of sweeps for the up cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_coarse <1>: Number of sweeps for the coarse level (None)
  -pc_hypre_boomeramg_relax_type_all <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_down <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_up <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_coarse <Gaussian-elimination> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_weight_all <1>: Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_relax_weight_level <1>: Set the relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_outer_relax_weight_all <1>: Outer relaxation weight for all levels (-k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_outer_relax_weight_level <1>: Set the outer relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_no_CF: <FALSE> Do not use CF-relaxation (None)
  -pc_hypre_boomeramg_measure_type <local> (choose one of) local global (None)
  -pc_hypre_boomeramg_coarsen_type <Falgout> (choose one of) CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None)
  -pc_hypre_boomeramg_interp_type <classical> (choose one of) classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1 (None)
  -pc_hypre_boomeramg_print_statistics: Print statistics (None)
  -pc_hypre_boomeramg_print_debug: Print debug information (None)
  -pc_hypre_boomeramg_nodal_coarsen: <FALSE> HYPRE_BoomerAMGSetNodal() (None)
  -pc_hypre_boomeramg_nodal_relaxation: <FALSE> Nodal relaxation via Schwarz (None)
Krylov Method (KSP) options -------------------------------------------------
  -ksp_type <cg>: Krylov method (one of) cg groppcg pipecg cgne nash stcg gltr richardson
      chebyshev gmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl cgs tfqmr cr pipecr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd gcr pgmres specest dgmres python (KSPSetType)
  -ksp_max_it <10000>: Maximum number of iterations (KSPSetTolerances)
  -ksp_rtol <1e-05>: Relative decrease in residual norm (KSPSetTolerances)
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
  KSP CG and CGNE options
  -ksp_cg_single_reduction: <FALSE> Merge inner products into single MPI_Allreduce() (KSPCGUseSingleReduction)
  -ksp_view: View linear solver parameters (KSPView)
Preconditioner (PC) options -------------------------------------------------
  -pc_type <hypre>: Preconditioner (one of) none jacobi pbjacobi bjacobi sor lu shell mg
      eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic hmpi asa cp lsc redistribute svd gamg ml hypre pfmg syspfmg tfs python (PCSetType)
  -pc_use_amat: <FALSE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  HYPRE preconditioner options
  -pc_hypre_type <boomeramg> (choose one of) pilut parasails boomeramg euclid (PCHYPRESetType)
  HYPRE BoomerAMG Options
  -pc_hypre_boomeramg_cycle_type <V> (choose one of) V W (None)
  -pc_hypre_boomeramg_max_levels <25>: Number of levels (of grids) allowed (None)
  -pc_hypre_boomeramg_max_iter <1>: Maximum iterations used PER hypre call (None)
  -pc_hypre_boomeramg_tol <0>: Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations) (None)
  -pc_hypre_boomeramg_truncfactor <0>: Truncation factor for interpolation (0=no truncation) (None)
  -pc_hypre_boomeramg_P_max <0>: Max elements per row for interpolation operator (0=unlimited) (None)
  -pc_hypre_boomeramg_agg_nl <0>: Number of levels of aggressive coarsening (None)
  -pc_hypre_boomeramg_agg_num_paths <1>: Number of paths for aggressive coarsening (None)
  -pc_hypre_boomeramg_strong_threshold <0.25>: Threshold for being strongly connected (None)
  -pc_hypre_boomeramg_max_row_sum <0.9>: Maximum row sum (None)
  -pc_hypre_boomeramg_grid_sweeps_all <1>: Number of sweeps for the up and down grid levels (None)
  -pc_hypre_boomeramg_grid_sweeps_down <1>: Number of sweeps for the down cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_up <1>: Number of sweeps for the up cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_coarse <1>: Number of sweeps for the coarse level (None)
  -pc_hypre_boomeramg_relax_type_all <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_down <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_up <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_coarse <Gaussian-elimination> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_weight_all <1>: Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_relax_weight_level <1>: Set the relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_outer_relax_weight_all <1>: Outer relaxation weight for all levels (-k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_outer_relax_weight_level <1>: Set the outer relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_no_CF: <FALSE> Do not use CF-relaxation (None)
  -pc_hypre_boomeramg_measure_type <local> (choose one of) local global (None)
  -pc_hypre_boomeramg_coarsen_type <Falgout> (choose one of) CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None)
  -pc_hypre_boomeramg_interp_type <classical> (choose one of) classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1 (None)
  -pc_hypre_boomeramg_print_statistics: Print statistics (None)
  -pc_hypre_boomeramg_print_debug: Print debug information (None)
  -pc_hypre_boomeramg_nodal_coarsen: <FALSE> HYPRE_BoomerAMGSetNodal() (None)
  -pc_hypre_boomeramg_nodal_relaxation: <FALSE> Nodal relaxation via Schwarz (None)
Krylov Method (KSP) options -------------------------------------------------
  -ksp_type <richardson>: Krylov method (one of) cg groppcg pipecg cgne nash stcg gltr richardson
      chebyshev gmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl cgs tfqmr cr pipecr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd gcr pgmres specest dgmres python (KSPSetType)
  -ksp_max_it <1>: Maximum number of iterations (KSPSetTolerances)
  -ksp_rtol <1e-05>: Relative decrease in residual norm (KSPSetTolerances)
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
  KSP Richardson Options
  -ksp_richardson_scale <1>: damping factor (KSPRichardsonSetScale)
  -ksp_richardson_self_scale: <FALSE> dynamically determine optimal damping factor (KSPRichardsonSetSelfScale)
  -ksp_view: View linear solver parameters (KSPView)
Options for SEQAIJ matrix -------------------------------------------------
  -mat_no_unroll: <FALSE> Do not optimize for inodes (slower) (None)
  -mat_no_inode: <FALSE> Do not optimize for inodes -slower- (None)
  -mat_inode_limit <5>: Do not use inodes larger then this value (None)
Options for SEQAIJ matrix -------------------------------------------------
  -mat_no_unroll: <FALSE> Do not optimize for inodes (slower) (None)
  -mat_no_inode: <FALSE> Do not optimize for inodes -slower- (None)
  -mat_inode_limit <5>: Do not use inodes larger then this value (None)
18
18
19
18
19
19
18
19
20
19
20
19
20
20
20
20
20
20
20
20
20
20
19
20
19
20
19
20
19
20
19
20
19
20
18
20
20
20
20
20
20
20
20
20
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
18
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
18
20
19
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
18
20
19
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
17
20
19
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
18
20
19
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
19
20
18
20
19
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
20
time to solve:  [ 0.25116205]
error norm = 1.31767e-05
2


V:   [ 176.] Q:   [ 81.] W:   [ 257.] 


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
0.0119700431824
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
Preconditioner (PC) options -------------------------------------------------
  -pc_type <ilu>: Preconditioner (one of) none jacobi pbjacobi bjacobi sor lu shell mg
      eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic hmpi asa cp lsc redistribute svd gamg ml hypre pfmg syspfmg tfs python (PCSetType)
  -pc_use_amat: <FALSE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  ILU Options
  -pc_factor_in_place: <FALSE> Form factored matrix in the same memory as the matrix (PCFactorSetUseInPlace)
  -pc_factor_fill <1>: Expected non-zeros in factored matrix (PCFactorSetFill)
  -pc_factor_shift_type <INBLOCKS> (choose one of) NONE NONZERO POSITIVE_DEFINITE INBLOCKS (PCFactorSetShiftType)
  -pc_factor_shift_amount <2.22045e-14>: Shift added to diagonal (PCFactorSetShiftAmount)
  -pc_factor_zeropivot <2.22045e-14>: Pivot is considered zero if less than (PCFactorSetZeroPivot)
  -pc_factor_column_pivot <-2>: Column pivot tolerance (used only for some factorization) (PCFactorSetColumnPivot)
  -pc_factor_pivot_in_blocks: <TRUE> Pivot inside matrix dense blocks for BAIJ and SBAIJ (PCFactorSetPivotInBlocks)
  -pc_factor_reuse_fill: <FALSE> Use fill from previous factorization (PCFactorSetReuseFill)
  -pc_factor_reuse_ordering: <FALSE> Reuse ordering from previous factorization (PCFactorSetReuseOrdering)
  -pc_factor_mat_ordering_type <natural>: Reordering to reduce nonzeros in factored matrix (one of) natural nd 1wd rcm qmd rowlength amd (PCFactorSetMatOrderingType)
  -pc_factor_mat_solver_package <petsc>: Specific direct solver to use (MatGetFactor)
  -pc_factor_levels <0>: levels of fill (PCFactorSetLevels)
  -pc_factor_diagonal_fill: <FALSE> Allow fill into empty diagonal entry (PCFactorSetAllowDiagonalFill)
  -pc_factor_nonzeros_along_diagonal: Reorder to remove zeros from diagonal (PCFactorReorderForNonzeroDiagonal)
Krylov Method (KSP) options -------------------------------------------------
  -ksp_type <gmres>: Krylov method (one of) cg groppcg pipecg cgne nash stcg gltr richardson
      chebyshev gmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl cgs tfqmr cr pipecr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd gcr pgmres specest dgmres python (KSPSetType)
  -ksp_max_it <10000>: Maximum number of iterations (KSPSetTolerances)
  -ksp_rtol <1e-05>: Relative decrease in residual norm (KSPSetTolerances)
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
  KSP GMRES Options
  -ksp_gmres_restart <30>: Number of Krylov search directions (KSPGMRESSetRestart)
  -ksp_gmres_haptol <1e-30>: Tolerance for exact convergence (happy ending) (KSPGMRESSetHapTol)
  -ksp_gmres_preallocate: <FALSE> Preallocate Krylov vectors (KSPGMRESSetPreAllocateVectors)
  Pick at most one of -------------
    -ksp_gmres_classicalgramschmidt: Classical (unmodified) Gram-Schmidt (fast) (KSPGMRESSetOrthogonalization)
    -ksp_gmres_modifiedgramschmidt: Modified Gram-Schmidt (slow,more stable) (KSPGMRESSetOrthogonalization)
  -ksp_gmres_cgs_refinement_type <REFINE_NEVER> (choose one of) REFINE_NEVER REFINE_IFNEEDED REFINE_ALWAYS (KSPGMRESSetCGSRefinementType)
  -ksp_gmres_krylov_monitor: <FALSE> Plot the Krylov directions (KSPMonitorSet)
  -ksp_view: View linear solver parameters (KSPView)
Preconditioner (PC) options -------------------------------------------------
  -pc_type <hypre>: Preconditioner (one of) none jacobi pbjacobi bjacobi sor lu shell mg
      eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic hmpi asa cp lsc redistribute svd gamg ml hypre pfmg syspfmg tfs python (PCSetType)
  -pc_use_amat: <FALSE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  HYPRE preconditioner options
  -pc_hypre_type <boomeramg> (choose one of) pilut parasails boomeramg euclid (PCHYPRESetType)
  HYPRE BoomerAMG Options
  -pc_hypre_boomeramg_cycle_type <V> (choose one of) V W (None)
  -pc_hypre_boomeramg_max_levels <25>: Number of levels (of grids) allowed (None)
  -pc_hypre_boomeramg_max_iter <1>: Maximum iterations used PER hypre call (None)
  -pc_hypre_boomeramg_tol <0>: Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations) (None)
  -pc_hypre_boomeramg_truncfactor <0>: Truncation factor for interpolation (0=no truncation) (None)
  -pc_hypre_boomeramg_P_max <0>: Max elements per row for interpolation operator (0=unlimited) (None)
  -pc_hypre_boomeramg_agg_nl <0>: Number of levels of aggressive coarsening (None)
  -pc_hypre_boomeramg_agg_num_paths <1>: Number of paths for aggressive coarsening (None)
  -pc_hypre_boomeramg_strong_threshold <0.25>: Threshold for being strongly connected (None)
  -pc_hypre_boomeramg_max_row_sum <0.9>: Maximum row sum (None)
  -pc_hypre_boomeramg_grid_sweeps_all <1>: Number of sweeps for the up and down grid levels (None)
  -pc_hypre_boomeramg_grid_sweeps_down <1>: Number of sweeps for the down cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_up <1>: Number of sweeps for the up cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_coarse <1>: Number of sweeps for the coarse level (None)
  -pc_hypre_boomeramg_relax_type_all <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_down <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_up <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_coarse <Gaussian-elimination> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_weight_all <1>: Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_relax_weight_level <1>: Set the relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_outer_relax_weight_all <1>: Outer relaxation weight for all levels (-k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_outer_relax_weight_level <1>: Set the outer relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_no_CF: <FALSE> Do not use CF-relaxation (None)
  -pc_hypre_boomeramg_measure_type <local> (choose one of) local global (None)
  -pc_hypre_boomeramg_coarsen_type <Falgout> (choose one of) CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None)
  -pc_hypre_boomeramg_interp_type <classical> (choose one of) classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1 (None)
  -pc_hypre_boomeramg_print_statistics: Print statistics (None)
  -pc_hypre_boomeramg_print_debug: Print debug information (None)
  -pc_hypre_boomeramg_nodal_coarsen: <FALSE> HYPRE_BoomerAMGSetNodal() (None)
  -pc_hypre_boomeramg_nodal_relaxation: <FALSE> Nodal relaxation via Schwarz (None)
Krylov Method (KSP) options -------------------------------------------------
  -ksp_type <cg>: Krylov method (one of) cg groppcg pipecg cgne nash stcg gltr richardson
      chebyshev gmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl cgs tfqmr cr pipecr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd gcr pgmres specest dgmres python (KSPSetType)
  -ksp_max_it <10000>: Maximum number of iterations (KSPSetTolerances)
  -ksp_rtol <1e-05>: Relative decrease in residual norm (KSPSetTolerances)
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
  KSP CG and CGNE options
  -ksp_cg_single_reduction: <FALSE> Merge inner products into single MPI_Allreduce() (KSPCGUseSingleReduction)
  -ksp_view: View linear solver parameters (KSPView)
Preconditioner (PC) options -------------------------------------------------
  -pc_type <hypre>: Preconditioner (one of) none jacobi pbjacobi bjacobi sor lu shell mg
      eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic hmpi asa cp lsc redistribute svd gamg ml hypre pfmg syspfmg tfs python (PCSetType)
  -pc_use_amat: <FALSE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  HYPRE preconditioner options
  -pc_hypre_type <boomeramg> (choose one of) pilut parasails boomeramg euclid (PCHYPRESetType)
  HYPRE BoomerAMG Options
  -pc_hypre_boomeramg_cycle_type <V> (choose one of) V W (None)
  -pc_hypre_boomeramg_max_levels <25>: Number of levels (of grids) allowed (None)
  -pc_hypre_boomeramg_max_iter <1>: Maximum iterations used PER hypre call (None)
  -pc_hypre_boomeramg_tol <0>: Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations) (None)
  -pc_hypre_boomeramg_truncfactor <0>: Truncation factor for interpolation (0=no truncation) (None)
  -pc_hypre_boomeramg_P_max <0>: Max elements per row for interpolation operator (0=unlimited) (None)
  -pc_hypre_boomeramg_agg_nl <0>: Number of levels of aggressive coarsening (None)
  -pc_hypre_boomeramg_agg_num_paths <1>: Number of paths for aggressive coarsening (None)
  -pc_hypre_boomeramg_strong_threshold <0.25>: Threshold for being strongly connected (None)
  -pc_hypre_boomeramg_max_row_sum <0.9>: Maximum row sum (None)
  -pc_hypre_boomeramg_grid_sweeps_all <1>: Number of sweeps for the up and down grid levels (None)
  -pc_hypre_boomeramg_grid_sweeps_down <1>: Number of sweeps for the down cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_up <1>: Number of sweeps for the up cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_coarse <1>: Number of sweeps for the coarse level (None)
  -pc_hypre_boomeramg_relax_type_all <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_down <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_up <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_coarse <Gaussian-elimination> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_weight_all <1>: Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_relax_weight_level <1>: Set the relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_outer_relax_weight_all <1>: Outer relaxation weight for all levels (-k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_outer_relax_weight_level <1>: Set the outer relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_no_CF: <FALSE> Do not use CF-relaxation (None)
  -pc_hypre_boomeramg_measure_type <local> (choose one of) local global (None)
  -pc_hypre_boomeramg_coarsen_type <Falgout> (choose one of) CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None)
  -pc_hypre_boomeramg_interp_type <classical> (choose one of) classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1 (None)
  -pc_hypre_boomeramg_print_statistics: Print statistics (None)
  -pc_hypre_boomeramg_print_debug: Print debug information (None)
  -pc_hypre_boomeramg_nodal_coarsen: <FALSE> HYPRE_BoomerAMGSetNodal() (None)
  -pc_hypre_boomeramg_nodal_relaxation: <FALSE> Nodal relaxation via Schwarz (None)
Krylov Method (KSP) options -------------------------------------------------
  -ksp_type <richardson>: Krylov method (one of) cg groppcg pipecg cgne nash stcg gltr richardson
      chebyshev gmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl cgs tfqmr cr pipecr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd gcr pgmres specest dgmres python (KSPSetType)
  -ksp_max_it <1>: Maximum number of iterations (KSPSetTolerances)
  -ksp_rtol <1e-05>: Relative decrease in residual norm (KSPSetTolerances)
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
  KSP Richardson Options
  -ksp_richardson_scale <1>: damping factor (KSPRichardsonSetScale)
  -ksp_richardson_self_scale: <FALSE> dynamically determine optimal damping factor (KSPRichardsonSetSelfScale)
  -ksp_view: View linear solver parameters (KSPView)
Options for SEQAIJ matrix -------------------------------------------------
  -mat_no_unroll: <FALSE> Do not optimize for inodes (slower) (None)
  -mat_no_inode: <FALSE> Do not optimize for inodes -slower- (None)
  -mat_inode_limit <5>: Do not use inodes larger then this value (None)
Options for SEQAIJ matrix -------------------------------------------------
  -mat_no_unroll: <FALSE> Do not optimize for inodes (slower) (None)
  -mat_no_inode: <FALSE> Do not optimize for inodes -slower- (None)
  -mat_inode_limit <5>: Do not use inodes larger then this value (None)
52
48
65
48
61
52
62
55
56
55
62
56
59
59
59
56
60
55
56
59
56
59
56
55
59
59
55
59
56
59
56
60
59
59
59
56
59
59
55
59
55
60
55
59
59
55
59
59
59
59
56
59
55
59
58
59
59
60
55
58
55
59
55
59
56
56
55
55
55
59
59
58
59
59
time to solve:  [ 0.22528696]
error norm = 4.53052e-05
3


V:   [ 672.] Q:   [ 289.] W:   [ 961.] 


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
0.0175828933716
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
Preconditioner (PC) options -------------------------------------------------
  -pc_type <ilu>: Preconditioner (one of) none jacobi pbjacobi bjacobi sor lu shell mg
      eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic hmpi asa cp lsc redistribute svd gamg ml hypre pfmg syspfmg tfs python (PCSetType)
  -pc_use_amat: <FALSE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  ILU Options
  -pc_factor_in_place: <FALSE> Form factored matrix in the same memory as the matrix (PCFactorSetUseInPlace)
  -pc_factor_fill <1>: Expected non-zeros in factored matrix (PCFactorSetFill)
  -pc_factor_shift_type <INBLOCKS> (choose one of) NONE NONZERO POSITIVE_DEFINITE INBLOCKS (PCFactorSetShiftType)
  -pc_factor_shift_amount <2.22045e-14>: Shift added to diagonal (PCFactorSetShiftAmount)
  -pc_factor_zeropivot <2.22045e-14>: Pivot is considered zero if less than (PCFactorSetZeroPivot)
  -pc_factor_column_pivot <-2>: Column pivot tolerance (used only for some factorization) (PCFactorSetColumnPivot)
  -pc_factor_pivot_in_blocks: <TRUE> Pivot inside matrix dense blocks for BAIJ and SBAIJ (PCFactorSetPivotInBlocks)
  -pc_factor_reuse_fill: <FALSE> Use fill from previous factorization (PCFactorSetReuseFill)
  -pc_factor_reuse_ordering: <FALSE> Reuse ordering from previous factorization (PCFactorSetReuseOrdering)
  -pc_factor_mat_ordering_type <natural>: Reordering to reduce nonzeros in factored matrix (one of) natural nd 1wd rcm qmd rowlength amd (PCFactorSetMatOrderingType)
  -pc_factor_mat_solver_package <petsc>: Specific direct solver to use (MatGetFactor)
  -pc_factor_levels <0>: levels of fill (PCFactorSetLevels)
  -pc_factor_diagonal_fill: <FALSE> Allow fill into empty diagonal entry (PCFactorSetAllowDiagonalFill)
  -pc_factor_nonzeros_along_diagonal: Reorder to remove zeros from diagonal (PCFactorReorderForNonzeroDiagonal)
Krylov Method (KSP) options -------------------------------------------------
  -ksp_type <gmres>: Krylov method (one of) cg groppcg pipecg cgne nash stcg gltr richardson
      chebyshev gmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl cgs tfqmr cr pipecr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd gcr pgmres specest dgmres python (KSPSetType)
  -ksp_max_it <10000>: Maximum number of iterations (KSPSetTolerances)
  -ksp_rtol <1e-05>: Relative decrease in residual norm (KSPSetTolerances)
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
  KSP GMRES Options
  -ksp_gmres_restart <30>: Number of Krylov search directions (KSPGMRESSetRestart)
  -ksp_gmres_haptol <1e-30>: Tolerance for exact convergence (happy ending) (KSPGMRESSetHapTol)
  -ksp_gmres_preallocate: <FALSE> Preallocate Krylov vectors (KSPGMRESSetPreAllocateVectors)
  Pick at most one of -------------
    -ksp_gmres_classicalgramschmidt: Classical (unmodified) Gram-Schmidt (fast) (KSPGMRESSetOrthogonalization)
    -ksp_gmres_modifiedgramschmidt: Modified Gram-Schmidt (slow,more stable) (KSPGMRESSetOrthogonalization)
  -ksp_gmres_cgs_refinement_type <REFINE_NEVER> (choose one of) REFINE_NEVER REFINE_IFNEEDED REFINE_ALWAYS (KSPGMRESSetCGSRefinementType)
  -ksp_gmres_krylov_monitor: <FALSE> Plot the Krylov directions (KSPMonitorSet)
  -ksp_view: View linear solver parameters (KSPView)
Preconditioner (PC) options -------------------------------------------------
  -pc_type <hypre>: Preconditioner (one of) none jacobi pbjacobi bjacobi sor lu shell mg
      eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic hmpi asa cp lsc redistribute svd gamg ml hypre pfmg syspfmg tfs python (PCSetType)
  -pc_use_amat: <FALSE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  HYPRE preconditioner options
  -pc_hypre_type <boomeramg> (choose one of) pilut parasails boomeramg euclid (PCHYPRESetType)
  HYPRE BoomerAMG Options
  -pc_hypre_boomeramg_cycle_type <V> (choose one of) V W (None)
  -pc_hypre_boomeramg_max_levels <25>: Number of levels (of grids) allowed (None)
  -pc_hypre_boomeramg_max_iter <1>: Maximum iterations used PER hypre call (None)
  -pc_hypre_boomeramg_tol <0>: Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations) (None)
  -pc_hypre_boomeramg_truncfactor <0>: Truncation factor for interpolation (0=no truncation) (None)
  -pc_hypre_boomeramg_P_max <0>: Max elements per row for interpolation operator (0=unlimited) (None)
  -pc_hypre_boomeramg_agg_nl <0>: Number of levels of aggressive coarsening (None)
  -pc_hypre_boomeramg_agg_num_paths <1>: Number of paths for aggressive coarsening (None)
  -pc_hypre_boomeramg_strong_threshold <0.25>: Threshold for being strongly connected (None)
  -pc_hypre_boomeramg_max_row_sum <0.9>: Maximum row sum (None)
  -pc_hypre_boomeramg_grid_sweeps_all <1>: Number of sweeps for the up and down grid levels (None)
  -pc_hypre_boomeramg_grid_sweeps_down <1>: Number of sweeps for the down cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_up <1>: Number of sweeps for the up cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_coarse <1>: Number of sweeps for the coarse level (None)
  -pc_hypre_boomeramg_relax_type_all <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_down <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_up <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_coarse <Gaussian-elimination> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_weight_all <1>: Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_relax_weight_level <1>: Set the relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_outer_relax_weight_all <1>: Outer relaxation weight for all levels (-k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_outer_relax_weight_level <1>: Set the outer relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_no_CF: <FALSE> Do not use CF-relaxation (None)
  -pc_hypre_boomeramg_measure_type <local> (choose one of) local global (None)
  -pc_hypre_boomeramg_coarsen_type <Falgout> (choose one of) CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None)
  -pc_hypre_boomeramg_interp_type <classical> (choose one of) classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1 (None)
  -pc_hypre_boomeramg_print_statistics: Print statistics (None)
  -pc_hypre_boomeramg_print_debug: Print debug information (None)
  -pc_hypre_boomeramg_nodal_coarsen: <FALSE> HYPRE_BoomerAMGSetNodal() (None)
  -pc_hypre_boomeramg_nodal_relaxation: <FALSE> Nodal relaxation via Schwarz (None)
Krylov Method (KSP) options -------------------------------------------------
  -ksp_type <cg>: Krylov method (one of) cg groppcg pipecg cgne nash stcg gltr richardson
      chebyshev gmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl cgs tfqmr cr pipecr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd gcr pgmres specest dgmres python (KSPSetType)
  -ksp_max_it <10000>: Maximum number of iterations (KSPSetTolerances)
  -ksp_rtol <1e-05>: Relative decrease in residual norm (KSPSetTolerances)
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
  KSP CG and CGNE options
  -ksp_cg_single_reduction: <FALSE> Merge inner products into single MPI_Allreduce() (KSPCGUseSingleReduction)
  -ksp_view: View linear solver parameters (KSPView)
Preconditioner (PC) options -------------------------------------------------
  -pc_type <hypre>: Preconditioner (one of) none jacobi pbjacobi bjacobi sor lu shell mg
      eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic hmpi asa cp lsc redistribute svd gamg ml hypre pfmg syspfmg tfs python (PCSetType)
  -pc_use_amat: <FALSE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  HYPRE preconditioner options
  -pc_hypre_type <boomeramg> (choose one of) pilut parasails boomeramg euclid (PCHYPRESetType)
  HYPRE BoomerAMG Options
  -pc_hypre_boomeramg_cycle_type <V> (choose one of) V W (None)
  -pc_hypre_boomeramg_max_levels <25>: Number of levels (of grids) allowed (None)
  -pc_hypre_boomeramg_max_iter <1>: Maximum iterations used PER hypre call (None)
  -pc_hypre_boomeramg_tol <0>: Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations) (None)
  -pc_hypre_boomeramg_truncfactor <0>: Truncation factor for interpolation (0=no truncation) (None)
  -pc_hypre_boomeramg_P_max <0>: Max elements per row for interpolation operator (0=unlimited) (None)
  -pc_hypre_boomeramg_agg_nl <0>: Number of levels of aggressive coarsening (None)
  -pc_hypre_boomeramg_agg_num_paths <1>: Number of paths for aggressive coarsening (None)
  -pc_hypre_boomeramg_strong_threshold <0.25>: Threshold for being strongly connected (None)
  -pc_hypre_boomeramg_max_row_sum <0.9>: Maximum row sum (None)
  -pc_hypre_boomeramg_grid_sweeps_all <1>: Number of sweeps for the up and down grid levels (None)
  -pc_hypre_boomeramg_grid_sweeps_down <1>: Number of sweeps for the down cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_up <1>: Number of sweeps for the up cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_coarse <1>: Number of sweeps for the coarse level (None)
  -pc_hypre_boomeramg_relax_type_all <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_down <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_up <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_coarse <Gaussian-elimination> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_weight_all <1>: Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_relax_weight_level <1>: Set the relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_outer_relax_weight_all <1>: Outer relaxation weight for all levels (-k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_outer_relax_weight_level <1>: Set the outer relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_no_CF: <FALSE> Do not use CF-relaxation (None)
  -pc_hypre_boomeramg_measure_type <local> (choose one of) local global (None)
  -pc_hypre_boomeramg_coarsen_type <Falgout> (choose one of) CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None)
  -pc_hypre_boomeramg_interp_type <classical> (choose one of) classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1 (None)
  -pc_hypre_boomeramg_print_statistics: Print statistics (None)
  -pc_hypre_boomeramg_print_debug: Print debug information (None)
  -pc_hypre_boomeramg_nodal_coarsen: <FALSE> HYPRE_BoomerAMGSetNodal() (None)
  -pc_hypre_boomeramg_nodal_relaxation: <FALSE> Nodal relaxation via Schwarz (None)
Krylov Method (KSP) options -------------------------------------------------
  -ksp_type <richardson>: Krylov method (one of) cg groppcg pipecg cgne nash stcg gltr richardson
      chebyshev gmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl cgs tfqmr cr pipecr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd gcr pgmres specest dgmres python (KSPSetType)
  -ksp_max_it <1>: Maximum number of iterations (KSPSetTolerances)
  -ksp_rtol <1e-05>: Relative decrease in residual norm (KSPSetTolerances)
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
  KSP Richardson Options
  -ksp_richardson_scale <1>: damping factor (KSPRichardsonSetScale)
  -ksp_richardson_self_scale: <FALSE> dynamically determine optimal damping factor (KSPRichardsonSetSelfScale)
  -ksp_view: View linear solver parameters (KSPView)
Options for SEQAIJ matrix -------------------------------------------------
  -mat_no_unroll: <FALSE> Do not optimize for inodes (slower) (None)
  -mat_no_inode: <FALSE> Do not optimize for inodes -slower- (None)
  -mat_inode_limit <5>: Do not use inodes larger then this value (None)
Options for SEQAIJ matrix -------------------------------------------------
  -mat_no_unroll: <FALSE> Do not optimize for inodes (slower) (None)
  -mat_no_inode: <FALSE> Do not optimize for inodes -slower- (None)
  -mat_inode_limit <5>: Do not use inodes larger then this value (None)
123
116
148
114
151
112
152
134
151
time to solve:  [ 0.20772195]
error norm = 7.19727e-05
4


V:   [ 2624.] Q:   [ 1089.] W:   [ 3713.] 


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
0.0231890678406
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
Preconditioner (PC) options -------------------------------------------------
  -pc_type <ilu>: Preconditioner (one of) none jacobi pbjacobi bjacobi sor lu shell mg
      eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic hmpi asa cp lsc redistribute svd gamg ml hypre pfmg syspfmg tfs python (PCSetType)
  -pc_use_amat: <FALSE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  ILU Options
  -pc_factor_in_place: <FALSE> Form factored matrix in the same memory as the matrix (PCFactorSetUseInPlace)
  -pc_factor_fill <1>: Expected non-zeros in factored matrix (PCFactorSetFill)
  -pc_factor_shift_type <INBLOCKS> (choose one of) NONE NONZERO POSITIVE_DEFINITE INBLOCKS (PCFactorSetShiftType)
  -pc_factor_shift_amount <2.22045e-14>: Shift added to diagonal (PCFactorSetShiftAmount)
  -pc_factor_zeropivot <2.22045e-14>: Pivot is considered zero if less than (PCFactorSetZeroPivot)
  -pc_factor_column_pivot <-2>: Column pivot tolerance (used only for some factorization) (PCFactorSetColumnPivot)
  -pc_factor_pivot_in_blocks: <TRUE> Pivot inside matrix dense blocks for BAIJ and SBAIJ (PCFactorSetPivotInBlocks)
  -pc_factor_reuse_fill: <FALSE> Use fill from previous factorization (PCFactorSetReuseFill)
  -pc_factor_reuse_ordering: <FALSE> Reuse ordering from previous factorization (PCFactorSetReuseOrdering)
  -pc_factor_mat_ordering_type <natural>: Reordering to reduce nonzeros in factored matrix (one of) natural nd 1wd rcm qmd rowlength amd (PCFactorSetMatOrderingType)
  -pc_factor_mat_solver_package <petsc>: Specific direct solver to use (MatGetFactor)
  -pc_factor_levels <0>: levels of fill (PCFactorSetLevels)
  -pc_factor_diagonal_fill: <FALSE> Allow fill into empty diagonal entry (PCFactorSetAllowDiagonalFill)
  -pc_factor_nonzeros_along_diagonal: Reorder to remove zeros from diagonal (PCFactorReorderForNonzeroDiagonal)
Krylov Method (KSP) options -------------------------------------------------
  -ksp_type <gmres>: Krylov method (one of) cg groppcg pipecg cgne nash stcg gltr richardson
      chebyshev gmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl cgs tfqmr cr pipecr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd gcr pgmres specest dgmres python (KSPSetType)
  -ksp_max_it <10000>: Maximum number of iterations (KSPSetTolerances)
  -ksp_rtol <1e-05>: Relative decrease in residual norm (KSPSetTolerances)
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
  KSP GMRES Options
  -ksp_gmres_restart <30>: Number of Krylov search directions (KSPGMRESSetRestart)
  -ksp_gmres_haptol <1e-30>: Tolerance for exact convergence (happy ending) (KSPGMRESSetHapTol)
  -ksp_gmres_preallocate: <FALSE> Preallocate Krylov vectors (KSPGMRESSetPreAllocateVectors)
  Pick at most one of -------------
    -ksp_gmres_classicalgramschmidt: Classical (unmodified) Gram-Schmidt (fast) (KSPGMRESSetOrthogonalization)
    -ksp_gmres_modifiedgramschmidt: Modified Gram-Schmidt (slow,more stable) (KSPGMRESSetOrthogonalization)
  -ksp_gmres_cgs_refinement_type <REFINE_NEVER> (choose one of) REFINE_NEVER REFINE_IFNEEDED REFINE_ALWAYS (KSPGMRESSetCGSRefinementType)
  -ksp_gmres_krylov_monitor: <FALSE> Plot the Krylov directions (KSPMonitorSet)
  -ksp_view: View linear solver parameters (KSPView)
Preconditioner (PC) options -------------------------------------------------
  -pc_type <hypre>: Preconditioner (one of) none jacobi pbjacobi bjacobi sor lu shell mg
      eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic hmpi asa cp lsc redistribute svd gamg ml hypre pfmg syspfmg tfs python (PCSetType)
  -pc_use_amat: <FALSE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  HYPRE preconditioner options
  -pc_hypre_type <boomeramg> (choose one of) pilut parasails boomeramg euclid (PCHYPRESetType)
  HYPRE BoomerAMG Options
  -pc_hypre_boomeramg_cycle_type <V> (choose one of) V W (None)
  -pc_hypre_boomeramg_max_levels <25>: Number of levels (of grids) allowed (None)
  -pc_hypre_boomeramg_max_iter <1>: Maximum iterations used PER hypre call (None)
  -pc_hypre_boomeramg_tol <0>: Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations) (None)
  -pc_hypre_boomeramg_truncfactor <0>: Truncation factor for interpolation (0=no truncation) (None)
  -pc_hypre_boomeramg_P_max <0>: Max elements per row for interpolation operator (0=unlimited) (None)
  -pc_hypre_boomeramg_agg_nl <0>: Number of levels of aggressive coarsening (None)
  -pc_hypre_boomeramg_agg_num_paths <1>: Number of paths for aggressive coarsening (None)
  -pc_hypre_boomeramg_strong_threshold <0.25>: Threshold for being strongly connected (None)
  -pc_hypre_boomeramg_max_row_sum <0.9>: Maximum row sum (None)
  -pc_hypre_boomeramg_grid_sweeps_all <1>: Number of sweeps for the up and down grid levels (None)
  -pc_hypre_boomeramg_grid_sweeps_down <1>: Number of sweeps for the down cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_up <1>: Number of sweeps for the up cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_coarse <1>: Number of sweeps for the coarse level (None)
  -pc_hypre_boomeramg_relax_type_all <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_down <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_up <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_coarse <Gaussian-elimination> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_weight_all <1>: Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_relax_weight_level <1>: Set the relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_outer_relax_weight_all <1>: Outer relaxation weight for all levels (-k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_outer_relax_weight_level <1>: Set the outer relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_no_CF: <FALSE> Do not use CF-relaxation (None)
  -pc_hypre_boomeramg_measure_type <local> (choose one of) local global (None)
  -pc_hypre_boomeramg_coarsen_type <Falgout> (choose one of) CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None)
  -pc_hypre_boomeramg_interp_type <classical> (choose one of) classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1 (None)
  -pc_hypre_boomeramg_print_statistics: Print statistics (None)
  -pc_hypre_boomeramg_print_debug: Print debug information (None)
  -pc_hypre_boomeramg_nodal_coarsen: <FALSE> HYPRE_BoomerAMGSetNodal() (None)
  -pc_hypre_boomeramg_nodal_relaxation: <FALSE> Nodal relaxation via Schwarz (None)
Krylov Method (KSP) options -------------------------------------------------
  -ksp_type <cg>: Krylov method (one of) cg groppcg pipecg cgne nash stcg gltr richardson
      chebyshev gmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl cgs tfqmr cr pipecr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd gcr pgmres specest dgmres python (KSPSetType)
  -ksp_max_it <10000>: Maximum number of iterations (KSPSetTolerances)
  -ksp_rtol <1e-05>: Relative decrease in residual norm (KSPSetTolerances)
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
  KSP CG and CGNE options
  -ksp_cg_single_reduction: <FALSE> Merge inner products into single MPI_Allreduce() (KSPCGUseSingleReduction)
  -ksp_view: View linear solver parameters (KSPView)
Preconditioner (PC) options -------------------------------------------------
  -pc_type <hypre>: Preconditioner (one of) none jacobi pbjacobi bjacobi sor lu shell mg
      eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic hmpi asa cp lsc redistribute svd gamg ml hypre pfmg syspfmg tfs python (PCSetType)
  -pc_use_amat: <FALSE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  HYPRE preconditioner options
  -pc_hypre_type <boomeramg> (choose one of) pilut parasails boomeramg euclid (PCHYPRESetType)
  HYPRE BoomerAMG Options
  -pc_hypre_boomeramg_cycle_type <V> (choose one of) V W (None)
  -pc_hypre_boomeramg_max_levels <25>: Number of levels (of grids) allowed (None)
  -pc_hypre_boomeramg_max_iter <1>: Maximum iterations used PER hypre call (None)
  -pc_hypre_boomeramg_tol <0>: Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations) (None)
  -pc_hypre_boomeramg_truncfactor <0>: Truncation factor for interpolation (0=no truncation) (None)
  -pc_hypre_boomeramg_P_max <0>: Max elements per row for interpolation operator (0=unlimited) (None)
  -pc_hypre_boomeramg_agg_nl <0>: Number of levels of aggressive coarsening (None)
  -pc_hypre_boomeramg_agg_num_paths <1>: Number of paths for aggressive coarsening (None)
  -pc_hypre_boomeramg_strong_threshold <0.25>: Threshold for being strongly connected (None)
  -pc_hypre_boomeramg_max_row_sum <0.9>: Maximum row sum (None)
  -pc_hypre_boomeramg_grid_sweeps_all <1>: Number of sweeps for the up and down grid levels (None)
  -pc_hypre_boomeramg_grid_sweeps_down <1>: Number of sweeps for the down cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_up <1>: Number of sweeps for the up cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_coarse <1>: Number of sweeps for the coarse level (None)
  -pc_hypre_boomeramg_relax_type_all <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_down <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_up <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_coarse <Gaussian-elimination> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_weight_all <1>: Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_relax_weight_level <1>: Set the relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_outer_relax_weight_all <1>: Outer relaxation weight for all levels (-k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_outer_relax_weight_level <1>: Set the outer relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_no_CF: <FALSE> Do not use CF-relaxation (None)
  -pc_hypre_boomeramg_measure_type <local> (choose one of) local global (None)
  -pc_hypre_boomeramg_coarsen_type <Falgout> (choose one of) CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None)
  -pc_hypre_boomeramg_interp_type <classical> (choose one of) classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1 (None)
  -pc_hypre_boomeramg_print_statistics: Print statistics (None)
  -pc_hypre_boomeramg_print_debug: Print debug information (None)
  -pc_hypre_boomeramg_nodal_coarsen: <FALSE> HYPRE_BoomerAMGSetNodal() (None)
  -pc_hypre_boomeramg_nodal_relaxation: <FALSE> Nodal relaxation via Schwarz (None)
Krylov Method (KSP) options -------------------------------------------------
  -ksp_type <richardson>: Krylov method (one of) cg groppcg pipecg cgne nash stcg gltr richardson
      chebyshev gmres tcqmr bcgs ibcgs fbcgs fbcgsr bcgsl cgs tfqmr cr pipecr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd gcr pgmres specest dgmres python (KSPSetType)
  -ksp_max_it <1>: Maximum number of iterations (KSPSetTolerances)
  -ksp_rtol <1e-05>: Relative decrease in residual norm (KSPSetTolerances)
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
  KSP Richardson Options
  -ksp_richardson_scale <1>: damping factor (KSPRichardsonSetScale)
  -ksp_richardson_self_scale: <FALSE> dynamically determine optimal damping factor (KSPRichardsonSetSelfScale)
  -ksp_view: View linear solver parameters (KSPView)
Options for SEQAIJ matrix -------------------------------------------------
  -mat_no_unroll: <FALSE> Do not optimize for inodes (slower) (None)
  -mat_no_inode: <FALSE> Do not optimize for inodes -slower- (None)
  -mat_inode_limit <5>: Do not use inodes larger then this value (None)
Options for SEQAIJ matrix -------------------------------------------------
  -mat_no_unroll: <FALSE> Do not optimize for inodes (slower) (None)
  -mat_no_inode: <FALSE> Do not optimize for inodes -slower- (None)
  -mat_inode_limit <5>: Do not use inodes larger then this value (None)
252
231
292
229
293
210
300
249
284
[1;31m---------------------------------------------------------------------------[0m
[1;31mKeyboardInterrupt[0m                         Traceback (most recent call last)
[1;32m/home/mwathen/Dropbox/MastersResearch/MHD/FEniCS/FieldSplit/Maxwell/LargeScale/Maxwell.py[0m in [0;36m<module>[1;34m()[0m
[0;32m    170[0m         [1;31m# Solve![0m[1;33m[0m[1;33m[0m[0m
[0;32m    171[0m         [0mtic[0m[1;33m([0m[1;33m)[0m[1;33m[0m[0m
[1;32m--> 172[1;33m         [0mksp[0m[1;33m.[0m[0msolve[0m[1;33m([0m[0mbb[0m[1;33m,[0m [0mx[0m[1;33m)[0m[1;33m[0m[0m
[0m[0;32m    173[0m         [0mSolTime[0m[1;33m[[0m[0mxx[0m[1;33m-[0m[1;36m1[0m[1;33m][0m [1;33m=[0m [0mtoc[0m[1;33m([0m[1;33m)[0m[1;33m[0m[0m
[0;32m    174[0m         [1;32mprint[0m [1;34m"time to solve: "[0m[1;33m,[0m[0mSolTime[0m[1;33m[[0m[0mxx[0m[1;33m-[0m[1;36m1[0m[1;33m][0m[1;33m[0m[0m

[1;32m/home/mwathen/programs4/petsc/petsc4py/lib/python2.7/site-packages/petsc4py/lib/arch-linux2-c-opt/PETSc.so[0m in [0;36mpetsc4py.PETSc.KSP.solve (src/petsc4py.PETSc.c:115228)[1;34m()[0m

[1;32m/home/mwathen/Dropbox/MastersResearch/MHD/FEniCS/FieldSplit/Maxwell/LargeScale/libpetsc4py.pyx[0m in [0;36mlibpetsc4py.PCApply_Python (src/libpetsc4py/libpetsc4py.c:12889)[1;34m()[0m

[1;32m/home/mwathen/Dropbox/MastersResearch/MHD/FEniCS/FieldSplit/Maxwell/MaxwellPrecond.py[0m in [0;36mapply[1;34m(self, pc, x, y)[0m
[0;32m    131[0m [1;33m[0m[0m
[0;32m    132[0m [1;33m[0m[0m
[1;32m--> 133[1;33m         [0mself[0m[1;33m.[0m[0mkspCurlCurl[0m[1;33m.[0m[0msolve[0m[1;33m([0m[0mx1[0m[1;33m,[0m [0my1[0m[1;33m)[0m[1;33m[0m[0m
[0m[0;32m    134[0m         [1;32mprint[0m [0mself[0m[1;33m.[0m[0mkspCurlCurl[0m[1;33m.[0m[0mits[0m[1;33m[0m[0m
[0;32m    135[0m         [0mself[0m[1;33m.[0m[0mkspLapl[0m[1;33m.[0m[0msolve[0m[1;33m([0m[0mx2[0m[1;33m,[0m [0my2[0m[1;33m)[0m[1;33m[0m[0m

[1;31mKeyboardInterrupt[0m: 
