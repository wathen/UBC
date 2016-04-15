#include "PetscCalls.h"

extern PetscInt SIZE, RANK;

void Petsc_Init(PetscInt argc, char **args, char *help)
{
	PetscErrorCode ierr;
	//PetscInt n;
	//PetscMPIINT size;
	ierr = PetscInitialize(&argc, &args, (char*)0, help);//CHKERRQ(ierr);
	ierr = MPI_Comm_size(PETSC_COMM_WORLD,&SIZE);//CHKERRQ(ierr);
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&RANK);//CHKERRQ(ierr);

}

void Petsc_End()
{
	PetscErrorCode ierr;

	ierr = PetscFinalize();

}

void Mat_Create(Mat& mat, PetscInt m, PetscInt n, PetscInt M, PetscInt N)
{
	PetscErrorCode ierr;


	ierr = MatCreate(PETSC_COMM_WORLD, &mat);
	ierr = MatSetSizes(mat, m, n, M,N);
	ierr = MatSetFromOptions(mat);	//Options Database Keys: -mat_type

}

void Mat_Create(PetscInt size, PetscInt rank, Mat& mat, PetscInt M, PetscInt N)
{
	PetscErrorCode ierr;

	ierr = MatCreate(PETSC_COMM_WORLD, &mat);
	ierr = MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, M,N);
	ierr = MatSetFromOptions(mat);	//Options Database Keys: -mat_type

}


void Mat_Assemble(Mat& mat)
{
	PetscErrorCode ierr;

	ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
	ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);

}


void Mat_View(Mat& mat, char *obj, PetscInt N, PetscViewerFormat format)
{
	PetscViewer viewer;
	PetscErrorCode ierr;
	char fn[100];

	sprintf(fn, "out_%d_%s.txt", N, obj);

	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, fn, &viewer);
	ierr = PetscViewerPushFormat(viewer, format);
	ierr = PetscObjectSetName((PetscObject)mat, obj);
	ierr = MatView(mat, viewer);
	PetscViewerDestroy(viewer);
}


void Vec_Create(PetscInt size, PetscInt rank, Vec& vec, PetscInt M)
{
	PetscErrorCode ierr;

	ierr = VecCreate(PETSC_COMM_WORLD, &vec);
	ierr = VecSetSizes(vec, PETSC_DECIDE, M);
	ierr = VecSetFromOptions(vec);	//Options Database Keys????
}

void Vec_Create(Vec& vec, PetscInt m, PetscInt M)
{
	PetscErrorCode ierr;

	ierr = VecCreate(PETSC_COMM_WORLD, &vec);
	ierr = VecSetSizes(vec, m, M);
	ierr = VecSetFromOptions(vec);	//Options Database Keys: ??
}

void Vec_Assemble(Vec& vec)
{
	PetscErrorCode ierr;

	ierr = VecAssemblyBegin(vec);
	ierr = VecAssemblyEnd(vec);

}

void Vec_View(Vec& vec, char *obj, PetscInt N, PetscViewerFormat format)
{
	PetscViewer viewer;
	PetscErrorCode ierr;
	char fn[100];

	sprintf(fn, "out_%d_%s.txt", N, obj);

	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, fn, &viewer);
	ierr = PetscViewerPushFormat(viewer, format);
	ierr = PetscObjectSetName((PetscObject)vec, obj);
	ierr = VecView(vec, viewer);
	PetscViewerDestroy(viewer);
}

void MPI_GetCommPartition(PetscInt size, PetscInt M, PetscInt *partition)
{

	for (int i=0; i<size; i++)
		partition[i] = (PetscInt)(M/size) + ((M%size)>i);
}

PetscInt MPI_GetCommPartition(PetscInt size, PetscInt i, PetscInt M)
{
	//return the local size on proc rank i
	return ((PetscInt)(M/size) + ((M%size)>i));

}

void MPI_MatGetValue(PetscInt size, PetscInt rank, Mat& mat,  PetscInt proc, PetscInt indxm, PetscInt indxn, PetscScalar *val)
{
	PetscErrorCode ierr;
	PetscInt i, j, tmpi, tmpj, indx[2], tmpindx[2];
	PetscInt *nreq;
	PetscScalar tmp;
	//PetscInt proc;

	MPI_Status status;
	const PetscInt tag_req=1, tag_indx=2, tag_val=3;

	//if require local entry
	if (proc==rank) {
		ierr = MatGetValues(mat, 1, &indxm, 1, &indxn, val);
		return;
	}

	//if require remote entry
	PetscMalloc((size)*sizeof(PetscInt),&nreq);

	for (i=0; i<size; i++){
		nreq[i] = 0;
	}


	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] *** want mat[%d, %d] on process%d\n",rank,indxm,indxn, proc);
	for (i=0; i<size; i++)
		PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] *** nreq[%d]=%d\n",rank,i,nreq[i]);



	indx[0]=indxm;
	indx[1]=indxn;

	//send request to other (size-1) processs
	for (i=0; i<size; i++)
		if (i!=rank){
			if(i!=proc) {
				tmpi = 0;
                MPI_Send(&tmpi, 1, MPI_INT, i, tag_req, MPI_COMM_WORLD);
			}
			else{
				tmpi = 1;
                MPI_Send(&tmpi, 1, MPI_INT, i, tag_req, MPI_COMM_WORLD);
				MPI_Send(&indx, 2, MPI_INT, i, tag_indx, MPI_COMM_WORLD);
			}
		}


	//collect # of requests from other (size-1) process
	for (i=0; i<size; i++) {
		if (i!=rank) {
			MPI_Recv(&tmpi, 1, MPI_INT, i, tag_req, MPI_COMM_WORLD, &status);
			if (tmpi !=0 )	nreq[i] += tmpi;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]  exit 2st barrier\n",rank);
	for (i=0; i<size; i++)
		PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] *** nreq[%d]=%d\n",rank,i,nreq[i]);



	for (i=0; i<size; i++)
		if (nreq[i]!=0)
			for (j=0; j<nreq[i]; j++) {
				MPI_Recv(tmpindx, 2, MPI_INT, i, tag_indx, MPI_COMM_WORLD, &status);
				ierr = MatGetValues(mat,1,&tmpindx[0],1, &tmpindx[1] ,val);
				MPI_Send(val, 1,MPI_DOUBLE, i, tag_val, MPI_COMM_WORLD);
			}


	MPI_Recv(val, 1, MPI_DOUBLE, proc, tag_val, MPI_COMM_WORLD, &status);

    //ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] mat[%d, %d] = %g\n",rank,indxm,indxn, *val);

	PetscFree(nreq);

	PetscSynchronizedFlush(PETSC_COMM_WORLD);

}



void MPI_MatGetValues(PetscInt size, PetscInt rank, Mat& mat, PetscInt proc, PetscInt m, PetscInt *indxm, PetscInt n, PetscInt *indxn, PetscScalar *val)
{
	PetscErrorCode ierr;
	PetscInt i,j;

	for (i=0; i<m; i++)
		for (j=0; j<n; j++) {
            MPI_MatGetValue(size, rank, mat, proc, indxm[i], indxn[j], &val[i*n+j]);
	}
}


/*
#undef __FUNCT__
#define __FUNCT__ "MyShellPCApply"
PetscErrorCode MyShellPCApply(void *ctx,Vec x,Vec y)
{

	PetscErrorCode  ierr;

	//solve for y, given x
	Hiptmair *shell = (Hiptmair*)ctx;

	//----------get contribution from diagAM
	VecPointwiseMult(y,x,shell->diagAM);

	//allocate space for y1, yhat
	Vec y1, yhat;
	VecDuplicate(shell->diagAM,&y1);
	VecDuplicate(shell->diagL,&yhat);

	//----------get contribution from diagL
	// yhat = C'*x
	MatMultTranspose(shell->C,x, yhat);
	// scale by DiagL
	VecPointwiseMult(yhat,yhat,shell->diagL);
	// y1 = C*yhat
	MatMult(shell->C,yhat, y1);

	//y = y+1/(1-k^2)*y1
	VecAXPY(y, 1.0/(1-(shell->k)*(shell->k)), y1);


	//----------get contribution from diagLQ
	//Rx
	// yhat = Rx'*x
	MatMultTranspose(shell->Rx,x, yhat);
	// scale by DiagLQ
	VecPointwiseMult(yhat,yhat,shell->diagLQ);
	// y1 = Rx*yhat
	MatMult(shell->Rx,yhat, y1);
	//y = y+y1
	VecAXPY(y, 1.0, y1);

	//Ry
	// yhat = Ry'*x
	MatMultTranspose(shell->Ry,x, yhat);
	// scale by DiagLQ
	VecPointwiseMult(yhat,yhat,shell->diagLQ);
	// y1 = Ry*yhat
	MatMult(shell->Ry,yhat, y1);
	//y = y+y1
	VecAXPY(y, 1.0, y1);

	//Rz
	// yhat = Rz'*x
	MatMultTranspose(shell->Rz,x, yhat);
	// scale by DiagLQ
	VecPointwiseMult(yhat,yhat,shell->diagLQ);
	// y1 = Rz*yhat
	MatMult(shell->Rz,yhat, y1);
	//y = y+y1
	VecAXPY(y, 1.0, y1);

	// release memory
	VecDestroy(y1);
	VecDestroy(yhat);

	return 0;
}

*/


#undef __FUNCT__
#define __FUNCT__ "MyShellPCApply"
PetscErrorCode MyShellPCApply(void *ctx,Vec x,Vec y)
{

	PetscErrorCode  ierr;

	//solve for y, given x
	Hiptmair *shell = (Hiptmair*)ctx;

	//----------get contribution from diagAM
	VecPointwiseMult(y,x,shell->diagAM);

	//allocate space for y1, yhat
	Vec y1, yhat, xhat;
	y1=shell->y1;
	yhat=shell->yhat;
	xhat=shell->xhat;

	//----------get contribution from diagL
	//xhat = C'*x
	MatMultTranspose(sh
		ell->C,x, xhat);
	//solve L*yhat = xhat
	KSPSolve(shell->innerksp1,xhat,yhat);
	// y1 = C*yhat
	MatMult(shell->C,yhat, y1);

	//y = y+1/(1-k^2)*y1
	VecAXPY(y, 1.0/(1-(shell->k)*(shell->k)), y1);

	//int its;
	//KSPGetIterationNumber(shell->innerksp1,&its);
	//PetscPrintf(PETSC_COMM_WORLD,"	++++ inner KSP1: Inner Iterations %d\n", its);


	//----------get contribution from diagLQ
	//Rx
	// yhat = Rx'*x
	MatMultTranspose(shell->Rx,x, xhat);
	//solve LQ*yhat = xhat
	KSPSolve(shell->innerksp2,xhat,yhat);
	// y1 = Rx*yhat
	MatMult(shell->Rx,yhat, y1);
	//y = y+y1
	VecAXPY(y, 1.0, y1);

	//Ry
	// yhat = Ry'*x
	MatMultTranspose(shell->Ry,x, xhat);
	//solve LQ*yhat = xhat
	KSPSolve(shell->innerksp2,xhat,yhat);
	// y1 = Ry*yhat
	MatMult(shell->Ry,yhat, y1);
	//y = y+y1
	VecAXPY(y, 1.0, y1);

	//Rz
	// yhat = Rz'*x
	MatMultTranspose(shell->Rz,x, xhat);
	//solve LQ*yhat = xhat
	KSPSolve(shell->innerksp2,xhat,yhat);
	// y1 = Rz*yhat
	MatMult(shell->Rz,yhat, y1);
	//y = y+y1
	VecAXPY(y, 1.0, y1);

	return 0;
}
