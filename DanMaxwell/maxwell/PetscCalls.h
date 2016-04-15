#ifndef _PETSCCALLSH

#define _PETSCCALLSH	1

#include "petscksp.h"

#include "petscext.h"
#include "petscext_vec.h"
#include "petscext_mat.h"
#include "petscext_ksp.h"
#include "petscext_pc.h"


typedef PetscScalar MAT33[3][3];
typedef PetscScalar MAT34[3][4];
typedef PetscScalar MAT66[6][6];
typedef PetscScalar MAT46[4][6];
typedef PetscScalar MAT44[4][4];
typedef PetscScalar VEC3[3];
typedef PetscScalar VEC6[6];


void Petsc_Init(PetscInt argc, char **args, char *help);
void Petsc_End();
void Mat_Create(PetscInt size, PetscInt rank, Mat& mat, PetscInt M, PetscInt N);
void Mat_Create(Mat& mat, PetscInt m, PetscInt n, PetscInt M, PetscInt N);
void Mat_Assemble(Mat& mat);
void Mat_View(Mat& mat, char *obj, PetscInt N, PetscViewerFormat format);

void Vec_Create(PetscInt size, PetscInt rank, Vec& vec, PetscInt M);
void Vec_Create(Vec& vec, PetscInt m, PetscInt M);
void Vec_Assemble(Vec& vec);
void Vec_View(Vec& vec, char *obj, PetscInt N, PetscViewerFormat format);

void MPI_GetCommPartition(PetscInt size, PetscInt M, PetscInt *partition);
PetscInt MPI_GetCommPartition(PetscInt size, PetscInt i, PetscInt M);
void MPI_MatGetValue(PetscInt size, PetscInt rank, Mat& mat,  PetscInt proc, PetscInt indxm, PetscInt indxn, PetscScalar *val);
void MPI_MatGetValues(PetscInt size, PetscInt rank, Mat& mat, PetscInt proc, PetscInt m, PetscInt *indxm, PetscInt n, PetscInt *indxn, PetscScalar *val);

//PetscScalar det33(MAT33 mat);
//void matinv33(MAT33 m, MAT33 minv); 
//void matmult334(MAT33 a, MAT34 b, MAT34 c);
//PetscScalar dot3(VEC3 a, VEC3 b);
//void cross3(VEC3 a, VEC3 b, VEC3 c);
//void matmult331(MAT33 a, VEC3 b, VEC3 c);
//void vecadd3(VEC3 a, VEC3 b, VEC3 c);	
//void vecscale3(VEC3 a, PetscScalar sf, VEC3 b);
//void vecsub3(VEC3 a, VEC3 b, VEC3 c);
//void matscale66(MAT66 a, PetscScalar sf, MAT66 b);
//void matscale46(MAT46 a, PetscScalar sf, MAT46 b);
//void matscale44(MAT44 a, PetscScalar sf, MAT44 b);
//void vecscale6(VEC6 a, PetscScalar sf, VEC6 b);
//
//void func_f(VEC3 a, PetscScalar k,VEC3 b);

class Hiptmair {
public:
	Mat L, Q, C, Rx, Ry, Rz;
	Vec diagAM; //diag(A+(1-k)^2 M)^-1
	Vec diagLQ; //diag(L+(1-k)^2 Q)^-1
	Vec diagL; //diag(L)^-1
	double k;
	KSP innerksp1, innerksp2;
	Vec xhat, yhat, y1;
};

PetscErrorCode MyShellPCApply(void *ctx,Vec x,Vec y);
#endif
