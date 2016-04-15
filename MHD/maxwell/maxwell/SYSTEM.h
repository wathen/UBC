#pragma once

#include <stdio.h>

#include "petscksp.h"
#include "MESH.h"
//#include "stencil.h"
#include "PetscCalls.h"

class AIJ;

class SYSTEM
{
public:
	//Variables
	PetscScalar paramk;
	Mat	Stiffmat;	//linear system matrix
	Vec	rhs;	//rhs vector
	Vec sol;
	Vec exactsol;

	Vec ue, pe;
	Vec u, p;

	Mat	Amat;
	Mat	Bmat;
	Mat	Mmat;
	Mat	Lmat;
	Vec	f;
	Vec g;

	Mat Qmat;
	Mat Cmat;
	Mat Rx, Ry, Rz;

	MESH mesh;
	//STENCIL stencil;

	FILE *stream;

	int variable; //variable coefficient or not

	//Functions
	SYSTEM(void);
	~SYSTEM(void);
	void Get_Exact_Sol(void);
	//void GetSparsity(void);
	void Assemble(void); //must call Get_Exact_Sol beofore calling Assemble
	void Apply_BC(void); //modify matrices to include BC, i.e. shrink size
	void Form_System(void); //put matrices together to form the linear system
	void Solve(void);
	void View(void);
	void Destroy(void);
	PetscScalar tancomp(PetscScalar *spt, PetscScalar *ept);
	void func_f(PetscScalar *coord, PetscScalar *f);
	void func_u(PetscScalar *coord, PetscScalar *u) ;
	void func_p(PetscScalar *coord, PetscScalar *p);

	//for variable coefficient
	double func_mu(double *coord);
	double func_eps(double *coord);


private:
	void MyshellSetup(Hiptmair *shell, Mat& LQ);

	// array2CSR is not used in the code, might be helpful for later....
	void array2CSR(AIJ* a, int size, int nrows, int* is, int* js, double* vs);	//size: # of elements in array a, nrows: # of local rows, CSR(is, js, vs), need to preallocate space for (is, js, vs)
};

// class AIJ is not used in the code, might be helpful for later
class AIJ{
public:
	AIJ() {i=-1; j=-1; value=0.0;};
	int i, j;
	double value;
	void set(int _i, int _j, double _v) {i=_i; j=_j; value=_v;}
};

//not used
class AIJCompare {
public:
	bool operator() (AIJ a, AIJ b) { return (a.i<b.i);}
};
