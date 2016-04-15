#include <time.h>
#include <algorithm>
//#include <iostream>
//using std::cout;

#ifndef _WIN32
#include <unistd.h>
#include <fstream>
#endif

#include "SYSTEM.h"
#include "PetscCalls.h"
#include "math_lib.h"

#include "petscext.h"
#include "petscext_vec.h"
#include "petscext_mat.h"
#include "petscext_ksp.h"
#include "petscext_pc.h"
#include "petscext_utils.h"

extern PetscInt SIZE, RANK;

SYSTEM::SYSTEM(void)
{
	paramk = 0.0;
}

SYSTEM::~SYSTEM(void)
{
}

void SYSTEM::Assemble()
{
	//print memory usage
	#ifndef _WIN32
	char buf[30];
    sprintf(buf, "/proc/%u/statm", (unsigned)getpid());
	fstream sysfile;
	sysfile.open (buf, fstream::in);
	int programSize, memorySize;
	sysfile>>programSize>>memorySize;
	sysfile.close();
	//PetscPrintf(PETSC_COMM_SELF, "[proc %d] %s\n", RANK, buf);
	PetscPrintf(PETSC_COMM_WORLD, "	--- Memory usage when Assemble begins\n [proc %d] Program size %.4f Mb, memory size %.4f Mb\n\n", RANK, programSize/1024.0, memorySize/1024.0);
	#endif

	//print mesh info to file
	PetscPrintf(PETSC_COMM_WORLD, "	Nnod=%d, Nnod_i=%d, Nel=%d, Nedge=%d, Nedge_i=%d, DOFs =%d\n", mesh.Nnod, mesh.Nnod_i, mesh.Nel, mesh.Nedge, mesh.Nedge_i, mesh.Nnod_i+mesh.Nedge_i);
	PetscFPrintf(PETSC_COMM_WORLD, stream, "	Nnod=%d, Nnod_i=%d, Nel=%d, Nedge=%d, Nedge_i=%d, DOFs =%d\n", mesh.Nnod, mesh.Nnod_i, mesh.Nel, mesh.Nedge, mesh.Nedge_i, mesh.Nnod_i+mesh.Nedge_i);
	for (int i=0; i<SIZE; i++) {
		PetscPrintf(PETSC_COMM_WORLD, "	[proc %d] Nel=%d, Nnod_i=%d, Nnod_b=%d, Nedge_i=%d, Nedge_b=%d\n", i, mesh.elem_count[i], mesh.nod_count_i[i], mesh.nod_count_b[i], mesh.edge_count_i[i], mesh.edge_count_b[i]);
		PetscFPrintf(PETSC_COMM_WORLD, stream, "	[proc %d] Nel=%d, Nnod_i=%d, Nnod_b=%d, Nedge_i=%d, Nedge_b=%d\n", i, mesh.elem_count[i], mesh.nod_count_i[i], mesh.nod_count_b[i], mesh.edge_count_i[i], mesh.edge_count_b[i]);
	}

	PetscScalar L[4], gradL[4][3];
	PetscScalar gradLhat[4][3] = {{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};

	//quadrature pts
	PetscScalar xnod[] = {0.25};
	PetscScalar ynod[] = {0.25};
	PetscScalar znod[] = {0.25};
	PetscScalar w[] = {1.0/6.0};
	PetscInt lw = 1;
	//PetscScalar xnod[] = {1.0/4, 1.0/2, 1.0/6, 1.0/6, 1.0/6};
	//PetscScalar ynod[] = {1.0/4, 1.0/6, 1.0/6, 1.0/6, 1.0/2};
	//PetscScalar znod[] = {1.0/4, 1.0/6, 1.0/6, 1.0/2, 1.0/6};
	//PetscScalar w[] = {-4.0/30, 9.0/120, 9.0/120, 9.0/120, 9.0/120};
	//PetscInt lw = 5;


	PetscInt edgeK[2][6] = {{0, 0, 0, 1, 1, 2},		//{1,1,1,2,2,3}
					   {1, 2, 3, 2, 3, 3}};		//{2,3,4,3,4,4}

	int Nnod_local, Nedge_local;
	Nnod_local = mesh.nod_count_i[RANK] + mesh.nod_count_b[RANK];
	Nedge_local = mesh.edge_count_i[RANK] + mesh.edge_count_b[RANK];

	//------allocate exact memory-------
	MatCreateMPIAIJ(PETSC_COMM_WORLD, Nedge_local, Nedge_local, mesh.Nedge, mesh.Nedge, 0, mesh.d_nnz_edge, 0, mesh.o_nnz_edge, &Amat);
	MatCreateMPIAIJ(PETSC_COMM_WORLD, Nedge_local, Nedge_local, mesh.Nedge, mesh.Nedge, 0, mesh.d_nnz_edge, 0, mesh.o_nnz_edge, &Mmat);
	MatCreateMPIAIJ(PETSC_COMM_WORLD, Nnod_local, Nnod_local, mesh.Nnod, mesh.Nnod, 0, mesh.d_nnz_nod, 0, mesh.o_nnz_nod, &Lmat);
	MatCreateMPIAIJ(PETSC_COMM_WORLD, Nnod_local, Nedge_local, mesh.Nnod, mesh.Nedge, 0, mesh.d_nnz_nod_edge, 0, mesh.o_nnz_nod_edge, &Bmat);

	VecCreateMPI(PETSC_COMM_WORLD, Nedge_local, mesh.Nedge, &f);
	VecCreateMPI(PETSC_COMM_WORLD, Nnod_local, mesh.Nnod, &g);

	MatCreateMPIAIJ(PETSC_COMM_WORLD, Nnod_local, Nnod_local, mesh.Nnod, mesh.Nnod, 0, mesh.d_nnz_nod, 0, mesh.o_nnz_nod, &Qmat);
	MatCreateMPIAIJ(PETSC_COMM_WORLD, Nedge_local, Nnod_local, mesh.Nedge, mesh.Nnod, 0, mesh.d_nnz_edge_nod, 0, mesh.o_nnz_edge_nod, &Cmat);
	MatCreateMPIAIJ(PETSC_COMM_WORLD, Nedge_local, Nnod_local, mesh.Nedge, mesh.Nnod, 0, mesh.d_nnz_edge_nod, 0, mesh.o_nnz_edge_nod, &Rx);
	MatCreateMPIAIJ(PETSC_COMM_WORLD, Nedge_local, Nnod_local, mesh.Nedge, mesh.Nnod, 0, mesh.d_nnz_edge_nod, 0, mesh.o_nnz_edge_nod, &Ry);
	MatCreateMPIAIJ(PETSC_COMM_WORLD, Nedge_local, Nnod_local, mesh.Nedge, mesh.Nnod, 0, mesh.d_nnz_edge_nod, 0, mesh.o_nnz_edge_nod, &Rz);


	//------------allocate no memory-----------
	//MatCreateMPIAIJ(PETSC_COMM_WORLD, Nedge_local, Nedge_local, mesh.Nedge, mesh.Nedge, 0, 0, 0, 0, &Amat);
	//MatCreateMPIAIJ(PETSC_COMM_WORLD, Nedge_local, Nedge_local, mesh.Nedge, mesh.Nedge, 0, 0, 0, 0, &Mmat);
	//MatCreateMPIAIJ(PETSC_COMM_WORLD, Nnod_local, Nnod_local, mesh.Nnod, mesh.Nnod, 0, 0, 0, 0, &Lmat);
	//MatCreateMPIAIJ(PETSC_COMM_WORLD, Nnod_local, Nedge_local, mesh.Nnod, mesh.Nedge, 0, 0, 0, 0, &Bmat);

	//VecCreateMPI(PETSC_COMM_WORLD, Nedge_local, mesh.Nedge, &f);
	//VecCreateMPI(PETSC_COMM_WORLD, Nnod_local, mesh.Nnod, &g);

	//MatCreateMPIAIJ(PETSC_COMM_WORLD, Nnod_local, Nnod_local, mesh.Nnod, mesh.Nnod, 0, 0, 0, 0, &Qmat);
	//MatCreateMPIAIJ(PETSC_COMM_WORLD, Nedge_local, Nnod_local, mesh.Nedge, mesh.Nnod, 0, 0, 0, 0, &Cmat);
	//MatCreateMPIAIJ(PETSC_COMM_WORLD, Nedge_local, Nnod_local, mesh.Nedge, mesh.Nnod, 0, 0, 0, 0, &Rx);
	//MatCreateMPIAIJ(PETSC_COMM_WORLD, Nedge_local, Nnod_local, mesh.Nedge, mesh.Nnod, 0, 0, 0, 0, &Ry);
	//MatCreateMPIAIJ(PETSC_COMM_WORLD, Nedge_local, Nnod_local, mesh.Nedge, mesh.Nnod, 0, 0, 0, 0, &Rz);


	//-------------allocate memory automatically (note malloc required)---------------
	//Mat_Create(Amat, Nedge_local, Nedge_local, mesh.Nedge, mesh.Nedge); //symm
	//Mat_Create(Mmat, Nedge_local, Nedge_local, mesh.Nedge, mesh.Nedge); //symm
	//Mat_Create(Lmat, Nnod_local, Nnod_local, mesh.Nnod, mesh.Nnod); //symm
	//Mat_Create(Bmat, Nnod_local, Nedge_local, mesh.Nnod, mesh.Nedge);
	//Vec_Create(f, Nedge_local, mesh.Nedge);
	//Vec_Create(g, Nnod_local, mesh.Nnod);

	//Mat_Create(Qmat, Nnod_local, Nnod_local, mesh.Nnod, mesh.Nnod); //symm
	//Mat_Create(Cmat, Nedge_local, Nnod_local, mesh.Nedge, mesh.Nnod);
	//Mat_Create(Rx, Nedge_local, Nnod_local, mesh.Nedge, mesh.Nnod);
	//Mat_Create(Ry, Nedge_local, Nnod_local, mesh.Nedge, mesh.Nnod);
	//Mat_Create(Rz, Nedge_local, Nnod_local, mesh.Nedge, mesh.Nnod);


	int start, end;
	//the following is useless, just to ensure MPI communication works appropriately
	//MatGetOwnershipRange(Amat,&start,&end);
	//MatGetOwnershipRange(Mmat,&start,&end);
	//MatGetOwnershipRange(Lmat,&start,&end);
	//MatGetOwnershipRange(Bmat,&start,&end);
	//VecGetOwnershipRange(f, &start, &end);
	//VecGetOwnershipRange(g, &start, &end);

	//MatGetOwnershipRange(Qmat,&start,&end);
	//MatGetOwnershipRange(Cmat,&start,&end);
	//MatGetOwnershipRange(Rx,&start,&end);
	//MatGetOwnershipRange(Ry,&start,&end);
	//MatGetOwnershipRange(Rz,&start,&end);



	//----------------------------
	//assemble elems on local proc
	//----------------------------
	start=0;
	for (int i=0; i<RANK; i++) start+=mesh.elem_count[i];
	end = start+mesh.elem_count[RANK];


	//------------------------------------
	// allocate space for entries to be inserted
	//------------------------------------
	//AIJ *AKArray = new AIJ[6*6*(end-start)];
	//AIJ *MKArray = new AIJ[6*6*(end-start)];
	//AIJ *BBKArray = new AIJ[4*6*(end-start)];
	//AIJ *LKArray = new AIJ[4*4*(end-start)];
	//AIJ *QKArray = new AIJ[4*4*(end-start)];
	//AIJ *fKArray = new AIJ[6*(end-start)];

	double *AKArray = new double[36];
	double *MKArray = new double[36];
	double *BBKArray = new double[24];
	double *LKArray = new double[16];
	double *QKArray = new double[16];
	double *fKArray = new double[6];



	for (int n=start; n<end; n++) {

		//initilize AK, MK, BK, fK, LK, QK
		vector<vector<double> > AK(6, vector<double>(6, 0));
		vector<vector<double> > MK(6, vector<double>(6, 0));
		vector<vector<double> > BBK(4, vector<double>(6, 0));
		vector<vector<double> > LK(4, vector<double>(4, 0));
		vector<vector<double> > QK(4, vector<double>(4, 0));
		vector<double> fK(6, 0);


		//get coords of local nodes
		int vert[4]; //index of local vertices
		double pt[4][3]; //coordinates of local vertices
		for (int i=0; i<4; i++) {
			vert[i] = mesh.elems[n][i];
			for (int j=0; j<3; j++)
				pt[i][j] = mesh.coords[vert[i]][j];
		}

		double bK[3] = {pt[0][0], pt[0][1], pt[0][2]};
		//BK = [pt2 - pt1, pt3 - pt1];
		double BK[3][3], BKI[3][3],detBK;
		for (int i=0; i<3; i++)
			for (int j=0; j<3; j++) {
				BK[i][j]=pt[j+1][i] - pt[0][i];	//Note: Jacobian is the tranpose of DK!
			}

		//gradL = gradLhat*inv(BK);
		detBK = mat_det(3,(double*)BK);		//Note: Jacobian is the tranpose of DK!
		mat_inv(3, (double*)BK, (double*)BKI);
		mat_mat_mult(4,3,3,(double*)gradLhat,(double*)BKI, (double*)gradL);

		if (detBK<0) PetscPrintf(PETSC_COMM_WORLD, "????????????????????wrong mesh??????????????????????????????\n");

		//find edgedir
		int edgedir[6], edgenum[6];
		for (int i=0; i<6; i++) {
			int spt, ept;
			spt = vert[edgeK[0][i]];
			ept = vert[edgeK[1][i]];
			if (spt>ept){
				edgedir[i]  = -1;
			}
			else {
				edgedir[i]  = 1;
			}

			edgenum[i] = mesh.edges_elem[n][i];
			//double val;
			//MatGetValues(mesh.EdgeIndx, 1, &spt, 1, &ept, &val);
			//edgenum[i] = (int)val;
		}


		// pre-calculate dot(cross(gradL, gradL))
		double dotcrossgradL[6][6];
		for (int i=0; i<6; i++) {
			int i1, i2;
			double t1[3], t2[3];
			i1 = edgeK[0][i];
			i2 = edgeK[1][i];
			mycross(3, gradL[i1], gradL[i2], t1);
			for (int j=0; j<6; j++) {
				int j1, j2;
				j1 = edgeK[0][j];
				j2 = edgeK[1][j];
				mycross(3, gradL[j1], gradL[j2], t2);
				dotcrossgradL[i][j] = mydot(3, t1, t2);
			}
		}

		for (int k = 0; k<lw; k++) {
			double pthat[3] = {xnod[k], ynod[k], znod[k]};
			double pt[3], v[3];
			//pt = bK + BK * pthat;
			mat_vec_mult(3,3,(double*)BK, pthat, v);
			vec_sum(3, 1, 1, bK, v, pt);
			L[0] = 1 - xnod[k] - ynod[k] - znod[k];
			L[1] = xnod[k];
			L[2] = ynod[k];
			L[3] = znod[k];

			//calculte N[6][3], Nedelec shape functions
			double N[6][3];
			for (int i=0; i<6; i++) {
				int i1, i2;
				i1 = edgeK[0][i];
				i2 = edgeK[1][i];
				vec_sum(3, edgedir[i]*L[i1], -edgedir[i]*L[i2], gradL[i2], gradL[i1], N[i]);
			}

			func_f(pt, v);
			double mu = func_mu(pt);
			double eps = func_eps(pt);

			for (int i=0; i<6; i++) {
				for (int j=i; j<6; j++){
					AK[i][j] += w[k]*4*edgedir[i]*edgedir[j]*dotcrossgradL[i][j]/mu;
					MK[i][j] += w[k]*eps*mydot(3,N[i], N[j]);
				}
				fK[i] += w[k]*mydot(3,v,N[i]);
				for (int j=0; j<4; j++){
					BBK[j][i] += w[k]*eps*mydot(3,N[i],gradL[j]);
				}
			}

			//compute scalar laplace
			for (int i=0; i<4; i++)
				for (int j=i; j<4; j++) {
					LK[i][j] += w[k]*eps*mydot(3, gradL[i], gradL[j]);
					QK[i][j] += w[k]*eps*L[i]*L[j];
				}

		}

		////compute AK without integration
		//for (int i=0; i<6; i++)
		//	for (int j=0; j<6; j++) {
		//		AK[i][j] = 4*edgedir[i]*edgedir[j]*dotcrossgradL[i][j]/6.0;
		//	}
		// compute the lower tri part
		for (int i=1; i<6; i++)
			for (int j=0; j<i; j++) {
				AK[i][j] = AK[j][i];
				MK[i][j] = MK[j][i];
			}

		for (int i=1; i<4; i++)
			for (int j=0; j<i; j++) {
				LK[i][j] = LK[j][i];
				QK[i][j] = QK[j][i];
			}


		mat_scale(AK, detBK);
		mat_scale(MK, detBK);
		mat_scale(BBK, detBK);
		mat_scale(LK, detBK);
		mat_scale(QK, detBK);
		vec_scale(fK, detBK);


		//--------------------------------------------------------------------
		//---------Start Assembling------------------------------------------
		//double constk = 0.0;

		for (int i=0; i<4; i++)
			for (int j=0; j<4; j++){
				//MatSetValue(Lmat, vert[i], vert[j], LK[i][j], ADD_VALUES);
				//MatSetValue(Qmat, vert[i], vert[j], QK[i][j], ADD_VALUES);
				LKArray[i*4+j] = LK[i][j];
				QKArray[i*4+j] = QK[i][j];
			}

		//set Stiffmat, mypc, rhs
		for (int i=0; i<6; i++) {
			for (int j=0; j<6; j++) {
				//MatSetValue(Amat, edgenum[i], edgenum[j],  AK[i][j], ADD_VALUES);
				//MatSetValue(Mmat, edgenum[i], edgenum[j],  MK[i][j], ADD_VALUES);
				AKArray[i*6+j] = AK[i][j];
				MKArray[i*6+j] = MK[i][j];
			}
			//VecSetValue(f, edgenum[i],  fK[i], ADD_VALUES);
			fKArray[i] = fK[i];

		}

		for (int i=0; i<4; i++)
			for (int j=0; j<6; j++) {
				//MatSetValue(Bmat, vert[i], edgenum[j], BBK[i][j], ADD_VALUES);
				BBKArray[i*6+j] = BBK[i][j];
			}

		MatSetValues(Lmat, 4, vert, 4, vert, LKArray, ADD_VALUES);
		MatSetValues(Qmat, 4, vert, 4, vert, QKArray, ADD_VALUES);
		MatSetValues(Amat, 6, edgenum, 6, edgenum, AKArray, ADD_VALUES);
		MatSetValues(Mmat, 6, edgenum, 6, edgenum, MKArray, ADD_VALUES);
		VecSetValues(f, 6, edgenum, fKArray, ADD_VALUES);
		MatSetValues(Bmat, 4, vert, 6, edgenum, BBKArray, ADD_VALUES);
	}

	delete[] AKArray;
	delete[] MKArray;
	delete[] BBKArray;
	delete[] LKArray;
	delete[] QKArray;
	delete[] fKArray;

	// find the CSR format arrays from AKArray
	//int *CSRi = new int[max(mesh.Nedge, mesh.Nnod)+1]; //allocate enough space for ALL matrices!!!!
	//int *CSRj = new int[6*6*(end-start)]; //allocate enough space
	//double *CSRv = new double[6*6*(end-start)];	//allocate enough space for ALL matrices
	//AIJCompare AIJCompareObj;

	//sort(AKArray, AKArray+6*6*(end-start), AIJCompareObj);

	//array2CSR(AKArray, 6*6*(end-start), Nedge_local, CSRi, CSRj, CSRv);


	////MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, Nedge_local, Nedge_local, mesh.Nedge, mesh.Nedge, CSRi, CSRj, CSRv, &Amat );
	//Mat_Assemble(Amat);

	//delete[] CSRi;
	//delete[] CSRj;
	//delete[] CSRv;

	//----------------------------
	// loop over edges on local proc
	//----------------------------
	start=0;
	for (int i=0; i<RANK; i++) start+=mesh.edge_count_i[i]+mesh.edge_count_b[i];
	end = start+mesh.edge_count_i[RANK]+mesh.edge_count_b[RANK];

	for (int n=start; n<end; n++) {
		int pts[2] = {mesh.edges[n][0], mesh.edges[n][1]};

		//assemble Cmat
		MatSetValue(Cmat, n, pts[0], -1, INSERT_VALUES);
		MatSetValue(Cmat, n, pts[1], 1, INSERT_VALUES);


		//assemble Rx,
		Ry, Rz
		PetscScalar spt[3], ept[3];
		for (int i=0; i<3; i++) {
			spt[i] = mesh.coords[pts[0]][i];
			ept[i] = mesh.coords[pts[1]][i];
		}
		double tandir[3];
		vec_sum(3,-1,1,spt, ept, tandir);
		double len = vec_normal(3, tandir);
		vec_scale(3, 1.0/len, tandir, tandir);

		// no need to do MatSetValues, only two entries to set per interation
		MatSetValue(Rx, n, pts[0], 0.5*tandir[0]*len, ADD_VALUES);
		MatSetValue(Ry, n, pts[0], 0.5*tandir[1]*len, ADD_VALUES);
		MatSetValue(Rz, n, pts[0], 0.5*tandir[2]*len, ADD_VALUES);
		MatSetValue(Rx, n, pts[1], 0.5*tandir[0]*len, ADD_VALUES);
		MatSetValue(Ry, n, pts[1], 0.5*tandir[1]*len, ADD_VALUES);
		MatSetValue(Rz, n, pts[1], 0.5*tandir[2]*len, ADD_VALUES);
	}


	Mat_Assemble(Amat);
	Mat_Assemble(Mmat);
	Mat_Assemble(Bmat);
	Mat_Assemble(Lmat);
	Mat_Assemble(Qmat);
	Mat_Assemble(Cmat);
	Mat_Assemble(Rx);
	Mat_Assemble(Ry);
	Mat_Assemble(Rz);
	Vec_Assemble(f); //no need for Vec_Assemble(g); g value is not changed

	//print memory usage
	#ifndef _WIN32
	sysfile.open (buf, fstream::in);
	sysfile>>programSize>>memorySize;
	sysfile.close();
	PetscPrintf(PETSC_COMM_WORLD, "	--- Memory usage when Assemble ends\n [proc %d] Program size %.4f Mb, memory size %.4f Mb\n\n", RANK, programSize/1024.0, memorySize/1024.0);
	#endif


}


void SYSTEM::Apply_BC()
{
	//-------------------------
	// apply BC
	//-------------------------
	Mat Aii, Abi, tmpmat;
	Mat Bii, Bbi, Bib;
	Mat Mbi;
	IS is_global_edge_i;
	IS is_global_edge_b;
	IS is_global_nod_i;

	ISAllGather(mesh.is_edge_i,&is_global_edge_i);
	ISAllGather(mesh.is_edge_b,&is_global_edge_b);
	ISAllGather(mesh.is_nod_i,&is_global_nod_i);

	MatGetSubMatrix(Mmat, mesh.is_edge_i, is_global_edge_i, mesh.edge_count_i[RANK], MAT_INITIAL_MATRIX, &tmpmat);
	MatGetSubMatrix(Mmat, mesh.is_edge_b, is_global_edge_i, mesh.edge_count_i[RANK], MAT_INITIAL_MATRIX, &Mbi);
	MatDestroy(Mmat);
	Mmat = tmpmat;

	MatGetSubMatrix(Lmat, mesh.is_nod_i, is_global_nod_i, mesh.nod_count_i[RANK], MAT_INITIAL_MATRIX, &tmpmat);
	MatDestroy(Lmat);
	Lmat = tmpmat;

	MatGetSubMatrix(Qmat, mesh.is_nod_i, is_global_nod_i, mesh.nod_count_i[RANK], MAT_INITIAL_MATRIX, &tmpmat);
	MatDestroy(Qmat);
	Qmat = tmpmat;

	MatGetSubMatrix(Cmat, mesh.is_edge_i, is_global_nod_i, mesh.nod_count_i[RANK], MAT_INITIAL_MATRIX, &tmpmat);
	MatDestroy(Cmat);
	Cmat = tmpmat;

	MatGetSubMatrix(Rx, mesh.is_edge_i, is_global_nod_i, mesh.nod_count_i[RANK], MAT_INITIAL_MATRIX, &tmpmat);
	MatDestroy(Rx);
	Rx = tmpmat;
	MatGetSubMatrix(Ry, mesh.is_edge_i, is_global_nod_i, mesh.nod_count_i[RANK], MAT_INITIAL_MATRIX, &tmpmat);
	MatDestroy(Ry);
	Ry = tmpmat;
	MatGetSubMatrix(Rz, mesh.is_edge_i, is_global_nod_i, mesh.nod_count_i[RANK], MAT_INITIAL_MATRIX, &tmpmat);
	MatDestroy(Rz);
	Rz = tmpmat;

	MatGetSubMatrix(Amat, mesh.is_edge_i, is_global_edge_i, mesh.edge_count_i[RANK], MAT_INITIAL_MATRIX, &Aii);
	MatGetSubMatrix(Amat, mesh.is_edge_b, is_global_edge_i, mesh.edge_count_i[RANK], MAT_INITIAL_MATRIX, &Abi);
	MatDestroy(Amat);
	Amat = Aii;
	MatGetSubMatrix(Bmat, mesh.is_nod_i, is_global_edge_i, mesh.edge_count_i[RANK], MAT_INITIAL_MATRIX, &Bii);
	MatGetSubMatrix(Bmat, mesh.is_nod_b, is_global_edge_i, mesh.edge_count_i[RANK], MAT_INITIAL_MATRIX, &Bbi);
	MatGetSubMatrix(Bmat, mesh.is_nod_i, is_global_edge_b, mesh.edge_count_b[RANK], MAT_INITIAL_MATRIX, &Bib);
	MatDestroy(Bmat);
	Bmat = Bii;

	//MatInfo info;
 //   MatGetInfo(Aii,MAT_GLOBAL_SUM,&info);
	//PetscPrintf(PETSC_COMM_SELF, "[%d] Aii(%g,%g), local (%g,%g)\n",RANK, info.rows_global, info.columns_global, info.rows_local, info.columns_local);
 //   MatGetInfo(Abi,MAT_GLOBAL_SUM,&info);
	//PetscPrintf(PETSC_COMM_SELF, "[%d] Abi(%g,%g), local (%g,%g)\n",RANK, info.rows_global, info.columns_global, info.rows_local, info.columns_local);
 //   MatGetInfo(Bii,MAT_GLOBAL_SUM,&info);
	//PetscPrintf(PETSC_COMM_SELF, "[%d] Bii(%g,%g), local (%g,%g)\n",RANK, info.rows_global, info.columns_global, info.rows_local, info.columns_local);
 //   MatGetInfo(Bbi,MAT_GLOBAL_SUM,&info);
	//PetscPrintf(PETSC_COMM_SELF, "[%d] Bbi(%g,%g), local (%g,%g)\n",RANK, info.rows_global, info.columns_global, info.rows_local, info.columns_local);
 //   MatGetInfo(Bib,MAT_GLOBAL_SUM,&info);
	//PetscPrintf(PETSC_COMM_SELF, "[%d] Bib(%g,%g), local (%g,%g)\n",RANK, info.rows_global, info.columns_global, info.rows_local, info.columns_local);

	ISDestroy(is_global_edge_i);
	ISDestroy(is_global_edge_b);
	ISDestroy(is_global_nod_i);

	//ISView(mesh.is_edge_i,PETSC_VIEWER_STDOUT_WORLD);

	//PetscPrintf(PETSC_COMM_WORLD, "Viewing Vec f\n");
	//VecView(f,PETSC_VIEWER_STDOUT_WORLD);


	// modify ue, pe
	Vec ub, pb, ui, pi;
	MatGetVecs(Bib, &ub, 0);
	MatGetVecs(Bbi, 0, &pb);
	MatGetVecs(Aii, 0, &ui);
	MatGetVecs(Bii, 0, &pi);

	int *indx;
	double *val;
	int t=0;

	#ifdef _WIN32
	const int *tmp_indx;
	#else
	int *tmp_indx; //const int for petsc3.0.0, int for 2.3.3
	#endif

	// get ub from ue
	val = new double[mesh.edge_count_b[RANK]];
	ISGetIndices(mesh.is_edge_b, &tmp_indx);
	if (mesh.edge_count_b[RANK]) VecGetValues(ue, mesh.edge_count_b[RANK], tmp_indx, val);
	ISRestoreIndices(mesh.is_edge_b, &tmp_indx);
	indx = new int[mesh.edge_count_b[RANK]];
	t=0;
	for (int i=0; i<RANK; i++) t+=mesh.edge_count_b[i];
	for (int i=0; i<mesh.edge_count_b[RANK]; i++) indx[i]=t+i;
	if (mesh.edge_count_b[RANK]) VecSetValues(ub, mesh.edge_count_b[RANK], indx, val, INSERT_VALUES);
	Vec_Assemble(ub);
	delete[]indx;
	delete[] val;

	// get pb from pe
	val = new double[mesh.nod_count_b[RANK]];
	ISGetIndices(mesh.is_nod_b, &tmp_indx);
	if (mesh.nod_count_b[RANK]) VecGetValues(pe, mesh.nod_count_b[RANK], tmp_indx, val);
	ISRestoreIndices(mesh.is_nod_b, &tmp_indx);
	indx = new int[mesh.nod_count_b[RANK]];
	t=0;
	for (int i=0; i<RANK; i++) t+=mesh.nod_count_b[i];
	for (int i=0; i<mesh.nod_count_b[RANK]; i++) indx[i]=t+i;
	if (mesh.nod_count_b[RANK]) VecSetValues(pb, mesh.nod_count_b[RANK], indx, val, INSERT_VALUES);
	Vec_Assemble(pb);
	delete[]indx;
	delete[] val;

	//PetscPrintf(PETSC_COMM_WORLD, "Viewing Vec ub\n");
	//VecView(ub,PETSC_VIEWER_STDOUT_WORLD);
	//PetscPrintf(PETSC_COMM_WORLD, "Viewing Vec pb\n");
	//VecView(pb,PETSC_VIEWER_STDOUT_WORLD);

	//change ue, be according to BC, i.e. remove BC from the vector

	// get ui from ue
	val = new double[mesh.edge_count_i[RANK]];
	ISGetIndices(mesh.is_edge_i, &tmp_indx);
	if (mesh.edge_count_i[RANK]) VecGetValues(ue, mesh.edge_count_i[RANK], tmp_indx, val);
	ISRestoreIndices(mesh.is_edge_i, &tmp_indx);
	indx = new int[mesh.edge_count_i[RANK]];
	t=0;
	for (int i=0; i<RANK; i++) t+=mesh.edge_count_i[i];
	for (int i=0; i<mesh.edge_count_i[RANK]; i++) indx[i]=t+i;
	if (mesh.edge_count_i[RANK]) VecSetValues(ui, mesh.edge_count_i[RANK], indx, val, INSERT_VALUES);
	Vec_Assemble(ui);
	delete[]indx;
	delete[] val;
	VecDestroy(ue);
	ue=ui;

	// get pi from pe
	val = new double[mesh.nod_count_i[RANK]];
	ISGetIndices(mesh.is_nod_i, &tmp_indx);
	if (mesh.nod_count_i[RANK]) VecGetValues(pe, mesh.nod_count_i[RANK], tmp_indx, val);
	ISRestoreIndices(mesh.is_nod_i, &tmp_indx);
	indx = new int[mesh.nod_count_i[RANK]];
	t=0;
	for (int i=0; i<RANK; i++) t+=mesh.nod_count_i[i];
	for (int i=0; i<mesh.nod_count_i[RANK]; i++) indx[i]=t+i;
	if (mesh.nod_count_i[RANK]) VecSetValues(pi, mesh.nod_count_i[RANK], indx, val, INSERT_VALUES);
	Vec_Assemble(pi);
	delete[]indx;
	delete[] val;
	VecDestroy(pe);
	pe=pi;

	//modify exact sol
	VecCreate( PETSC_COMM_WORLD, &exactsol );
	VecSetSizes( exactsol, 2, 2 );
	VecSetType( exactsol, "block" );
	VecBlockSetValue( exactsol, 0, ue, INSERT_VALUES );
	VecBlockSetValue( exactsol, 1, pe, INSERT_VALUES );
	Vec_Assemble(exactsol);

	//modify f
	Vec fi, tmpf;
	MatGetVecs(Aii, 0, &fi);
	val = new double[mesh.edge_count_i[RANK]];
	ISGetIndices(mesh.is_edge_i, &tmp_indx);
	if (mesh.edge_count_i[RANK]) VecGetValues(f, mesh.edge_count_i[RANK], tmp_indx, val);
	ISRestoreIndices(mesh.is_edge_i, &tmp_indx);
	indx = new int[mesh.edge_count_i[RANK]];
	t=0;
	for (int i=0; i<RANK; i++) t+=mesh.edge_count_i[i];
	for (int i=0; i<mesh.edge_count_i[RANK]; i++) indx[i]=t+i;
	if (mesh.edge_count_i[RANK]) VecSetValues(fi, mesh.edge_count_i[RANK], indx, val, INSERT_VALUES);
	Vec_Assemble(fi);
	VecDestroy(f);
	delete[] indx;
	delete[] val;
	f = fi;

	// f = f -Abi'*ub + k^2*Mbi'*ub - Bbi'*pb
	VecDuplicate(f, &tmpf);
	MatMultTranspose(Abi, ub, tmpf); //tempf =Abi'*ub
	VecAXPY(f, -1, tmpf);
	MatMultTranspose(Mbi, ub, tmpf); //tempf =k^2*Mbi'*ub
	VecAXPY(f, paramk*paramk, tmpf);
	MatMultTranspose(Bbi, pb, tmpf); //tempf = Bbi*pb
	VecAXPY(f, -1, tmpf);
	VecDestroy(tmpf);



	//modify g
	// g = -Bib*ub
	Vec gi;
	MatGetVecs(Bii, 0, &gi);
	MatMult(Bib, ub, gi);
	VecScale(gi, -1.0);
	//replace g with gi
	VecDestroy(g);
	g = gi;

	//PetscPrintf(PETSC_COMM_WORLD, "Viewing Vec fi\n");
	//VecView(f,PETSC_VIEWER_STDOUT_WORLD);
	//PetscPrintf(PETSC_COMM_WORLD, "Viewing Vec gi\n");
	//VecView(g,PETSC_VIEWER_STDOUT_WORLD);

	//releae memory
	VecDestroy(ub); //dont release ui, pi!!!
	VecDestroy(pb);

	MatDestroy(Abi); //don't release space for Aii and Bii, they are now Amat and Bmat!!!
	MatDestroy(Mbi);
	MatDestroy(Bbi);
	MatDestroy(Bib);

	//PetscPrintf(PETSC_COMM_WORLD, "--Amat-----\n");
	//MatView(Amat, PETSC_VIEWER_STDOUT_WORLD);
	//PetscPrintf(PETSC_COMM_WORLD, "--Bmat-----\n");
	//MatView(Bmat, PETSC_VIEWER_STDOUT_WORLD);
	//PetscPrintf(PETSC_COMM_WORLD, "--f-----\n");
	//VecView(f, PETSC_VIEWER_STDOUT_WORLD);
	//PetscPrintf(PETSC_COMM_WORLD, "--g-----\n");
	//VecView(g, PETSC_VIEWER_STDOUT_WORLD);
	//PetscPrintf(PETSC_COMM_WORLD, "--ue-----\n");
	//VecView(ue, PETSC_VIEWER_STDOUT_WORLD);
	//PetscPrintf(PETSC_COMM_WORLD, "--pe-----\n");
	//VecView(pe, PETSC_VIEWER_STDOUT_WORLD);



}

void SYSTEM::Form_System()
{
	// Build a block system for the maxwell problem,
	// No need to form the preconditioner explicitly, use sub-blocks instead
	MatCreate( PETSC_COMM_WORLD, &Stiffmat );
	MatSetSizes( Stiffmat, 2,2, 2,2 );
	MatSetType( Stiffmat, "block" );

	Mat BT;
	MatCreateSymTrans( PETSC_COMM_WORLD, Bmat, &BT );

	Mat AmM; //AmM = A-k^2*M
	MatDuplicate(Amat,MAT_COPY_VALUES,&AmM);
	MatAXPY(AmM, -paramk*paramk, Mmat, DIFFERENT_NONZERO_PATTERN);

	// Prescribe sub matrices within the block
	MatBlockSetValue( Stiffmat, 0,0, AmM, DIFFERENT_NONZERO_PATTERN, INSERT_VALUES );
	MatBlockSetValue( Stiffmat, 0,1, BT, DIFFERENT_NONZERO_PATTERN, INSERT_VALUES );
	MatBlockSetValue( Stiffmat, 1,0, Bmat, DIFFERENT_NONZERO_PATTERN, INSERT_VALUES );
	Mat_Assemble(Stiffmat);

	//releae memory allocated for constructing Stiffmat
	MatDestroy(AmM);
	MatDestroy(BT);

	//allocate space for sol and rhs
	MatGetVecs( Stiffmat, 0, &rhs );

	// Prescribe sub vectors within the rhs
	VecBlockSetValue( rhs, 0, f, INSERT_VALUES );
	VecBlockSetValue( rhs, 1, g, INSERT_VALUES );
	Vec_Assemble(rhs);

}

#ifdef _WIN32
void SYSTEM::Solve(void)
{
}
#else

void SYSTEM::Solve(void)
{
	PetscErrorCode ierr;

	//----------------------------------
	// construct sub ksp's
	//----------------------------------
	KSP subksp1, subksp2;
	PC subpc1, subpc2;

	//------------------
	//construct subksp1
	//------------------
	Mat ApM; //ApM = A+(1-k^2)*M
	MatDuplicate(Amat,MAT_COPY_VALUES,&ApM);
	MatAXPY(ApM, 1-paramk*paramk, Mmat, DIFFERENT_NONZERO_PATTERN);

	KSPCreate(PETSC_COMM_WORLD, &subksp1);
	KSPSetOperators(subksp1, ApM, ApM, SAME_NONZERO_PATTERN);
	KSPGetPC(subksp1, &subpc1);
	KSPSetOptionsPrefix(subksp1, "ksp1_");
	PCSetOptionsPrefix(subpc1, "ksp1_");
	KSPSetType(subksp1,"cg");

	//hypre pc
	//PCSetType(subpc1, "hypre");
	//PetscOptionsSetValue("-ksp1_pc_hypre_type","boomeramg");
	//ierr = PCSetFromOptions(subpc1);
	//ierr = KSPSetFromOptions(subksp1);
	//char *pn1;
	//PCHYPREGetType(subpc1, (const char**)&pn1);
	//PetscPrintf(PETSC_COMM_WORLD,"(1,1): %s\n", pn1);

	//shell pc
	Hiptmair  shell;
	Mat LQ;	 //LQ = L+(1-k^2)*Q, to be set in MyshellSetup
	MyshellSetup(&shell, LQ);
	PCSetType(subpc1,PCSHELL);
	PCShellSetApply(subpc1,MyShellPCApply);
	PCShellSetContext(subpc1,&shell);

	KSPSetTolerances(subksp1,1.e-6,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
	KSPSetFromOptions(subksp1);


	//-------------------
	//construct subksp2
	//-------------------

	// run the program with -help to see options for hypre, check http://www.mcs.anl.gov/petsc/petsc-2/documentation/faq.html
	// Run with -ksp_view to see all the hypre options used and -help | grep boomeramg to see all the command line options.
	// Note KSPType "preonly" will only apply one v-cycle, if multiple cycles are wanted, use "richardson"
	// but, "richardson" doesn't give the right # of iterations.(why????)
	KSPCreate(PETSC_COMM_WORLD, &subksp2);
	KSPSetOperators(subksp2, Lmat, Lmat, SAME_NONZERO_PATTERN);
	KSPGetPC(subksp2, &subpc2);
	KSPSetOptionsPrefix(subksp2, "ksp2_");
	PCSetOptionsPrefix(subpc2, "ksp2_");

	KSPSetType(subksp2,"cg");
	PCSetType(subpc2, "hypre");
	PetscOptionsSetValue("-ksp2_pc_hypre_type","boomeramg");
	PetscOptionsSetValue("-ksp2_pc_hypre_boomeramg_strong_threshold", "0.5");
	//"-pc_hypre_boomeramg_P_max","Max elements per row for interpolation operator ( 0=unlimited )","
	//PetscOptionsSetValue("-ksp2_pc_hypre_boomeramg_P_max","50");
	//PCSetType(subpc2, "ml");
	//PetscOptionsSetValue("-ksp2_pc_hypre_type","pilut");

	PCSetFromOptions(subpc2); //will this work???? Yup!

	KSPSetTolerances(subksp2,1.e-6,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
	KSPSetFromOptions(subksp2);


	//---------------------------------------------
	// test sub ksp's as stand alone solver
	//---------------------------------------------
	Vec bb, xx, uu;
	clock_t t1, t2;
	double tused, tavg;
	double norm;
	int its;


	//-----------------------------------
	//use subksp1 as a stand alone solver
	//-----------------------------------
	MatGetVecs(Amat,&uu,&bb);
	VecDuplicate(uu,&xx);
	VecSet(uu,1.0);
	MatMult(ApM,uu,bb);

	t1 = clock();
	KSPSolve(subksp1,bb,xx);
	t2 = clock();
	tused = (t2-t1)/(double)CLOCKS_PER_SEC;
	MPI_Reduce(&tused, &tavg, 1, MPI_DOUBLE,MPI_SUM, 0, PETSC_COMM_WORLD);
	tavg /= (double)SIZE;
	PetscPrintf(PETSC_COMM_WORLD,"		subksp[1]: Time spend in solve = %g\n", tavg);
	PetscFPrintf(PETSC_COMM_WORLD, stream, "		[sub ksp1]:	Time spend in solve = %g\n", tavg);

	VecAXPY(xx,-1.0,uu);
	VecNorm(xx,NORM_2,&norm);
	KSPGetIterationNumber(subksp1,&its);
	PetscPrintf(PETSC_COMM_WORLD,"		subksp[1]: Norm of error %g, Outer Iterations %d\n", norm,its);
	PetscFPrintf(PETSC_COMM_WORLD, stream, "		[sub ksp1]: Norm of error %g, Outer Iterations %d\n", norm,its);

	VecDestroy(bb);
	VecDestroy(xx);
	VecDestroy(uu);

	//-----------------------------------
	//use subksp2 as a stand alone solver
	//-----------------------------------
	MatGetVecs(Lmat,&uu,&bb);
	VecDuplicate(uu,&xx);
	VecSet(uu,1.0);
	MatMult(Lmat,uu,bb);

	char *pn2;
	PCHYPREGetType(subpc2, (const char**)&pn2);
	PetscPrintf(PETSC_COMM_WORLD,"		subksp[2]: PCHypreType = %s\n", pn2);

	//-----------------------------
	//test preformance of one V-cycle. comment it out, if don't want to test
	KSPSetType(subksp2,"preonly");
	t1 = clock();
	KSPSolve(subksp2,bb,xx);
	t2 = clock();
	tused = (t2-t1)/(double)CLOCKS_PER_SEC;
	MPI_Reduce(&tused, &tavg, 1, MPI_DOUBLE,MPI_SUM, 0, PETSC_COMM_WORLD);
	tavg /= (double)SIZE;
	PetscPrintf(PETSC_COMM_WORLD,"		subksp[2]: Time spend in one V-cyc = %g\n", tavg);
	PetscFPrintf(PETSC_COMM_WORLD, stream, "		[sub ksp2]:	Time spend in one V-cyc = %g\n", tavg);
	KSPSetType(subksp2,"cg");
	//------------------------------


	t1 = clock();
	KSPSolve(subksp2,bb,xx);
	t2 = clock();
	tused = (t2-t1)/(double)CLOCKS_PER_SEC;
	MPI_Reduce(&tused, &tavg, 1, MPI_DOUBLE,MPI_SUM, 0, PETSC_COMM_WORLD);
	tavg /= (double)SIZE;
	PetscPrintf(PETSC_COMM_WORLD,"		subksp[2]: Time spend in solve = %g\n", tavg);
	PetscFPrintf(PETSC_COMM_WORLD, stream, "		[sub ksp2]:	Time spend in solve = %g\n", tavg);

	VecAXPY(xx,-1.0,uu);
	VecNorm(xx,NORM_2,&norm);
	KSPGetIterationNumber(subksp2,&its);
	PetscPrintf(PETSC_COMM_WORLD,"		subksp[2]: Norm of error %g, Outer Iterations %d\n\n", norm,its);
	PetscFPrintf(PETSC_COMM_WORLD, stream, "		[sub ksp2]: Norm of error %g, Outer Iterations %d\n\n", norm,its);

	VecDestroy(bb);
	VecDestroy(xx);
	VecDestroy(uu);

	//------------------------------------------------------
	// solve the block linear system
	//------------------------------------------------------


	//--------------------------------------
	// construct the global block ksp
	//--------------------------------------

	// construct block matrix mypc
	Mat mypc;
	MatCreate(PETSC_COMM_WORLD, &mypc);
	MatSetSizes(mypc, 2,2, 2,2);
	MatSetType(mypc, "block");

	// Prescribe sub matrices within the block
	MatBlockSetValue(mypc, 0,0, ApM, DIFFERENT_NONZERO_PATTERN, INSERT_VALUES);
	MatBlockSetValue(mypc, 1,1, Lmat, DIFFERENT_NONZERO_PATTERN, INSERT_VALUES);
	Mat_Assemble(mypc);


	//--------------------------------------
	// construct the global block ksp
	//--------------------------------------
	KSP	ksp;	//linear solver context
	PC	pc;		//preconditioner context
	KSPCreate(PETSC_COMM_WORLD,&ksp);
	KSPSetOperators(ksp,Stiffmat,mypc,SAME_NONZERO_PATTERN);
	KSPSetType(ksp,KSPMINRES);
	KSPGetPC(ksp,&pc);

	PCSetType(pc,"block");
	//ver2.3.3, PCBlock_SetBlockType; ver3.0.0 PCBlockSetBlockType
	//PCBlockSetBlockType(pc,PC_BLOCK_DIAGONAL); // petsc 3.0.0
	PCBlock_SetBlockType(pc,PC_BLOCK_DIAGONAL); //petsc 2.3.3

	//PetscViewerSetFormat( PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL );
	//KSPView( ksp, PETSC_VIEWER_STDOUT_WORLD );
	//PCView( pc, PETSC_VIEWER_STDOUT_WORLD );


	PCBlockSetSubKSP(pc, 0, subksp1);
	PCBlockSetSubKSP(pc, 1, subksp2);
	ierr = KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
	KSPSetFromOptions(ksp);


	//--------------------------------------
	//Solve the linear system
	//--------------------------------------
	//Vec sol;
	time_t stime, etime;
	VecDuplicate(rhs,&sol);
	t1 = clock();
	time( &stime );
	//PetscPrintf(PETSC_COMM_SELF,"	[%d]  Solve starts at %s\n",RANK, ctime( &ltime ));
	//PetscFPrintf(PETSC_COMM_SELF, stream, "	[%d]  Solve starts at %s\n",RANK, ctime( &ltime ));
	ierr = KSPSolve(ksp,rhs,sol);
	time( &etime );
	double dif = difftime (etime, stime);
	//PetscPrintf(PETSC_COMM_SELF,"	[%d]  Solve ends at %s\n",RANK, ctime( &ltime ));
	//PetscFPrintf(PETSC_COMM_SELF, stream, "	[%d]  Solve ends at %s\n",RANK, ctime( &ltime ));
	//PetscPrintf(PETSC_COMM_SELF,"	[%d]  Solve (wall clock) %.2f\n",RANK, dif);
	//PetscFPrintf(PETSC_COMM_SELF, stream, "	[%d]  Solve (wall clock) %.2f\n",RANK, dif);
	t2 = clock();
	tused = (t2-t1)/(double)CLOCKS_PER_SEC;
	//PetscPrintf(PETSC_COMM_SELF,"	[%d]  Tused = %g\n",RANK, tused);
	MPI_Reduce(&tused, &tavg, 1, MPI_DOUBLE,MPI_SUM, 0, PETSC_COMM_WORLD);
	tavg /= (double)SIZE;
	PetscPrintf(PETSC_COMM_WORLD,"	Time spend in solve = %g\n", tavg);
	PetscFPrintf(PETSC_COMM_WORLD, stream, "	Time spend in solve = %g\n", tavg);

	//get u, p from sol
	VecBlockGetSubVector(sol, 0, &u);
	VecBlockGetSubVector(sol, 1, &p);

	//PetscPrintf(PETSC_COMM_WORLD, "--sol-----\n");
	//VecView(u, PETSC_VIEWER_STDOUT_WORLD);
	//VecView(p, PETSC_VIEWER_STDOUT_WORLD);


	//release sol
	//VecDestroy(sol);

	//--------------------------------------
	//Check solution and clean up
	//--------------------------------------
	//Vec r1, r2;
	//double norm1, norm2;

	////Check the error in u
	//VecDuplicate(u, &r1);
	//VecCopy(u, r1); //r1=u
	//VecAYPX(r1, -1.0, ue); //r1=u-ue
	//VecNorm(r1, NORM_2, &norm1);
	//VecDestroy(r1);

	////Check the error in p
	//VecDuplicate(p, &r2);
	//VecCopy(p, r2); //r2=p
	//VecAYPX(r2, -1.0, pe); //r2=p-pe
	//VecNorm(r2, NORM_2, &norm2);
	//VecDestroy(r2);

	////compute NORM_2 for (u, p)
	//norm = sqrt(norm1*norm1 + norm2*norm2);

	//Check the error in the block format
	Vec r;
	VecDuplicate(sol, &r);
	VecCopy(sol, r); //r=sol
	VecAYPX(r, -1.0, exactsol); //r1=u-ue
	VecNorm(r, NORM_2, &norm);
	VecDestroy(r);


	KSPGetIterationNumber(ksp,&its);
	PetscPrintf(PETSC_COMM_WORLD,"	Norm of error %g, Outer Iterations %d\n", norm,its);
	PetscFPrintf(PETSC_COMM_WORLD, stream, "	Norm of error %g, Outer Iterations %d\n", norm,its);

	KSPGetIterationNumber(subksp1,&its);
	PetscPrintf(PETSC_COMM_WORLD,"	KSP1: Inner Iterations %d\n", its);
	PetscFPrintf(PETSC_COMM_WORLD, stream,"	KSP1: Inner Iterations %d\n", its);

	KSPGetIterationNumber(subksp2,&its);
	PetscPrintf(PETSC_COMM_WORLD,"	KSP2: Inner Iterations %d\n\n", its);
	PetscFPrintf(PETSC_COMM_WORLD, stream,"	KSP2: Inner Iterations %d\n\n", its);

	//KSPDestroy(subksp1);
	//KSPDestroy(subksp2);

	// clean up shell
	KSPDestroy(shell.innerksp1);
	KSPDestroy(shell.innerksp2);
	VecDestroy(shell.xhat);
	VecDestroy(shell.yhat);
	VecDestroy(shell.y1);

	MatDestroy(ApM);
	MatDestroy(LQ);

	VecDestroy(shell.diagAM);
	VecDestroy(shell.diagLQ);
	VecDestroy(shell.diagL);

	ierr = KSPDestroy(ksp);

}
#endif


/*
void SYSTEM::Solve(void)
{
	PetscErrorCode ierr;
	Hiptmair  shell;
	PetscScalar norm;
	Vec r;
	PetscInt its;

	KSP ksp; //remember to delete ksp
	PC pc;
	Mat mypc; //remember to delete mypc

	Mat ApM; //ApM = A+(1-k^2)*M
	MatDuplicate(Amat,MAT_COPY_VALUES,&ApM);
	MatAXPY(ApM, 1-paramk*paramk, Mmat, DIFFERENT_NONZERO_PATTERN);

	MatCreate(PETSC_COMM_WORLD, &mypc);
	MatSetSizes(mypc, 2,2, 2,2);
	MatSetType(mypc, "block");

	// Prescribe sub matrices within the block
	MatBlockSetValue(mypc, 0,0, ApM, DIFFERENT_NONZERO_PATTERN, INSERT_VALUES);
	MatBlockSetValue(mypc, 1,1, Lmat, DIFFERENT_NONZERO_PATTERN, INSERT_VALUES);
	Mat_Assemble(mypc);

	ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);
	ierr = KSPSetOperators(ksp,Stiffmat,mypc,SAME_NONZERO_PATTERN);
	ierr = KSPSetType(ksp,KSPMINRES);
	ierr = KSPGetPC(ksp,&pc);
	PCSetType(pc,"block");
	PCBlock_SetBlockType(pc,PC_BLOCK_DIAGONAL);

	KSP subksp1, subksp2;
	PC subpc1, subpc2;

	//------------------
	//construct subksp1
	//------------------
	Mat LQ;
	KSPCreate(PETSC_COMM_WORLD, &subksp1);
	KSPSetOperators(subksp1, ApM, ApM, SAME_NONZERO_PATTERN);
	KSPGetPC(subksp1, &subpc1);
	KSPSetOptionsPrefix(subksp1, "ksp1_");
	PCSetOptionsPrefix(subpc1, "ksp1_");
	KSPSetType(subksp1,"cg");
	//hypre pc
	//PCSetType(subpc1, "hypre");
	//PetscOptionsSetValue("-ksp1_pc_hypre_type","boomeramg");
	//ierr = PCSetFromOptions(subpc1);
	//ierr = KSPSetFromOptions(subksp1);
	//char *pn1;
	//PCHYPREGetType(subpc1, (const char**)&pn1);
	//PetscPrintf(PETSC_COMM_WORLD,"(1,1): %s\n", pn1);
	//shell pc
	MyshellSetup(&shell, LQ);
	PCSetType(subpc1,PCSHELL);
	PCShellSetApply(subpc1,MyShellPCApply);
	PCShellSetContext(subpc1,&shell);

	KSPSetTolerances(subksp1,1.e-6,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);

	//-------------------
	//construct subksp2
	//-------------------
	KSPCreate(PETSC_COMM_WORLD, &subksp2);
	KSPSetOperators(subksp2, Lmat, Lmat, SAME_NONZERO_PATTERN);
	KSPGetPC(subksp2, &subpc2);
	KSPSetOptionsPrefix(subksp2, "ksp2_");
	PCSetOptionsPrefix(subpc2, "ksp2_");
	KSPSetType(subksp2,"cg");
	//PCSetType(subpc2, "lu");
	PCSetType(subpc2, "hypre");
	PetscOptionsSetValue("-ksp2_pc_hypre_type","boomeramg");
	ierr = PCSetFromOptions(subpc2);
	ierr = KSPSetFromOptions(subksp2);
	//char *pn2;
	//PCHYPREGetType(subpc2, (const char**)&pn2);
	//PetscPrintf(PETSC_COMM_WORLD,"(2,2): %s\n", pn2);

	KSPSetTolerances(subksp2,1.e-6,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);



	PCBlockSetSubKSP(pc, 0, subksp1);
	PCBlockSetSubKSP(pc, 1, subksp2);
	ierr = KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);


	//-----------------------------------
	//use subksp2 as a stand alone solver
	//-----------------------------------
	Vec bb, xx, uu;
	MatGetVecs(Lmat,&uu,&bb);
	ierr = VecDuplicate(uu,&xx);
	ierr = VecSet(uu,1.0);
	ierr = MatMult(Lmat,uu,bb);

	int t1, t2;
	t1 = clock();
	ierr = KSPSolve(subksp2,bb,xx);
	t2 = clock();
	PetscPrintf(PETSC_COMM_WORLD,"(2,2):	Time spend in solve = %g\n", (t2-t1)/(double)CLOCKS_PER_SEC);
	PetscFPrintf(PETSC_COMM_WORLD, stream, "(2,2):	Time spend in solve = %g\n", (t2-t1)/(double)CLOCKS_PER_SEC);

	ierr = VecAXPY(xx,-1.0,uu);
	ierr = VecNorm(xx,NORM_2,&norm);
	KSPGetIterationNumber(subksp2,&its);
	PetscPrintf(PETSC_COMM_WORLD,"(2,2): Norm of error %g, Outer Iterations %d\n", norm,its);
	PetscFPrintf(PETSC_COMM_WORLD, stream, "(2,2): Norm of error %g, Outer Iterations %d\n", norm,its);



	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	//Solve the linear system
	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	VecDuplicate(rhs,&sol);
	int starttime, stoptime;
	double timeused;
	starttime = clock();
	ierr = KSPSolve(ksp,rhs,sol);
	stoptime = clock();
	PetscPrintf(PETSC_COMM_WORLD,"	Time spend in solve = %g\n", (stoptime-starttime)/(double)CLOCKS_PER_SEC);
	PetscFPrintf(PETSC_COMM_WORLD, stream, "	Time spend in solve = %g\n", (stoptime-starttime)/(double)CLOCKS_PER_SEC);

	//get u, p from sol
	VecBlockGetSubVector(sol, 0, &u);
	VecBlockGetSubVector(sol, 1, &p);


	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	//Check solution and clean up
	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	//
	//Check the error
	//
	VecDuplicate(sol,&r);
	VecCopy(sol,r);
	VecAYPX(r,-1.0,exactsol);
	VecNorm(r, NORM_2, &norm);
	KSPGetIterationNumber(ksp,&its);
	PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Outer Iterations %d\n", norm,its);
	PetscFPrintf(PETSC_COMM_WORLD, stream, "Norm of error %g, Outer Iterations %d\n", norm,its);

	KSPGetIterationNumber(subksp1,&its);
	PetscPrintf(PETSC_COMM_WORLD,"	KSP1: Inner Iterations %d\n", its);
	PetscFPrintf(PETSC_COMM_WORLD, stream,"	KSP1: Inner Iterations %d\n", its);

	KSPGetIterationNumber(subksp2,&its);
	PetscPrintf(PETSC_COMM_WORLD,"	KSP2: Inner Iterations %d\n", its);
	PetscFPrintf(PETSC_COMM_WORLD, stream,"	KSP2: Inner Iterations %d\n", its);


	//PetscViewerSetFormat( PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL );
	//PetscPrintf(PETSC_COMM_WORLD, "--sol-----\n");
	//VecView(sol, PETSC_VIEWER_STDOUT_WORLD);
	//PetscPrintf(PETSC_COMM_WORLD, "--rhs-----\n");
	//VecView(rhs, PETSC_VIEWER_STDOUT_WORLD);
	//PetscPrintf(PETSC_COMM_WORLD, "--Stiffmat-----\n");
	//MatView(Stiffmat, PETSC_VIEWER_STDOUT_WORLD);

	//KSPDestroy(subksp1);
	//KSPDestroy(subksp2);

	// clean up shell
	KSPDestroy(shell.innerksp1);
	KSPDestroy(shell.innerksp2);
	VecDestroy(shell.xhat);
	VecDestroy(shell.yhat);
	VecDestroy(shell.y1);

	MatDestroy(ApM);
	MatDestroy(LQ);

	ierr = VecDestroy(uu);  ierr = VecDestroy(xx);
	ierr = VecDestroy(bb);

	VecDestroy(r);

	VecDestroy(shell.diagAM);
	VecDestroy(shell.diagLQ);
	VecDestroy(shell.diagL);

	ierr = KSPDestroy(ksp);

}
*/


void SYSTEM::View(void)
{
	PetscViewer viewer;
	PetscErrorCode ierr;

	ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "output_system.m", &viewer);
	ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
	ierr = PetscObjectSetName((PetscObject)Amat, "Amat");
	ierr = PetscObjectSetName((PetscObject)Bmat, "Bmat");
	ierr = PetscObjectSetName((PetscObject)Mmat, "Mmat");
	ierr = PetscObjectSetName((PetscObject)Lmat, "Lmat");
	ierr = PetscObjectSetName((PetscObject)g, "g");
	ierr = MatView(Amat, viewer);
	ierr = MatView(Bmat, viewer);
	ierr = MatView(Mmat, viewer);
	ierr = MatView(Lmat, viewer);
	ierr = VecView(g, viewer);
	PetscViewerDestroy(viewer);

}

void SYSTEM::Destroy(void)
{
	PetscErrorCode ierr;

	ierr = MatDestroy(Stiffmat);
	ierr = MatDestroy(Amat);
	ierr = MatDestroy(Bmat);
	ierr = MatDestroy(Mmat);
	ierr = MatDestroy(Lmat);
	ierr = MatDestroy(Qmat);
	ierr = MatDestroy(Cmat);
	ierr = MatDestroy(Rx);
	ierr = MatDestroy(Ry);
	ierr = MatDestroy(Rz);

	ierr = VecDestroy(sol);
	ierr = VecDestroy(exactsol);
	ierr = VecDestroy(rhs);

	//ierr = VecDestroy(u);
	//ierr = VecDestroy(p);
	//ierr = VecDestroy(ue);
	//ierr = VecDestroy(pe);
	ierr = VecDestroy(f);
	ierr = VecDestroy(g);
}

void SYSTEM::Get_Exact_Sol()
{
	Vec_Create(ue, mesh.edge_count_i[RANK]+mesh.edge_count_b[RANK], mesh.Nedge);
	Vec_Create(pe, mesh.nod_count_i[RANK]+mesh.nod_count_b[RANK], mesh.Nnod);

	int start, end;

	start=0;
	for (int i=0; i<RANK; i++) start+=mesh.edge_count_i[i]+mesh.edge_count_b[i];
	end = start+mesh.edge_count_i[RANK]+mesh.edge_count_b[RANK];

	//ue
	for (int n=start; n<end; n++) {
		PetscScalar spt[3], ept[3];
		for (int i=0; i<3; i++) {
			spt[i] = mesh.coords[mesh.edges[n][0]][i];
			ept[i] = mesh.coords[mesh.edges[n][1]][i];
		}

		// u(n) = tancomp(coords(:,spt), coords(:,ept));
		double t=tancomp(spt, ept);
		VecSetValue(ue, n, tancomp(spt, ept), INSERT_VALUES);
	}


	//pe
	start=0;
	for (int i=0; i<RANK; i++) start+=mesh.nod_count_i[i]+mesh.nod_count_b[i];
	end = start+mesh.nod_count_i[RANK]+mesh.nod_count_b[RANK];
	for (int n=start; n<end; n++) {
		double coord[3], p;
		coord[0] = mesh.coords[n][0];
		coord[1] = mesh.coords[n][1];
		coord[2] = mesh.coords[n][2];
		func_p(coord, &p);
		VecSetValue(pe, n, p, INSERT_VALUES);
	}

	Vec_Assemble(ue);
	Vec_Assemble(pe);

}

PetscScalar SYSTEM::tancomp(PetscScalar *spt, PetscScalar *ept)
{
	PetscScalar xnod[6] = {0.9324695, -0.9324695, 0.6612094, -0.6612094, 0.2386192, -0.2386192};
	PetscScalar w[6] = {0.1713245, 0.1713245, 0.3607616, 0.3607616, 0.4679139, 0.46799139};
	PetscInt lw = 6;

	//PetscScalar xnod[2] = {-sqrt(1.0/3), sqrt(1.0/3)};
	//PetscScalar w[2] = {1,1};
	//PetscInt lw = 2;

	double diff[3], len;
	vec_sum(3, 1, -1, ept, spt, diff);
	len = vec_normal(3, diff);

	PetscScalar sol = 0;

	for (int i=0; i<lw; i++) {
		double t = (xnod[i]+1.0)/2.0;

		//pt=spt + (ept - spt) * t;
		double pt[3];
		vec_sum(3, 1-t, t, spt, ept, pt);

		//func_u(pt)
		double u[3];
		func_u(pt, u);

		//sol = sol + w(i)*b(pt)(dot)(ept-spt)/norm(ept-spt);
		sol += w[i]*mydot(3, u, diff)/len;
	}

	return sol*0.5*len;
}

void SYSTEM::func_f(PetscScalar *coord, PetscScalar *f)
{
	PetscScalar x = coord[0];
	PetscScalar y = coord[1];
	PetscScalar z = coord[2];
	double curlcurlu[3];
	curlcurlu[0] =2*(2-y*y-z*z);
	curlcurlu[1] =2*(2-x*x-z*z);
	curlcurlu[2] =2*(2-x*x-y*y);

	double u[3];
	func_u(coord, u);

	double gradp[3];
	gradp[0] = -2*x*(1-y*y)*(1-z*z);
	gradp[1] = -2*y*(1-x*x)*(1-z*z);
	gradp[2] = -2*z*(1-x*x)*(1-y*y);

	double muinv = 1.0/func_mu(coord);
	double eps = func_eps(coord);

	// sol = 1/mu * curlcurlu - eps*k^2 * u + eps * gradp;
	vec_sum(3, muinv, -eps*paramk*paramk, curlcurlu, u, f);
	vec_sum(3, 1, eps, f, gradp, f);

}

void SYSTEM::func_u(PetscScalar *coord, PetscScalar *u)
{
	PetscScalar x,y,z;
	x = coord[0];
	y = coord[1];
	z = coord[2];
	u[0] =(1-y*y)*(1-z*z);
	u[1] = (1-x*x)*(1-z*z);
	u[2] = (1-x*x)*(1-y*y);
}

void SYSTEM::func_p(PetscScalar *coord, PetscScalar *p)
{
	PetscScalar x,y,z;
	x = coord[0];
	y = coord[1];
	z = coord[2];
	p[0] = (1-x*x)*(1-y*y)*(1-z*z);
}

double SYSTEM::func_mu(double *coord)
{
	//double mu0 = 4*3.1415926*1e-7;
	double mu0 = 1.0;

	if (variable<0) return 1/mu0;

	double x, y, z;
	x = coord[0];
	y = coord[1];
	z = coord[2];
	//if (z<0) {
	//	if (y<0) {
	//		if (x<0) return 1/mu0*variable;
	//		else return 2/mu0*variable; //2;
	//	}
	//	else {
	//		if (x<0) return 3/mu0*variable; //3;
	//		else return 4/mu0*variable; //4;
	//	}
	//}
	//else {
	//	if (y<0) {
	//		if (x<0) return 5/mu0*variable; //5;
	//		else return 6/mu0*variable; //6;
	//	}
	//	else {
	//		if (x<0) return 7/mu0*variable; //7;
	//		else return 8/mu0*variable; //8;
	//	}
	//}


	//// special code for m747
	//if (z<0.5) {
	//	if (x<0.5) return 1/mu0*variable;
	//	else return 2/mu0*variable; //2;
	//}
	//else {
	//	if (x<0.5) return 3/mu0*variable; //3;
	//	else return 4/mu0*variable; //4;
	//}

	// special code for m747
	return 1;

	////special code for cube with 2 subdomains
	//return 1;
	//if (x<0.0)
	//	return 1;
	//else
	//	return variable;

}

double SYSTEM::func_eps(double *coord)
{
	//double e0 = 1/(36*3.1415926)*1e-9;
	double e0 = 1.0;

	if (variable<0) return 1/e0;

	double x, y, z;
	x = coord[0];
	y = coord[1];
	z = coord[2];
	//if (z<0) {
	//	if (y<0) {
	//		if (x<0) return 1/e0*variable;
	//		else return 2/e0*variable; //2;
	//	}
	//	else {
	//		if (x<0) return 3/e0*variable; //3;
	//		else return 4/e0*variable; //4;
	//	}
	//}
	//else {
	//	if (y<0) {
	//		if (x<0) return 5/e0*variable; //5;
	//		else return 6/e0*variable; //6;
	//	}
	//	else {
	//		if (x<0) return 7/e0*variable; //7;
	//		else return 8/e0*variable; //8;
	//	}
	//}

	//// special code for m747
	//if (z<0.5) {
	//	if (x<0.5) return 1/e0*variable;
	//	else return 2/e0*variable; //2;
	//}
	//else {
	//	if (x<0.5) return 3/e0*variable; //3;
	//	else return 4/e0*variable; //4;
	//}

	// special code for m747
	if (z<0.5) {
		return 1; //2;
	}
	else {
		return variable; //4;
	}
	////special code for cube with 2 subdomains
	//if (x<0)
	//	return 1;
	//else
	//	return variable;

}



#ifdef _WIN32
void SYSTEM::MyshellSetup(Hiptmair *shell, Mat& LQ)
{
}

#else

void SYSTEM::MyshellSetup(Hiptmair *shell, Mat& LQ)
{

	shell->C = Cmat;
	shell->L = Lmat;
	shell->Q = Qmat;
	shell->Rx = Rx;
	shell->Ry = Ry;
	shell->Rz = Rz;
	shell->k = paramk;

	// Setup diagAM
	Vec diagA, diagM;
	MatGetVecs(Amat,&diagA, &diagM);
	MatGetDiagonal(Amat,diagA);
	MatGetDiagonal(Mmat,diagM);
	VecAXPY(diagA, (1-paramk*paramk), diagM);
	VecReciprocal(diagA);
	VecDestroy(diagM);

	shell->diagAM = diagA;

	// setup diagLQ and diagL
	Vec diagL, diagQ;
	MatGetVecs(Lmat,&diagL, &diagQ);
	MatGetDiagonal(Lmat,diagL);
	MatGetDiagonal(Qmat,diagQ);
	VecAXPY(diagQ, (1-paramk*paramk), diagL); //diagQ = diagL+(1-k^2)*diagQ
	VecReciprocal(diagQ);
	VecReciprocal(diagL);

	shell->diagAM = diagA; //diagA, diagQ, diagL should not be deleted
	shell->diagLQ = diagQ;
	shell->diagL = diagL;

	//---------------------
	//setup inner ksp
	//---------------------

	//------------------
	//construct subksp1
	//------------------
	KSP innerksp1;
	PC innerpc1;
	KSPCreate(PETSC_COMM_WORLD, &innerksp1);
	KSPSetOperators(innerksp1, Lmat, Lmat, SAME_NONZERO_PATTERN);
	KSPGetPC(innerksp1, &innerpc1);
	KSPSetOptionsPrefix(innerksp1, "innerksp1_");
	PCSetOptionsPrefix(innerpc1, "innerpc1_");
	KSPSetType(innerksp1,"preonly");
	//hypre pc
	PCSetType(innerpc1, "hypre");
	PetscOptionsSetValue("-innerpc1_pc_hypre_type","boomeramg");
	PetscOptionsSetValue("-innerpc1_pc_hypre_boomeramg_strong_threshold", "0.5");
	//PetscOptionsSetValue("-innerpc1_pc_hypre_boomeramg_max_iter","2");
	PCSetFromOptions(innerpc1);
	KSPSetFromOptions(innerksp1);
	char *pn1;
	PCHYPREGetType(innerpc1, (const char**)&pn1);
	PetscPrintf(PETSC_COMM_WORLD,"		innerksp1: %s\n", pn1);
	//KSPSetTolerances(subksp1,1.e-6,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
	shell->innerksp1 = innerksp1;

	//------------------
	//construct subksp2
	//------------------
	KSP innerksp2;
	PC innerpc2;
	MatDuplicate(Lmat,MAT_COPY_VALUES,&LQ);
	MatAXPY(LQ, (1-paramk*paramk), Qmat, SAME_NONZERO_PATTERN);
	KSPCreate(PETSC_COMM_WORLD, &innerksp2);
	KSPSetOperators(innerksp2, LQ, LQ, SAME_NONZERO_PATTERN);
	KSPGetPC(innerksp2, &innerpc2);
	KSPSetOptionsPrefix(innerksp2, "innerksp2_");
	PCSetOptionsPrefix(innerpc2, "innerpc2_");
	KSPSetType(innerksp2,"preonly");
	//hypre pc
	PCSetType(innerpc2, "hypre");
	PetscOptionsSetValue("-innerpc2_pc_hypre_type","boomeramg");
	PetscOptionsSetValue("-innerpc2_pc_hypre_boomeramg_strong_threshold", "0.5");
	//PetscOptionsSetValue("-innerpc2_pc_hypre_boomeramg_max_iter","2");
	PCSetFromOptions(innerpc2);
	KSPSetFromOptions(innerksp2);
	char *pn2;
	PCHYPREGetType(innerpc2, (const char**)&pn2);
	PetscPrintf(PETSC_COMM_WORLD,"		innerksp2: %s\n", pn2);
	shell->innerksp2 = innerksp2;

	//set up the workspace
	MatGetVecs(Lmat, &(shell->xhat), &(shell->yhat));
	MatGetVecs(Amat, &(shell->y1), PETSC_NULL);

	//PetscViewerSetFormat( PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL );
	//PCView(innerpc1, PETSC_VIEWER_STDOUT_WORLD );

}

#endif


void SYSTEM::array2CSR(AIJ* a, int size, int nrows, int* is, int* js, double* vs)
{
	double SMALL = 1e-10;
	int pos = 0;  //pos is the postion to write the current nnz
	int posi = 0; //posi is the current row to process


	for (int i = 0; i<size; i++) {
		//ignore the small entries
		if (fabs(a[i].value)<SMALL) continue;

		if (a[i].i >= posi) {//find a new non zero entry on a different row
			for (int j=posi; j<=a[i].i; j++) is[j] = pos;
			posi = a[i].i+1;
		}
		js[pos] = a[i].j;
		vs[pos] = a[i].value;
		pos++;
	}

	for (int i=posi; i<=nrows; i++)
		is[i] = pos;
}

