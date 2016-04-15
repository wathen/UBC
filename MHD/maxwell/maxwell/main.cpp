#include <math.h>
#include <time.h>
//#include <cstdlib>

#include "MESH.h"
#include "SYSTEM.h"
#include "PetscCalls.h"

#include "petscext.h"
#include "petscext_vec.h"
#include "petscext_mat.h"
#include "petscext_pc.h"
#include "petscext_utils.h"

PetscInt SIZE, RANK;	

static char help[] = "Solves Maxwell's Equations in parallel. \n";
int main(int argc, char **args)
{


	SYSTEM sys;
	char fn[100]; //name of output file
	
	time_t ltime;
	time_t starttime, stoptime;

	double k=0.0, variable = -1.0;
	PetscTruth genpartition = PETSC_FALSE; //note PETSC_TURE: 1, FALSE 0
	char input[PETSC_MAX_PATH_LEN];  
	char tetgenOptions[PETSC_MAX_PATH_LEN];  //switches for tetgen
	char tetgenInput[PETSC_MAX_PATH_LEN];	//input file for tetgen

	int N; // = 0;
	PetscTruth flgK, flgF, flgN, flgTo, flgTi, flgvari, flggen; //note PETSC_TURE: 1, FALSE 0

	PetscViewerFormat format= PETSC_VIEWER_ASCII_MATLAB; 
	//PETSC_VIEWER_ASCII_DEFAULT  - default format  
	//PETSC_VIEWER_ASCII_MATLAB  - Matlab format  
	//PETSC_VIEWER_ASCII_IMPL  - implementation-specific format (which is in many cases the same as the default)  
	//PETSC_VIEWER_ASCII_INFO  - basic information about object  
	//PETSC_VIEWER_ASCII_INFO_DETAIL  - more detailed info about object  
	//PETSC_VIEWER_ASCII_COMMON  - identical output format for all objects of a particular type  
	//PETSC_VIEWER_ASCII_INDEX  - (for vectors) prints the vector element number next to each vector entry  


	Petsc_Init(argc, args, help);
	PetscExtInitialize();

	//input[0] = '\0';
	//tetgenOptions[0] = '\0';
	//tetgenInput[0] = '\0';

	PetscOptionsGetScalar(PETSC_NULL,"-k",&k,&flgK);
	PetscOptionsGetString(PETSC_NULL,"-f",input,PETSC_MAX_PATH_LEN-1,&flgF);
	PetscOptionsGetInt(PETSC_NULL, "-N", &N, &flgN);
	PetscOptionsGetString(PETSC_NULL,"-to",tetgenOptions,PETSC_MAX_PATH_LEN-1,&flgTo);
	PetscOptionsGetString(PETSC_NULL,"-ti",tetgenInput,PETSC_MAX_PATH_LEN-1,&flgTi);
	PetscOptionsGetScalar(PETSC_NULL, "-var", &variable, &flgvari);
	PetscOptionsGetTruth(PETSC_NULL, "-g", &genpartition, &flggen);


	sys.paramk = k;
	sys.variable = variable;
	sys.mesh.genpartition = genpartition;

	if (flgF == PETSC_TRUE)
		if (flgvari == PETSC_TRUE)
			sprintf(fn, "time_%s_k%.2f_np%d_var%.2f.txt", input, k, SIZE, variable);
		else
			sprintf(fn, "time_%s_k%.2f_np%d.txt", input, k, SIZE);
	else if (flgTo == PETSC_TRUE)
		sprintf(fn, "time_%s_k%.2f_np%d.txt", tetgenInput, k, SIZE);
	else
		if (flgvari == PETSC_TRUE)
			sprintf(fn, "time_mycube_N%d_k%.2f_np%d_var%.2f.txt", N, k, SIZE, variable);
		else		
			sprintf(fn, "time_mycube_N%d_k%.2f_np%d.txt", N, k, SIZE);

	sys.stream = fopen( fn, "a" );

	PetscPrintf(PETSC_COMM_WORLD, "---------------------------------------------------\n");
	PetscPrintf(PETSC_COMM_WORLD, "       file = %s, NP = %d, variable = %g, k = %g\n", input, SIZE, variable, k);
	PetscFPrintf(PETSC_COMM_WORLD,sys.stream, "---------------------------------------------------\n");
	PetscFPrintf(PETSC_COMM_WORLD,sys.stream, "       f = %s, NP = %d, variable = %g, k = %g\n", input, SIZE, variable, k);

	time( &ltime );
	PetscPrintf(PETSC_COMM_WORLD,"[%d]  starts at %s\n",RANK, ctime( &ltime ));
	PetscFPrintf(PETSC_COMM_WORLD, sys.stream, "[%d]  starts at %s\n",RANK, ctime( &ltime ));

	//create mesh
	starttime = clock();
	if (input[0] != '\0') 
		sys.mesh.Create(input);
	else
		sys.mesh.Create(N);
	stoptime = clock();
	time( &ltime );
	PetscPrintf(PETSC_COMM_WORLD,"[%d]  mesh created (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));
	PetscFPrintf(PETSC_COMM_WORLD, sys.stream, "[%d]  mesh created (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));

	// get exact solution
	starttime = clock();
	sys.Get_Exact_Sol();
	stoptime = clock();
	time( &ltime );
	PetscPrintf(PETSC_COMM_WORLD,"[%d]  exactsol created (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));
	PetscFPrintf(PETSC_COMM_WORLD, sys.stream, "[%d]  exactsol created (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));


	// Assemble matrices
	starttime = clock();
	sys.Assemble();
	stoptime = clock();
	time( &ltime );
	PetscPrintf(PETSC_COMM_WORLD,"[%d]  system assembled (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));
	PetscFPrintf(PETSC_COMM_WORLD, sys.stream, "[%d]  system assembled (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));

	// apply BC
	starttime = clock();
	sys.Apply_BC();
	stoptime = clock();
	time( &ltime );
	PetscPrintf(PETSC_COMM_WORLD,"[%d]  system BC applied (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));
	PetscFPrintf(PETSC_COMM_WORLD, sys.stream, "[%d]  system BC applied (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));

	// Destroy mesh to save space
	starttime = clock();
	sys.mesh.Destroy();
	stoptime = clock();
	time( &ltime );
	PetscPrintf(PETSC_COMM_WORLD,"[%d]  mesh destroyed (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));
	PetscFPrintf(PETSC_COMM_WORLD, sys.stream, "[%d]  mesh destroyed (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));

	// form the block system
	starttime = clock();
	sys.Form_System();
	stoptime = clock();
	time( &ltime );
	PetscPrintf(PETSC_COMM_WORLD,"[%d]  System formed (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));
	PetscFPrintf(PETSC_COMM_WORLD, sys.stream, "[%d]  System formed (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));

	//sys.View();

	// solve the linear system
	starttime = clock();
	sys.Solve();
	stoptime = clock();
	time( &ltime );
	PetscPrintf(PETSC_COMM_WORLD,"[%d]  system solved (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));
	PetscFPrintf(PETSC_COMM_WORLD, sys.stream, "[%d]  system solved (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));


	// clean up 
	starttime = clock();
	sys.Destroy();
	stoptime = clock();
	time( &ltime );
	PetscPrintf(PETSC_COMM_WORLD,"[%d]  system destroyed (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));
	PetscFPrintf(PETSC_COMM_WORLD, sys.stream, "[%d]  system destroyed (%g sec) %s\n",RANK, (stoptime-starttime)/(double)CLOCKS_PER_SEC, ctime( &ltime ));

	fclose(sys.stream);

	PetscExtFinalize();
	Petsc_End();

	return 0;
}
