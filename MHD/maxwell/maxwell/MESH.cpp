#include <math.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <vector>
#include <iostream>
//#include <bitset> //for get sparsity
#include <set> //for get sparsity
#include <map> //for edgeMap, newEdgeMap


#include "MESH.h"
#include "PetscCalls.h"
//#include "tetgen.h"
#ifndef _WIN32
extern "C" {
#include "metis.h"
}
#endif

using namespace std;

extern PetscInt SIZE, RANK;

MESH::MESH(void)
{
	filename[0] = '\0';
}

MESH::~MESH(void)
{
}

void MESH::Create_coords()
{
	//system("cd Y:\\code\\cpp\\maxwell\\maxwell");
	//system("y:");
	//system("chdir");
	//system("path");


	char name[100];
    strcpy(name, filename);
    strcat(name, ".node");	

	ifstream infile;
	infile.open (name, ifstream::in);

	int Ndim, Nattr, BdryMarker=0, attr, count;

	Nnod_b=0;
	Nnod_i=0;

	infile>>Nnod;
	infile>>Ndim;
	infile>>Nattr;
	infile>>BdryMarker;

	//allocate space for coords
	coords = new double* [Nnod];
	for (int i=0; i<Nnod; i++) coords[i] = new double [Ndim+1];

	double xmin, xmax, ymin, ymax, zmin, zmax;
		
	//read coords from file
	for (int i=0; i<Nnod; i++) {
		double f;
		infile>>count;
		//read coords
		for (int j=0; j<Ndim; j++) {
			infile>>coords[i][j];
		}
		if (i==0) {
			xmin = coords[i][0];
			xmax = coords[i][0];
			ymin = coords[i][1];
			ymax = coords[i][1];
			zmin = coords[i][2];
			zmax = coords[i][2];
		}
		else {
			if (coords[i][0]<xmin) xmin=coords[i][0];
			if (coords[i][0]>xmax) xmax=coords[i][0];
			if (coords[i][1]<ymin) ymin=coords[i][1];
			if (coords[i][1]>ymax) ymax=coords[i][1];
			if (coords[i][2]<zmin) zmin=coords[i][2];
			if (coords[i][2]>zmax) zmax=coords[i][2];
		}

		//read attributes
		for (int j=0; j<Nattr; j++) {
			infile>>attr;
		}
		//read boundary markers. 
		//  Note here, don't assign boundary node according to the marker. 
		//   Sometimes the marker could be wrong (Eg. -43534535).
		//   read the .face file to determine boundary nodes and boundary edges
		if (BdryMarker) {
			int marker;
			infile>>marker;
		}		
	}

	infile.close();

	//print bounding box
	PetscPrintf(PETSC_COMM_WORLD, "       bounding box = [%g, %g] [%g, %g] [%g, %g]\n", xmin, xmax, ymin, ymax, zmin, zmax);


	//find boundary nodes from .face file
    strcpy(name, filename);
    strcat(name, ".face");	

	infile.open (name, ifstream::in);

	// all faces MUST have boundary marker!
	// Otherwise, I can't identify which face is on the boundary, 
	// because tetgen also list faces on the internal boundary
	int NfaceTemp, *facesTemp;
	infile>>NfaceTemp; //read # of faces including those on the internal boundary
	infile>>BdryMarker;

	if (!BdryMarker) {
		cout<<"Error!!! Faces must have a boundary marker!"<<endl;
		exit(1);
	}

	//alocate space for faces
	faces = new int* [NfaceTemp];

	//read coords from file
	Nface = 0;
	for (int i=0; i<NfaceTemp; i++) {
		infile>>count; //ignore the line counter
		//read boundary nodes
		int *indices = new int[4];	//the first 3 are for vertex indices, the 4th is for bdry marker
		infile>>indices[0];
		infile>>indices[1];
		infile>>indices[2];
		infile>>indices[3];
		//if (indices[3]>0) {	//find a boundary face
		//	coords[--indices[0]][3] = indices[3];	//set bdry markder for nodes on the bounday
		//	coords[--indices[1]][3] = indices[3];	//Note -- is used, because index is 1-based, but we want 0-based.			
		//	coords[--indices[2]][3] = indices[3];		
		//	faces[Nface++] = indices;
		//}

		//special code for m747.mesh. Because the input file has a boundary marker 0.
		//for my own .poly files, non zero boundary marker is required, because that tetgen mark interior boundary as 0
		// and exterior boundary as 1
		// However, for some of the mesh files form inria, they have 0 boundary marker
		if (indices[3]>=0) {	//find a boundary face
			coords[--indices[0]][3] = indices[3]+1;	//set bdry markder for nodes on the bounday
			coords[--indices[1]][3] = indices[3]+1;	//Note -- is used, because index is 1-based, but we want 0-based.			
			coords[--indices[2]][3] = indices[3]+1;		
			faces[Nface++] = indices;
		}
	}

	infile.close();

	//count # of boundary nodes
	Nnod_b = 0;
	for (int i=0; i<Nnod; i++)
		if (coords[i][Ndim]>0.5) //find a boundary node
			Nnod_b++;
	Nnod_i = Nnod - Nnod_b;


}


void MESH::Create_elems()
{
	char name[100];
    strcpy(name, filename);
    strcat(name, ".ele");	

	ifstream infile;
	infile.open (name, ifstream::in);

	int Ndim, region, attr, count;

	infile>>Nel;
	infile>>Ndim;
	infile>>region;

	//allocate space for elems
	elems = new int* [Nel];
	for (int i=0; i<Nel; i++) {
		elems[i] =  new int [Ndim+1];
		elems[i][Ndim] = 0;	//default attribute
	}

	for (int i=0; i<Nel; i++) {
		infile>>count;
		for (int j=0; j<Ndim; j++) {
			infile>>count;
			elems[i][j] = count-1;
		}
		if (region) {
			infile>>attr;
			elems[i][Ndim] = attr;
		}
	}

	infile.close();

}

void MESH::Create_edges()
{

	//-------------------------------------------------------------------------------------------------------------------
	// Note if "ee" is used with tetgen, it is supposed to generate a list of all edges,
	//   However, sometimes the result is wrong.
	//   If "ee" option can be trusted, the following code can be used. Otherwise, generate edges from elems and faces
	// Note only tetgen 1.4.2 gives a list of all edges, 1.4.3 gives a list of edges as segments.
	//-------------------------------------------------------------------------------------------------------------------
	//char name[100];
	//strcpy(name, filename);
	//strcat(name, ".edge");	

	//ifstream infile;
	//infile.open (name, ifstream::in);

	//int Ndim, attr, count;

	//infile>>Nedge;
	//infile>>attr;

	////allocate space edges
	//edges = new int* [Nedge];
	//for (int i=0; i<Nedge; i++)
	//	edges[i] = new int[3];

	//Nedge_i = 0;
	//Nedge_b = 0;

	//for (int i=0; i<Nedge; i++) {
	//	infile>>count;
	//	infile>>count;
	//	edges[i][0]=count-1;
	//	infile>>count;
	//	edges[i][1]=count-1;
	//	if (attr>0) {
	//		infile>>edges[i][2];
	//		if (edges[i][2]>0) Nedge_b++;
	//		else Nedge_i++;
	//	}
	//}

	//infile.close();
	//-------------------------------------------------------------------------------------------------------------------

	
	//-------------------------------------------------------------------------------------------------
	//generate edges from elems and faces, remember to release faces here, if it is not used any more
	//---------------------------------------
	int edgeK[2][6] = {{0, 0, 0, 1, 1, 2},		//{1,1,1,2,2,3}
					   {1, 2, 3, 2, 3, 3}};		//{2,3,4,3,4,4}


	vector<map<int, int> > EdgeMap(Nnod); //EdgeMap[sep][ept]
	Nedge = 0;
	for (int i=0; i<Nel; i++) {
		for (int j=0; j<6; j++) {
			int spt = elems[i][edgeK[0][j]];
			int ept = elems[i][edgeK[1][j]];
			if (ept<spt) swap(spt, ept);
			
			if (EdgeMap[spt].find(ept) == EdgeMap[spt].end()) { //find a new edge 
				//allocate space for the new edge
				EdgeMap[spt][ept] = 0;
				Nedge++;
			}
		} //end of loop j
	} //end of loop i

	//identify boundary edges
	int faceK[2][3] = {{0, 0, 1}, {1, 2, 2}};
	for (int i=0; i<Nface; i++) {
		int marker = faces[i][3];	//just in case faces doesn't have bdry marker (default bdry marker is 0), all edges should still have a non-zero marker
		if (!marker) marker = 1;
		for (int j=0; j<3; j++) {
			int spt = faces[i][faceK[0][j]];
			int ept = faces[i][faceK[1][j]];
			if (ept<spt) swap(spt, ept);
			EdgeMap[spt][ept] = marker;
		}
	}



	//allocate space edges
	edges = new int* [Nedge];
	for (int i=0; i<Nedge; i++) {
		edges[i] = new int[3];
		edges[i][2] = 0;	//default bdry marker
	}

	// construct edges from EdgeMap
	map<int,int>::iterator it;
	Nedge_b = 0;
	int cc=0;
	for (int i=0; i<Nnod; i++) {
		for (it=EdgeMap[i].begin(); it !=EdgeMap[i].end(); it++) {
			edges[cc][0] = i;
			edges[cc][1] = it->first;
			edges[cc][2] = it->second;
			cc++;
			if (it->second>0) Nedge_b++;
		}
	}

	Nedge_i = Nedge - Nedge_b;

	//release memory for faces
	for (int i=0; i<Nface; i++) {
		delete[] faces[i];
	}
	delete[] faces;


}

void MESH::Create_neighs()
{
	char name[100];
    strcpy(name, filename);
    strcat(name, ".neigh");	

	ifstream infile;
	infile.open (name, ifstream::in);

	int Ndim, count;

	infile>>count>>Ndim;

	//allocate space
	neighs = new int* [Nel];
	for (int i=0; i<Nel; i++) neighs[i] = new int [Ndim];

	for (int i=0; i<Nel; i++) {
		infile>>count;
		for (int j=0; j<Ndim; j++) {
			infile>>count;
			neighs[i][j]=count-1;
		}
	}

	infile.close();

	////get ia, ja
	//int *ia, *ja;
	//int net = 0;
	//ia = new int[Nnod];
	//ja = new int[Nnod*6];
	//ia[0] = 0;
	//for (int i=0; i<Nel; i++) {
	//	for (int j=0; j<Ndim; j++) {
	//		if (neighs[i][j]>=0) ja[net++] = neighs[i][j];
	//		ia[i+1] = net;
	//	}
	//}

}

void MESH::Create_partition(void)
{
	int sizesize = SIZE;
	//sizesize = 2;

	int edgeK[2][6] = {{0, 0, 0, 1, 1, 2},		//{1,1,1,2,2,3}
					   {1, 2, 3, 2, 3, 3}};		//{2,3,4,3,4,4}


	//compute partition on the 0-th proc
	int *epart = new int[Nel];

	for (int i=0; i<Nel; i++) epart[i] = 0;
	
#ifdef _WIN32
	//--------------------------------------------
	// create partition for windows version
	//--------------------------------------------
	//partition the mesh if SIZE>1
	if (sizesize>1) {
		char name[100];

		if (!RANK) {
			
			//out put mesh
			ofstream outfile;
			strcpy(name, filename);
			strcat(name, ".mesh");	
			outfile.open (name, ofstream::out);
			outfile<<Nel<<" "<<2<<endl;
			for (int i=0; i<Nel; i++){
				for (int j=0; j<4; j++)
					outfile<<elems[i][j]+1<<" ";
				outfile<<endl;
			}
			outfile.close();

			char cmd[100];
			sprintf(cmd, "partnmesh %s %d\n", name, sizesize);
			system(cmd);

		}

		//broadcast to the other procs, slow on large data!!!
		//MPI_Bcast(epart,Nel,MPI_INT,0,PETSC_COMM_WORLD);

		//sycronize processes
		MPI_Barrier(PETSC_COMM_WORLD);

		sprintf(name, "%s.mesh.epart.%d", filename, sizesize);
		ifstream infile;
		infile.open (name, ifstream::in);
		for (int i=0; i<Nel; i++) {
			infile>>epart[i];
		}
		infile.close();
	}
	//--------------------------------------------
#else
	//--------------------------------------------
	// create partition for linux version
	//--------------------------------------------
	if (sizesize>1) {
		//METIS_PartMeshNodal (int *ne, int *nn, idxtype *elmnts, int *etype, int *numflag, int *nparts, int *edgecut, idxtype *epart, idxtype *npart);

		int* elmnts = new int[Nel*4];
		int etype = 2;		//1: triangles, 2: tetrahedra, 3 bricks, 4 quads
		int numflag = 0;	//0: c-style, 1: fortran_style
		int edgecut;
		int* npart = new int[Nnod];

		for (int i=0; i<Nel; i++)
			for (int j=0; j<4; j++)
				elmnts[i*4+j] = elems[i][j];

		METIS_PartMeshNodal(&Nel, &Nnod, elmnts, &etype, &numflag, &sizesize, &edgecut, epart, npart);

		delete[] npart;		//npart is not used, release the meomory
		delete[] elmnts;		
	}
#endif

	//generate output for partitioner mesh viewer
	if (genpartition && !RANK) Create_pmvoutput(epart);



	//------------------------------------------------------
	// reorder elems, coords, edges according to partition
	//------------------------------------------------------

	//----------------------------------------
	// get partition result back for elements
	//----------------------------------------
	int *elem_new_indx = new int[Nel];
	int *elem_sum = new int[sizesize]; //elem_sum[i] is the # of elements on the 0 to (i-1)-th partitions
	elem_count = new int[sizesize]; //elem_count[i] is # of elems on the i-th partition

	for (int i=0; i<sizesize; i++) elem_count[i]=0;

	for (int i=0; i<Nel; i++) {
		elem_count[epart[i]]++;
	}

	elem_sum[0]=0;
	for (int i=1; i<sizesize; i++)
		elem_sum[i] = elem_sum[i-1]+elem_count[i-1];

	for (int i=0; i<Nel; i++) elem_new_indx[i] = elem_sum[epart[i]]++;

	//keep the original ordering, not really useful, for debugging
	if (sizesize==1) for (int i=0; i<Nel; i++) elem_new_indx[i] = i;

	// release memory for epart
	delete[] epart;



	//----------------------------------------
	//reorder elems according to new partition
	//----------------------------------------
	//allocate space te,for tmp elems
	int **te;
	te = new int* [Nel];
	for (int i=0; i<Nel; i++) te[elem_new_indx[i]] = elems[i];
	for (int i=0; i<Nel; i++) elems[i] = te[i];
	delete[] te;

	////----------------------------------------
	//// get partition result back for nodes
	////----------------------------------------
	//sprintf(name, "%s.mesh.npart.%d", filename, sizesize);
	//int *npart = new int[Nnod];
	//int *nod_new_indx = new int[Nnod];
	//int *nod_sum = new int[sizesize]; //nod_sum[i] is the # of nodes on the 0 to (i-1)-th partitions
	//nod_count = new int[sizesize]; //nod_count[i] is # of nodes on the i-th partition

	//for (int i=0; i<sizesize; i++) nod_count[i]=0;

	//infile.open (name, ifstream::in);
	//for (int i=0; i<Nnod; i++) {
	//	infile>>npart[i];
	//	nod_count[npart[i]]++;
	//}
	//infile.close();

	//nod_sum[0]=0;
	//for (int i=1; i<sizesize; i++)
	//	nod_sum[i] = nod_sum[i-1]+nod_count[i-1];

	//for (int i=0; i<Nnod; i++) nod_new_indx[i] = nod_sum[npart[i]]++;


	//----------------------------------------
	// partition nodes according to element partition
	//----------------------------------------
	vector<bool> flag(Nnod, false);
	vector<int> nod_new_indx(Nnod, -1);
	vector<int> elem_nod_count_i(Nel, 0); //# non-repeading nodes per element
	vector<int> elem_nod_count_b(Nel, 0); //# non-repeading nodes per element
	int count=0;
	for (int i=0; i<Nel; i++) {
		for (int j=0; j<4; j++) {
			if (!flag[elems[i][j]]) { //the current node is not counted
				nod_new_indx[elems[i][j]] = count++;
				flag[elems[i][j]]=true;
				if (coords[elems[i][j]][3]>0.1) //use 0.1 instead of 1 to avoid fl pt error (coords[][3] is a double)
					elem_nod_count_b[i]++;
				else
					elem_nod_count_i[i]++;

			}
		}
	}
	// compute nod_count_i and node_count_b based on elemental results
	nod_count_i = new int[sizesize];
	nod_count_b = new int[sizesize];
	int t=0;
	for (int i=0; i<sizesize; i++) {
		nod_count_i[i]=0;
		nod_count_b[i]=0;
		for (int j=t; j<t+elem_count[i]; j++) {
			nod_count_i[i] += elem_nod_count_i[j];
			nod_count_b[i] += elem_nod_count_b[j];
		}
		t+=elem_count[i];
	}

	//keep the original ordering, not really useful, for debugging
	if (sizesize==1) for (int i=0; i<Nnod; i++) nod_new_indx[i] = i;


	//----------------------------------------
	//reorder nodes according to new partition
	//----------------------------------------
	//allocate space tn,for tmp coords
	double **tc;
	tc = new double* [Nnod];
	for (int i=0; i<Nnod; i++) tc[nod_new_indx[i]] = coords[i];
	for (int i=0; i<Nnod; i++) coords[i] = tc[i];
	delete[] tc;

	// modify indx_nod_i, indx_nod_b;
	indx_nod_i = new int [Nnod_i]; 
	indx_nod_b = new int [Nnod_b]; 

	Nnod_i = 0;
	Nnod_b = 0;
	for (int i=0; i<Nnod; i++) {
		if (int(coords[i][3])>0) 
			indx_nod_b[Nnod_b++]=i;
		else 
			indx_nod_i[Nnod_i++]=i;
	}

	//----------------------------------------
	// modify elems according to new node ordering
	//----------------------------------------
	for (int i=0; i<Nel; i++) 
		for (int j=0; j<4; j++)
			elems[i][j] = nod_new_indx[elems[i][j]];



	//----------------------------------------
	// modify edges according to new node ordering
	//----------------------------------------
	//vector<int> edgeMap(Nnod*Nnod, -1); //contains the original indices of edges 
	vector<map<int, int> > edgeMap(Nnod);
	for (int i=0; i<Nedge; i++) {
		int x, y;
		x = nod_new_indx[edges[i][0]];
		y = nod_new_indx[edges[i][1]];
		if (x>y) swap(x, y);
		edges[i][0] = x;
		edges[i][1] = y;
		//edgeMap[x*Nnod+y] = i;
		edgeMap[x][y] = i;
	}

	edges_elem = new int* [Nel];

	//allocate space for indx_edge_i, indx_edge_b
	indx_edge_i = new int[Nedge_i];
	indx_edge_b = new int[Nedge_b];


	//allocate space fo edge_count_i, edge_count_b
	edge_count_i = new int[sizesize];
	edge_count_b = new int[sizesize];


	//vector<int> newEdgeMap(Nnod*Nnod, -1); //contains the new indices of edges according to partition
	vector<map<int, int> > newEdgeMap(Nnod);
	int c1=-1, c2=-1, c3=-1; //c1 counts the number of all edges, c2 counts the number of interior edges, c3 counts for # of bdry edges
	t = 0;
	for (int s=0; s<sizesize; s++) {
		edge_count_i[s]=0;
		edge_count_b[s]=0;
		for (int i=t; i<t+elem_count[s]; i++) {
			edges_elem[i] = new int[6];
			for (int j=0; j<6; j++) {
				int spt = elems[i][edgeK[0][j]];
				int ept = elems[i][edgeK[1][j]];
				if (ept<spt) swap(spt, ept);
				
				if (newEdgeMap[spt].find(ept) == newEdgeMap[spt].end()) { //find a new edge 
					//allocate space for the new edge
					newEdgeMap[spt][ept] = ++c1;

					if (edges[edgeMap[spt][ept]][2]>0) { //check if it is a bdry edge
						indx_edge_b[++c3] = c1;
						edge_count_b[s]++;
					}
					else {
						indx_edge_i[++c2] = c1;
						edge_count_i[s]++;
					}
				}

				edges_elem[i][j] = newEdgeMap[spt][ept];
			} //end of loop j
		} //end of loop i
		t += elem_count[s];
	} //end of loop s

	// modify edges according to parition results
	te = new int* [Nedge];
	for (int i=0; i<Nedge; i++) te[newEdgeMap[edges[i][0]][edges[i][1]]] = edges[edgeMap[edges[i][0]][edges[i][1]]];
	for (int i=0; i<Nedge; i++) edges[i] = te[i];
	delete[] te;




	////----------------------------------------
	//// modify edges according to new node ordering
	////----------------------------------------

	//// suppose we don't know Nedge, Nedge_b, Nedge_i
	//// first find the values for them
	//// possibly to improve this using some result in graph theory????
	////bitset<Nnod*Nnod> edgeBits; // given a edge (spt, ept), set bit (spt*Nnod+ept), initally all zeros
	////bitset<Nnod*Nnod> bdryEdgeBits;

	////vector<bool> edgeFlags(Nnod*Nnod, false); // given a edge (spt, ept), set flag (spt*Nnod+ept), initally all false
	//vector<bool> bdryEdgeFlags(Nnod*Nnod, false);
	//vector<int> edgeMap(Nnod*Nnod, -1); 

	//Nedge = 0; 
	//Nedge_b = 0;

	//for (int i=0; i<Nel; i++) {
	//	for (int j=0; j<6; j++) {
	//		int spt = elems[i][edgeK[0][j]];
	//		int ept = elems[i][edgeK[1][j]];
	//		if (ept<spt) swap(spt, ept);

	//		//check if the edge is on the boundary
	//		if (coords[spt][3]>0.1 && coords[ept][3]>0.1)
	//			if (!bdryEdgeFlags[spt*Nnod+ept]) {
	//				bdryEdgeFlags[spt*Nnod+ept] = true;
	//				Nedge_b++;
	//			}
	//							
	//		//count the edge
	//		if (edgeMap[spt*Nnod+ept]<0)
	//			edgeMap[spt*Nnod+ept] = Nedge++;	
	//	}
	//}
	//
	//Nedge_i = Nedge - Nedge_b; 

	//// what is edges used for? can we use edges_elem to replace it????
	//// edges is used to assemble the interpolation from nodal space to edge space, possibly need to keep...
	//// edges-elem is needed to identify the sparsity patten, possibly need to keep
	//// solution: assemble edges, and edges_elem, drop EdgeIndx
	//
	//// allocate space for edges, edges_elem
	//edges = new int* [Nedge];
	//edges_elem = new int* [Nel];

	////allocate space for indx_edge_i, indx_edge_b
	//indx_edge_i = new int[Nedge_i];
	//indx_edge_b = new int[Nedge_b];

	////array of length Nnod*Nnod, edgeMap[spt*Nnod+ept] containing the index of edge(spt, ept)
	//edgeMap = vector<int>(Nnod*Nnod, -1); 

	////allocate space fo edge_count_i, edge_count_b
	//edge_count_i = new int[sizesize];
	//edge_count_b = new int[sizesize];
	//
	//int c1=-1, c2=-1, c3=-1; //c1 counts the number of all edges, c2 counts the number of interior edges, c3 counts for # of bdry edges
	//t = 0;
	//for (int s=0; s<sizesize; s++) {
	//	edge_count_i[s]=0;
	//	edge_count_b[s]=0;
	//	for (int i=t; i<t+elem_count[s]; i++) {
	//		edges_elem[i] = new int[6];
	//		for (int j=0; j<6; j++) {
	//			int spt = elems[i][edgeK[0][j]];
	//			int ept = elems[i][edgeK[1][j]];
	//			if (ept<spt) swap(spt, ept);
	//			
	//			if (edgeMap[spt*Nnod+ept]<0) { //find a new edge 
	//				//allocate space for the new edge
	//				edges[++c1] = new int[3]; //use the third component as a bdry marker
	//				edges[c1][0] = spt;
	//				edges[c1][1] = ept;
	//				edgeMap[spt*Nnod+ept] = c1;

	//				if (bdryEdgeFlags[spt*Nnod+ept]) { //check if it is a bdry edge
	//					indx_edge_b[++c3] = c1;
	//					edges[c1][2] = 1; 
	//					edge_count_b[s]++;
	//				}
	//				else {
	//					indx_edge_i[++c2] = c1;
	//					edges[c1][2] = 0;
	//					edge_count_i[s]++;
	//				}
	//			}

	//			edges_elem[i][j] = edgeMap[spt*Nnod+ept];
	//		} //end of loop j
	//	} //end of loop i
	//	t += elem_count[s];
	//} //end of loop s

	//cout<<"coords"<<endl;
	//for (int i=0; i<Nnod; i++)
	//	cout<<"["<<i<<"]"<<coords[i][0]<<","<<coords[i][1]<<","<<coords[i][2]<<","<<coords[i][3]<<endl;

	//cout<<"elems"<<endl;
	//for (int i=0; i<Nel; i++)
	//	cout<<"["<<i<<"]"<<elems[i][0]<<","<<elems[i][1]<<","<<elems[i][2]<<","<<elems[i][3]<<endl;

	//cout<<"edges"<<endl;
	//for (int i=0; i<Nedge; i++)
	//	cout<<edges[i][0]<<","<<edges[i][1]<<","<<edges[i][2]<<endl;
}


void MESH::Create_is(void)
{

	//get indices of local interior nodes
	int t=0;
	for (int i=0; i<RANK; i++)
		t += nod_count_i[i]+nod_count_b[i];

	//allocate space for local_nz
	int *local_nz = new int[nod_count_i[RANK]];
	int *local_z  = new int[nod_count_b[RANK]];
	int pos_i=0, pos_b=0;
	for (int i=t; i<t+nod_count_i[RANK]+nod_count_b[RANK]; i++)
		if (fabs(coords[i][3])<0.1) //Note coords[i][3] is double, use 0.1 instead of 1 to get rid off fl. pt. round off errors,
			local_nz[pos_i++] = i;
		else
			local_z[pos_b++] = i;

	ISCreateGeneral(PETSC_COMM_WORLD,nod_count_i[RANK], local_nz,&is_nod_i);
	ISCreateGeneral(PETSC_COMM_WORLD,nod_count_b[RANK], local_z,&is_nod_b);

	//get indices of local interior edges
	t=0;
	for (int i=0; i<RANK; i++)
		t += edge_count_i[i]+edge_count_b[i];

	//allocate space for local_nz
	local_nz = new int[edge_count_i[RANK]];
	local_z  = new int[edge_count_b[RANK]];
	pos_i=0;
	pos_b=0;
	for (int i=t; i<t+edge_count_i[RANK]+edge_count_b[RANK]; i++)
		if (!edges[i][2]) local_nz[pos_i++] = i;
		else local_z[pos_b++] = i;

	ISCreateGeneral(PETSC_COMM_WORLD,edge_count_i[RANK], local_nz,&is_edge_i);
	ISCreateGeneral(PETSC_COMM_WORLD,edge_count_b[RANK], local_z,&is_edge_b);

	delete[] local_nz;
	delete[] local_z;
	
	//PetscPrintf(PETSC_COMM_WORLD, "is_edge_i\n");
	//ISView(is_edge_i,PETSC_VIEWER_STDOUT_WORLD);
	//PetscPrintf(PETSC_COMM_WORLD, "is_edge_b\n");
	//ISView(is_edge_b,PETSC_VIEWER_STDOUT_WORLD);
	//PetscPrintf(PETSC_COMM_WORLD, "is_nod_i\n");
	//ISView(is_nod_i,PETSC_VIEWER_STDOUT_WORLD);
	//PetscPrintf(PETSC_COMM_WORLD, "is_nod_b\n");
	//ISView(is_nod_b,PETSC_VIEWER_STDOUT_WORLD);
}

/*
void MESH::Create_sparsity(void)
{
	// get sparsity for edges
	int nLocalEdges = edge_count_i[RANK] + edge_count_b[RANK];
	int nLocalNodes = nod_count_i[RANK] + nod_count_b[RANK];

	// allocate space for unique
	vector<set<int> > v_d_nnz_nod(nLocalNodes);
	vector<set<int> > v_d_nnz_edge(nLocalEdges);
	vector<set<int> > v_d_nnz_edge_nod(nLocalEdges);
	vector<set<int> > v_d_nnz_nod_edge(nLocalNodes);

	vector<set<int> > v_o_nnz_nod(nLocalNodes);
	vector<set<int> > v_o_nnz_edge(nLocalEdges);
	vector<set<int> > v_o_nnz_edge_nod(nLocalEdges);
	vector<set<int> > v_o_nnz_nod_edge(nLocalNodes);


	int edgeStart = 0;
	int edgeEnd = 0;
	for (int s=0; s<RANK; s++) 
		edgeStart += edge_count_i[s] + edge_count_b[s];
	edgeEnd = edgeStart + edge_count_i[RANK] + edge_count_b[RANK];

	int nodeStart = 0;
	int nodeEnd = 0;
	for (int s=0; s<RANK; s++) 
		nodeStart += nod_count_i[s] + nod_count_b[s];
	nodeEnd = nodeStart + nod_count_i[RANK] + nod_count_b[RANK];


	// find # of nnz
	for (int n=0; n<Nel; n++) {

		// find sparisty for edges
		for (int i=0; i<6; i++) {
			if (edges_elem[n][i]>=edgeStart && edges_elem[n][i]<edgeEnd) {
				for (int j=0; j<6; j++) {				
					if (edges_elem[n][j]>=edgeStart && edges_elem[n][j]<edgeEnd) //find a diagonal non zero
						//d_nnz_edge[edges_elem[n][i]-edgeStart]++;
						v_d_nnz_edge[edges_elem[n][i]-edgeStart].insert(edges_elem[n][j]);
					else 
						//o_nnz_edge[edges_elem[n][i]-edgeStart]++;
						v_o_nnz_edge[edges_elem[n][i]-edgeStart].insert(edges_elem[n][j]);
				}
			}
		}

		//find sparsity for nodes
		for (int i=0; i<4; i++) {
			if (elems[n][i]>=nodeStart && elems[n][i]<nodeEnd) {
				for (int j=0; j<4; j++) {				
					if (elems[n][j]>=nodeStart && elems[n][j]<nodeEnd) //find a diagonal non zero
						//d_nnz_nod[elems[n][i]-nodeStart]++;
						v_d_nnz_nod[elems[n][i]-nodeStart].insert(elems[n][j]);
					else 
						//o_nnz_nod[elems[n][i]-nodeStart]++;
						v_o_nnz_nod[elems[n][i]-nodeStart].insert(elems[n][j]);
				}
			}
		}

		// find sparisty for node-edge
		for (int i=0; i<4; i++) {
			if (elems[n][i]>=nodeStart && elems[n][i]<nodeEnd) {
				for (int j=0; j<6; j++) {				
					if (edges_elem[n][j]>=edgeStart && edges_elem[n][j]<edgeEnd) //find a diagonal non zero
						//d_nnz_nod_edge[elems[n][i]-nodeStart]++;
						v_d_nnz_nod_edge[elems[n][i]-nodeStart].insert(edges_elem[n][j]);
					else 
						//o_nnz_nod_edge[elems[n][i]-nodeStart]++;
						v_o_nnz_nod_edge[elems[n][i]-nodeStart].insert(edges_elem[n][j]);
				}
			}
		}


		// find sparisty for edge-node
		for (int i=0; i<6; i++) {
			if (edges_elem[n][i]>=edgeStart && edges_elem[n][i]<edgeEnd) {
				for (int j=0; j<4; j++) {				
					if (elems[n][j]>=nodeStart && elems[n][j]<nodeEnd) //find a diagonal non zero
						//d_nnz_edge_nod[edges_elem[n][i]-edgeStart]++;
						v_d_nnz_edge_nod[edges_elem[n][i]-edgeStart].insert(elems[n][j]);
					else 
						//o_nnz_edge_nod[edges_elem[n][i]-edgeStart]++;
						v_o_nnz_edge_nod[edges_elem[n][i]-edgeStart].insert(elems[n][j]);
				}
			}
		}

	}

	// allocate space for edge nnz
	d_nnz_edge = new int[nLocalEdges]; //diagonal nonzeros
	o_nnz_edge = new int[nLocalEdges]; //off diagonal nonzeros
	for (int i=0; i<nLocalEdges; i++) {
		d_nnz_edge[i] = v_d_nnz_edge[i].size();
		o_nnz_edge[i] = v_o_nnz_edge[i].size();
	}

	// allocate space for node nnz
	d_nnz_nod = new int[nLocalNodes]; //diagonal nonzeros
	o_nnz_nod = new int[nLocalNodes]; //off diagonal nonzeros
	for (int i=0; i<nLocalNodes; i++) {
		d_nnz_nod[i] = v_d_nnz_nod[i].size();
		o_nnz_nod[i] = v_o_nnz_nod[i].size();
	}

	// allocate space for node-edge nnz
	d_nnz_nod_edge = new int[nLocalNodes]; //diagonal nonzeros
	o_nnz_nod_edge = new int[nLocalNodes]; //off diagonal nonzeros
	for (int i=0; i<nLocalNodes; i++) {
		d_nnz_nod_edge[i] = v_d_nnz_nod_edge[i].size();
		o_nnz_nod_edge[i] = v_o_nnz_nod_edge[i].size();
	}

	// allocate space for edge-node nnz
	d_nnz_edge_nod = new int[nLocalEdges]; //diagonal nonzeros
	o_nnz_edge_nod = new int[nLocalEdges]; //off diagonal nonzeros
	for (int i=0; i<nLocalEdges; i++) {
		d_nnz_edge_nod[i] = v_d_nnz_edge_nod[i].size();
		o_nnz_edge_nod[i] = v_o_nnz_edge_nod[i].size();
	}


}
*/



void MESH::Create_sparsity(void)
{
	// get sparsity for edges
	int nLocalEdges = edge_count_i[RANK] + edge_count_b[RANK];
	int nLocalNodes = nod_count_i[RANK] + nod_count_b[RANK];

	// allocate space for edge nnz
	d_nnz_edge = new int[nLocalEdges]; //diagonal nonzeros
	o_nnz_edge = new int[nLocalEdges]; //off diagonal nonzeros
	for (int i=0; i<nLocalEdges; i++) {
		d_nnz_edge[i] = 0;
		o_nnz_edge[i] = 0;
	}

	// allocate space for node nnz
	d_nnz_nod = new int[nLocalNodes]; //diagonal nonzeros
	o_nnz_nod = new int[nLocalNodes]; //off diagonal nonzeros
	for (int i=0; i<nLocalNodes; i++) {
		d_nnz_nod[i] = 0;
		o_nnz_nod[i] = 0;
	}

	// allocate space for node-edge nnz
	d_nnz_nod_edge = new int[nLocalNodes]; //diagonal nonzeros
	o_nnz_nod_edge = new int[nLocalNodes]; //off diagonal nonzeros
	for (int i=0; i<nLocalNodes; i++) {
		d_nnz_nod_edge[i] = 0;
		o_nnz_nod_edge[i] = 0;
	}

	// allocate space for edge-node nnz
	d_nnz_edge_nod = new int[nLocalEdges]; //diagonal nonzeros
	o_nnz_edge_nod = new int[nLocalEdges]; //off diagonal nonzeros
	for (int i=0; i<nLocalEdges; i++) {
		d_nnz_edge_nod[i] = 0;
		o_nnz_edge_nod[i] = 0;
	}


	int edgeStart = 0;
	int edgeEnd = 0;
	for (int s=0; s<RANK; s++) 
		edgeStart += edge_count_i[s] + edge_count_b[s];
	edgeEnd = edgeStart + edge_count_i[RANK] + edge_count_b[RANK];

	int nodeStart = 0;
	int nodeEnd = 0;
	for (int s=0; s<RANK; s++) 
		nodeStart += nod_count_i[s] + nod_count_b[s];
	nodeEnd = nodeStart + nod_count_i[RANK] + nod_count_b[RANK];


	// find # of nnz
	for (int n=0; n<Nel; n++) {

		// find sparisty for edges
		for (int i=0; i<6; i++) {
			if (edges_elem[n][i]>=edgeStart && edges_elem[n][i]<edgeEnd) {
				for (int j=0; j<6; j++) {				
					if (edges_elem[n][j]>=edgeStart && edges_elem[n][j]<edgeEnd) //find a diagonal non zero
						d_nnz_edge[edges_elem[n][i]-edgeStart]++;
					else 
						o_nnz_edge[edges_elem[n][i]-edgeStart]++;
				}
			}
		}

		//find sparsity for nodes
		for (int i=0; i<4; i++) {
			if (elems[n][i]>=nodeStart && elems[n][i]<nodeEnd) {
				for (int j=0; j<4; j++) {				
					if (elems[n][j]>=nodeStart && elems[n][j]<nodeEnd) //find a diagonal non zero
						d_nnz_nod[elems[n][i]-nodeStart]++;
					else 
						o_nnz_nod[elems[n][i]-nodeStart]++;
				}
			}
		}

		// find sparisty for node-edge
		for (int i=0; i<4; i++) {
			if (elems[n][i]>=nodeStart && elems[n][i]<nodeEnd) {
				for (int j=0; j<6; j++) {				
					if (edges_elem[n][j]>=edgeStart && edges_elem[n][j]<edgeEnd) //find a diagonal non zero
						d_nnz_nod_edge[elems[n][i]-nodeStart]++;
					else 
						o_nnz_nod_edge[elems[n][i]-nodeStart]++;
				}
			}
		}


		// find sparisty for edge-node
		for (int i=0; i<6; i++) {
			if (edges_elem[n][i]>=edgeStart && edges_elem[n][i]<edgeEnd) {
				for (int j=0; j<4; j++) {				
					if (elems[n][j]>=nodeStart && elems[n][j]<nodeEnd) //find a diagonal non zero
						d_nnz_edge_nod[edges_elem[n][i]-edgeStart]++;
					else 
						o_nnz_edge_nod[edges_elem[n][i]-edgeStart]++;
				}
			}
		}

	}

	// allocate space for edge nnz
	for (int i=0; i<nLocalEdges; i++) {
		d_nnz_edge[i] = min(d_nnz_edge[i], edgeEnd-edgeStart);
		o_nnz_edge[i] = min(o_nnz_edge[i], Nedge-d_nnz_edge[i]);
	}

	// allocate space for node nnz
	for (int i=0; i<nLocalNodes; i++) {
		d_nnz_nod[i] = min(d_nnz_nod[i], nodeEnd-nodeStart);
		o_nnz_nod[i] = min(o_nnz_nod[i], Nnod-d_nnz_nod[i]);
	}

	// allocate space for node-edge nnz
	for (int i=0; i<nLocalNodes; i++) {
		d_nnz_nod_edge[i] = min(d_nnz_nod_edge[i], edgeEnd-edgeStart);
		o_nnz_nod_edge[i] = min(o_nnz_nod_edge[i], Nedge-d_nnz_nod_edge[i]);
	}

	// allocate space for edge-node nnz
	for (int i=0; i<nLocalEdges; i++) {
		d_nnz_edge_nod[i] = min(d_nnz_edge_nod[i], nodeEnd-nodeStart);
		o_nnz_edge_nod[i] = min(o_nnz_edge_nod[i], Nnod-d_nnz_edge_nod[i]);
	}


}



void MESH::Create(char *name)
{
	strcpy(filename, name);

	Create_coords();
	Create_elems();
	Create_edges();
	//Create_neighs();

	Create_partition();
	Create_sparsity();
	Create_is();
}


void MESH::View(void)
{
	PetscViewer viewer;
	PetscErrorCode ierr;
	PetscViewerFormat format; 

	//format = PETSC_VIEWER_ASCII_DEFAULT; //PETSC_VIEWER_ASCII_MATLAB
	format = PETSC_VIEWER_ASCII_MATLAB;

	//ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "out_EdgeIndx.txt", &viewer);
	//ierr = PetscViewerPushFormat(viewer, format);
	//ierr = PetscObjectSetName((PetscObject)EdgeIndx, "EdgeIndx");
	//ierr = MatView(EdgeIndx, viewer);
	//PetscViewerDestroy(viewer);

}

void MESH::Destroy(void)
{

	//MatDestroy(EdgeIndx);
	ISDestroy(is_edge_i);
	ISDestroy(is_edge_b);
	ISDestroy(is_nod_i);
	ISDestroy(is_nod_b);

	for (int i=0; i<Nnod; i++)
		delete[] coords[i];
	delete[] coords;

	for (int i=0; i<Nel; i++)
		delete[] elems[i];
	delete[] elems;

	for (int i=0; i<Nedge; i++)
		delete[] edges[i];
	delete[] edges;

	//for (int i=0; i<Nel; i++)
	//	delete[] neighs[i];
	//delete[] neighs;

	for (int i=0; i<Nel; i++)
		delete[] edges_elem[i];
	delete[] edges_elem;


	delete[] indx_edge_i;
	delete[] indx_edge_b;

	delete[] indx_nod_i;
	delete[] indx_nod_b;

	delete[] elem_count;
	delete[] nod_count_i;
	delete[] nod_count_b;
	delete[] edge_count_i;
	delete[] edge_count_b;

	delete[] d_nnz_edge;
	delete[] o_nnz_edge;
	delete[] d_nnz_nod;	
	delete[] o_nnz_nod;
	delete[] d_nnz_nod_edge;
	delete[] o_nnz_nod_edge;
	delete[] d_nnz_edge_nod;
	delete[] o_nnz_edge_nod;

}


//-------------------------------------------------------
// incorporate my own mesher
//-------------------------------------------------------

void MESH::Create_coords(int N)
{
	double xmax, xmin, ymax, ymin, zmax, zmin;
	xmax = ymax = zmax = 1.0;
	xmin = ymin = zmin = -1.0;

	double hx = (xmax - xmin)/N;
	double hy = (ymax - ymin)/N;
	double hz = (zmax - zmin)/N;


	//allocate space for coords
	coords = new double* [Nnod];
	for (int i=0; i<Nnod; i++) coords[i] = new double [4];

	for (int k = 0; k < N+1; k++)
		for (int j = 0; j < N+1; j++)
			for (int i = 0; i < N+1; i++){
				int indx = k*(N+1)*(N+1) + j*(N+1) + i;
				coords[indx][0] = i*hx+xmin;
				coords[indx][1] = j*hy+ymin;
				coords[indx][2] = k*hz+zmin;
				coords[indx][3] = (k==0 || k==N || j==0 || j==N || i==0 || i==N);
			}

}


void MESH::Create_elems(int N)
{

	int elemsK[4][6] = {{3, 3, 4, 4, 4, 3},
						{1, 7, 6, 7, 1, 0},
						{7, 6, 7, 1, 7, 6},
						{0, 0, 0, 0, 5, 2}};

	//allocate space for elems
	elems = new int* [Nel];
	for (int i=0; i<Nel; i++) {
		elems[i] =  new int [5];
		elems[i][4] = 0;	//default region property
	}


	int Ncnt1 = 0;
	for (int k = 0; k < N; k++)
		for (int j = 0; j < N; j++)
			for (int i = 0; i < N; i++){
				int indx = k*(N+1)*(N+1) + j*(N+1) + i;
				Ncnt1 = k*N*N + j*N + i;
				int node[8];
				node[0] = indx;
				node[1] = indx + 1;
				node[2] = indx + N + 1;
				node[3] = indx + N + 2;
				node[4] = indx + (N+1)*(N+1);
				node[5] = indx + 1 + (N+1)*(N+1);
				node[6] = indx + N + 1 + (N+1)*(N+1);
				node[7] = indx + N + 2 + (N+1)*(N+1);
				for (int ii = 0; ii < 4; ii++)
					for (int jj = 0; jj < 6; jj++)
						elems[Ncnt1*6+jj][ii] = node[elemsK[ii][jj]];
			}


}

void MESH::Create_edges(int N)
{

	PetscInt elemsK[4][6] = {{3, 3, 4, 4, 4, 3},
						{1, 7, 6, 7, 1, 0},
						{7, 6, 7, 1, 7, 6},
						{0, 0, 0, 0, 5, 2}};
	
	edges = new int* [Nedge];
	for (int i=0; i<Nedge; i++)
		edges[i] = new int[3];

	int indx;
	int Ncnt1, Ncnt2;
    bool layer0, layerN, layerk; 

	int MinIndx, MaxIndx;


	Ncnt1 = -1; //internal
	Ncnt2 = Nedge_i-1; //bdry

	
	//loop local layers according to k
	for (int k = 0; k < N+1; k++){
		layer0 = (k==0);
		layerN = (k==N);
		layerk = !layer0 && !layerN;


		//center_internal N*N
		if (layer0 || layerk) 
			for (int j=0; j<N; j++)
				for (int i=0; i<N; i++) {
					indx = k*(N+1)*(N+1) + j*(N+1) + i;
					MinIndx = indx;
					MaxIndx = indx+(N+1)*(N+1)+(N+1) + 1;
                    Ncnt1++;
					edges[Ncnt1][0] = MinIndx;
					edges[Ncnt1][1] = MaxIndx;
					edges[Ncnt1][2] = 0; 
				}
					

		//ox_internal (N-1)*N
		for (int j=1; j<N; j++)
			for (int i=0; i<N; i++){
				indx = k*(N+1)*(N+1) + j*(N+1) + i;
				MinIndx = indx;
				MaxIndx = indx+1;
				if (layerk) {
                    Ncnt1++;
					edges[Ncnt1][0] = MinIndx;
					edges[Ncnt1][1] = MaxIndx;
					edges[Ncnt1][2] = 0; 
				}
				if (layer0 || layerN) {
                    Ncnt2++;
					edges[Ncnt2][0] = MinIndx;
					edges[Ncnt2][1] = MaxIndx;
					edges[Ncnt2][2] = 1; 
				}
			}

		//ox_bdry (N-1)*N
		for (int j=0; j<N+1; j+=N)
			for (int i=0; i<N; i++){
				indx = k*(N+1)*(N+1) + j*(N+1) + i;
				MinIndx = indx;
				MaxIndx = indx+1;
                Ncnt2++;
				edges[Ncnt2][0] = MinIndx;
				edges[Ncnt2][1] = MaxIndx;
				edges[Ncnt2][2] = 1; 
			}

		//oy_internal (N-1)*N
		for (int j=0; j<N; j++)
			for (int i=1; i<N; i++){
				indx = k*(N+1)*(N+1) + j*(N+1) + i;
				MinIndx = indx;
				MaxIndx = indx+N+1;
				if (layerk) {
                    Ncnt1++;
					edges[Ncnt1][0] = MinIndx;
					edges[Ncnt1][1] = MaxIndx;
					edges[Ncnt1][2] = 0; 
				}
				if (layer0 || layerN) {
                    Ncnt2++;
					edges[Ncnt2][0] = MinIndx;
					edges[Ncnt2][1] = MaxIndx;
					edges[Ncnt2][2] = 1; 
				}
			}

		//oy_bdry (N-1)*N
		for (int j=0; j<N; j++)
			for (int i=0; i<N+1; i+=N){
				indx = k*(N+1)*(N+1) + j*(N+1) + i;
				MinIndx = indx;
				MaxIndx = indx+N+1;
                Ncnt2++;
				edges[Ncnt2][0] = MinIndx;
				edges[Ncnt2][1] = MaxIndx;
				edges[Ncnt2][2] = 1; 
			}

		//oz_internal (N-1)*(N-1)
		if (layerk || layer0) {
			for (int j=1; j<N; j++)
				for (int i=1; i<N; i++){
					indx = k*(N+1)*(N+1) + j*(N+1) + i;
					MinIndx = indx;
					MaxIndx = indx+(N+1)*(N+1);
                    Ncnt1++;
					edges[Ncnt1][0] = MinIndx;
					edges[Ncnt1][1] = MaxIndx;
					edges[Ncnt1][2] = 0; 
				}
		}

		//oz_bdry 4*N
		if (layerk || layer0) {
			for (int j=0; j<N+1; j+=N)
				for (int i=0; i<N+1; i++){
					indx = k*(N+1)*(N+1) + j*(N+1) + i;
					MinIndx = indx;
					MaxIndx = indx+(N+1)*(N+1);
					Ncnt2++;
					edges[Ncnt2][0] = MinIndx;
					edges[Ncnt2][1] = MaxIndx;
					edges[Ncnt2][2] = 1; 
				}
			for (int j=1; j<N; j++)
				for (int i=0; i<N+1; i+=N){
					indx = k*(N+1)*(N+1) + j*(N+1) + i;
					MinIndx = indx;
					MaxIndx = indx+(N+1)*(N+1);
					Ncnt2++;
					edges[Ncnt2][0] = MinIndx;
					edges[Ncnt2][1] = MaxIndx;
					edges[Ncnt2][2] = 1; 
				}
		}

		//bottom N*N
		for (int j=0; j<N; j++)
			for (int i=0; i<N; i++) {
				indx = k*(N+1)*(N+1) + j*(N+1) + i;
				MinIndx = indx;
				MaxIndx = indx+(N+1)+1;
				if (layerk) {
                    Ncnt1++;
					edges[Ncnt1][0] = MinIndx;
					edges[Ncnt1][1] = MaxIndx;
					edges[Ncnt1][2] = 0; 
				}
				if (layer0 || layerN) {
                    Ncnt2++;
					edges[Ncnt2][0] = MinIndx;
					edges[Ncnt2][1] = MaxIndx;
					edges[Ncnt2][2] = 1; 
				}
			}

		//front 
		if (layerk || layer0) {
			//front_internal (N-1)*N
			for (int j=1; j<N; j++)
				for (int i=0; i<N; i++){
					indx = k*(N+1)*(N+1) + j*(N+1) + i;
					MinIndx = indx + 1;
					MaxIndx = indx+(N+1)*(N+1);
					Ncnt1++;
					edges[Ncnt1][0] = MinIndx;
					edges[Ncnt1][1] = MaxIndx;
					edges[Ncnt1][2] = 0; 
				}
			//front_bdry 2*N
			for (int j=0; j<N+1; j+=N)
				for (int i=0; i<N; i++){
					indx = k*(N+1)*(N+1) + j*(N+1) + i;
					MinIndx = indx + 1;
					MaxIndx = indx+(N+1)*(N+1);
					Ncnt2++;
					edges[Ncnt2][0] = MinIndx;
					edges[Ncnt2][1] = MaxIndx;
					edges[Ncnt2][2] = 1; 
				}
		}


		//left 
		if (layerk || layer0) {
			//left_internal (N-1)*N
			for (int j=0; j<N; j++)
				for (int i=1; i<N; i++){
					indx = k*(N+1)*(N+1) + j*(N+1) + i;
					MinIndx = indx;
					MaxIndx = indx+(N+1)*(N+1) + (N+1);
					Ncnt1++;
					edges[Ncnt1][0] = MinIndx;
					edges[Ncnt1][1] = MaxIndx;
					edges[Ncnt1][2] = 0; 
				}
			//left_bdry 2*N
			for (int j=0; j<N; j++)
				for (int i=0; i<N+1; i+=N){
					indx = k*(N+1)*(N+1) + j*(N+1) + i;
					MinIndx = indx;
					MaxIndx = indx+(N+1)*(N+1) + (N+1);
					Ncnt2++;
					edges[Ncnt2][0] = MinIndx;
					edges[Ncnt2][1] = MaxIndx;
					edges[Ncnt2][2] = 1; 
				}
		}
		
	}



}


void MESH::Create(int N)
{
	sprintf(filename, "mycube_N%d", N);


	Nel = N*N*N*6;
	Nnod = (N+1)*(N+1)*(N+1);
	Nnod_i = (N-1)*(N-1)*(N-1);
	Nnod_b = Nnod - Nnod_i;
	Nedge = 7*N*N*N + 9*N*N + 3*N;
	Nedge_b = 18*N*N;
	Nedge_i = Nedge - Nedge_b;

	PetscPrintf(PETSC_COMM_WORLD, "Nnod=%d, Nnod_i=%d, Nel=%d, Nedge=%d, Nedge_i=%d, DOFs =%d\n", Nnod, Nnod_i, Nel, Nedge, Nedge_i, Nnod_i+Nedge_i);
	PetscSynchronizedFlush(PETSC_COMM_WORLD); 


	Create_coords(N);
	Create_elems(N);
	Create_edges(N);

	Create_partition();
	Create_sparsity();
	Create_is();
}


//create output for partitioner mesh visualizer
void MESH::Create_pmvoutput(int *epart)
{
	char fn[1024];
	ofstream fout;

	//output coords
	sprintf(fn, "%s.xyz", filename);
	fout.open(fn, ofstream::out);
	for (int i=0; i<Nnod; i++) {
		for (int j=0; j<3; j++)
			fout<<coords[i][j]<<" ";
		fout<<endl;
	}
	fout.close();

	//output elems
	sprintf(fn, "%s.con", filename);
	fout.open(fn, ofstream::out);
	for (int i=0; i<Nel; i++) {
		for (int j=0; j<4; j++)
			fout<<elems[i][j]<<" ";
		fout<<endl;
	}
	fout.close();

	//output partitioning
	sprintf(fn, "%s.par", filename);
	fout.open(fn, ofstream::out);
	for (int i=0; i<Nel; i++)
		fout<<epart[i]<<endl;
	fout.close();
	
	//// special code to output coefficient distribution of the cube, Fichera corner
	//sprintf(fn, "%s.par", filename);
	//fout.open(fn, ofstream::out);
	//for (int i=0; i<Nel; i++)
	//	fout<<elems[i][4]/111-1<<endl;
	//fout.close();

	//// special code to output coefficient distribution of the gear
	//sprintf(fn, "%s.par", filename);
	//fout.open(fn, ofstream::out);
	//for (int i=0; i<Nel; i++) {
	//	double x=0, y=0;
	//	for (int j=0; j<4; j++) {
	//		x += coords[elems[i][j]][0];
	//		y += coords[elems[i][j]][1];
	//	}
	//	if (x>0 && y>0) fout<<0<<endl;
	//	else if (x>0 && y<0) fout<<1<<endl;
	//	else if (x<0 && y>0) fout<<2<<endl;
	//	else fout<<3<<endl;
	//}
	//fout.close();

	//// special code to output coefficient distribution of m747
	//sprintf(fn, "%s.par", filename);
	//fout.open(fn, ofstream::out);
	//for (int i=0; i<Nel; i++) {
	//	double x=0, z=0;
	//	for (int j=0; j<4; j++) {
	//		x += coords[elems[i][j]][0];
	//		z += coords[elems[i][j]][2];
	//	}
	//	x *= 0.25;
	//	z *= 0.25;
	//	if (x>0.5 && z>0.5) fout<<0<<endl;
	//	else if (x>0.5 && z<0.5) fout<<1<<endl;
	//	else if (x<0.5 && z>0.5) fout<<2<<endl;
	//	else fout<<3<<endl;
	//}
	//fout.close();



	//generate a .bat file for convenience
	sprintf(fn, "%s.bat", filename);
	fout.open(fn, ofstream::out);
	sprintf(fn, "pmvis -n %s.xyz -c %s.con -p %s.par -o 0 -g tet", filename, filename, filename);
	fout<<fn<<endl;
	fout.close();

}
