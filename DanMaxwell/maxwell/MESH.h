#pragma once
#include <vector>
#include "petscksp.h"

using namespace std;

class MESH
{
public:
	//variables
	PetscInt Nnod, Nnod_i, Nnod_b;
	PetscInt Nel;
	PetscInt Nedge, Nedge_i, Nedge_b;
	int Nface;

	bool genpartition;

	//vector<bool> bdryMarkers;

	//Mat EdgeIndx;	//useless??
	IS is_edge_i;
	IS is_nod_i;
	IS is_edge_b;
	IS is_nod_b;

	
	double **coords; //Nnod*4, coords[N][0..2] are the coordinates, coords[N][3] is the bdry marker, note it is a double, don't do coord[3]==0
	int **elems;	//Nel*5, elems[N][4] is the region marker
	int **edges;	//Nedge*3, edges[N][2] is the bdry marker, 0: iterior, 1: bdry 
	int **neighs;	//Nel*4, list of neigboring elements, -1 marks a non-existing neighbor
	int **faces;	//Nface*4, faces[N][0..2] are the indices of a surface triangle. faces[N][3] is boundary marker.
	int **edges_elem; //contains Nel rows, each row has 6 components: the indices of the six edges in a tet

	int *indx_edge_i;	//array of size Nedge_i, containing indices of interior edges
	int *indx_edge_b;	//array of size Nedge_b, containing indices of bdry edges 	
	int *indx_nod_i;
	int *indx_nod_b;

	int *elem_count;	//array of the comm size, containing the number of elements on the local processor
	int *nod_count_i;	//ditto
	int *nod_count_b;	//ditto
	int *edge_count_i;	//ditto
	int *edge_count_b;	//ditto

	char filename[100];

	int *d_nnz_edge;	//array of size local edges, d_nnz_edge[i] is the # of diagonal nnz of row i in matrices associated with edge DOFs 
	int *o_nnz_edge;	//off diagonal nnz

	int *d_nnz_nod;	//array of size local nodes, diagonal nnz associated with node DOFs
	int *o_nnz_nod;	//off diagonal nnz

	int *d_nnz_nod_edge; //array of size local nodes, diagonal nnz associated with matrix of size(Nnod, Nedge)
	int *o_nnz_nod_edge;

	int *d_nnz_edge_nod; //array of size local edges, diagonal nnz associated with matrix of size(Nedge, Nnod)
	int *o_nnz_edge_nod;

	//Functions
	MESH(void);
	~MESH(void);
	void Create(char *name);
	void Create(int N);	//create mesh using my own mesher
	void Destroy(void);
	void View(void);

	// create mesh using 
	void Create_coords(void);
	void Create_elems(void);
	void Create_edges(void);
	void Create_neighs(void);

	void Create_coords(int N);	//create coords using my mesher
	void Create_elems(int N);	//create elems using my mesher
	void Create_edges(int N);	//create edges using my mesher
	//need to add create_neighs for my own mesher????

	void Create_partition(void);
	void Create_is(void);
	void Create_sparsity(void);

	void Create_pmvoutput(int *epart);

};
