#include <dolfin.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/fem/fem_utils.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Vertex.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <stdlib.h>

namespace dolfin
{
    void Gradient(const FunctionSpace& V, const Array<int>& Mapping, double * column, double * row, double * data) {

        const Mesh mesh = *V.mesh();
        const GenericDofMap& dofmap_u = *V.dofmap();
        const int n = mesh.num_cells();
        const int m = mesh.num_edges();

        std::vector<int> edge2dof;
        edge2dof.reserve(m);

        Cell* dolfin_cell;

        for (int i = 0; i < n; ++i) {
            dolfin_cell = new Cell(mesh, i);
            std::vector<int> cellDOF = dofmap_u.cell_dofs(i);
            const uint* edgeVALUES = dolfin_cell->entities(1);
            for (int j = 0; j < cellDOF.size(); ++j) {
                edge2dof[edgeVALUES[j]]=cellDOF[j];
            }

        }


        Edge* dolfin_edge;
        int k = -1;
        for (int i = 0; i < m; ++i) {
            dolfin_edge = new Edge(mesh, i);
            const uint* edgeVERTICES = dolfin_edge->entities(0);

            k = k+1;
            row[k]=edge2dof[dolfin_edge->index()];
            column[k]=Mapping[edgeVERTICES[0]];
            data[k]=-1;
            k = k+1;
            row[k]=edge2dof[dolfin_edge->index()];
            column[k]=Mapping[edgeVERTICES[1]];
            data[k]=1;


        }

    }

    void ProlongationP(const FunctionSpace& V, const Array<int>& Mapping, const Array<double>& X, const Array<double>& Y, const Array<double>& Z, double * dataX, double * dataY, double * dataZ) {
        const Mesh mesh = *V.mesh();
        const GenericDofMap& dofmap_u = *V.dofmap();
        const int n = mesh.num_cells();
        const int m = mesh.num_edges();
        const int dim = mesh.geometry().dim();
        // std::vector<double> coord = mesh.coordinates();
        // /*
        // even are the x coords...
        // odd are the y coords...
        // */
        // std::vector<double> X;
        // std::vector<double> Y;
        // std::vector<double> Z;

        // for (int i = 0; i < coord.size()/dim; ++i)
        // {
        //     if (dim == 2) {
        //         std::cout << 2*i << std::endl;
        //         X.push_back(coord[2*i]);
        //         Y.push_back(coord[2*i+1]);
        //     }
        //     else {
        //         std::cout << 3*i << std::endl;
        //         X.push_back(coord[3*i]);
        //         Y.push_back(coord[3*i+1]);
        //         Z.push_back(coord[3*i+2]);
        //     }
        // }


        std::vector<double> tangent;
        std::vector<double> Ntangent;
        tangent.reserve(dim);
        Ntangent.reserve(dim);
        Edge* dolfin_edge;
        int k = -1;
        for (int i = 0; i < m; ++i) {
            k = k + 1;
            dolfin_edge = new Edge(mesh, i);
            const uint* edgeVERTICES = dolfin_edge->entities(0);

            tangent[0] = (X[edgeVERTICES[1]]-X[edgeVERTICES[0]]);
            tangent[1] = (Y[edgeVERTICES[1]]-Y[edgeVERTICES[0]]);
            if (dim == 3) {
                tangent[2] = Z[edgeVERTICES[1]]-Z[edgeVERTICES[0]];
            }

            for (int j = 0; j < dim; ++j) {
                if (dim == 3) {
                    if (tangent[0] == 0 && tangent[1] == 0 && tangent[2] == 0){
                        Ntangent[j] =0;
                    }
                    else {
                        Ntangent[j] = tangent[j]/(sqrt(pow(tangent[0],2.0)+pow(tangent[1],2.0)+pow(tangent[2],2.0)));
                    }
                }
                else {

                    if (tangent[0] == 0 && tangent[1] == 0){
                        Ntangent[j] =0;
                    }
                    else {
                        Ntangent[j] = tangent[j]/(sqrt(pow(tangent[0],2.0)+pow(tangent[1],2.0)));
                    }
                }
            }

            double len = sqrt(pow(tangent[0],2.0)+pow(tangent[1],2.0));


            // std::cout << len << " , " << dolfin_edge->length() << std::endl;
            dataX[k]=0.5*dolfin_edge->length()*Ntangent[0];
            dataY[k]=0.5*dolfin_edge->length()*Ntangent[1];

            if (dim == 3) {
                dataZ[k]=0.5*dolfin_edge->length()*Ntangent[2];
            }

            k = k + 1;
            dataX[k]=0.5*dolfin_edge->length()*Ntangent[0];
            dataY[k]=0.5*dolfin_edge->length()*Ntangent[1];

            if (dim == 3) {
                dataZ[k]=0.5*dolfin_edge->length()*Ntangent[2];
            }

        }


    }

    void Prolongation(const FunctionSpace& V, const Array<double>& Gx, const Array<double>& Gy, const Array<double>& Gz, double * dataX, double * dataY, double * dataZ) {
            const Mesh mesh = *V.mesh();
            const GenericDofMap& dofmap_u = *V.dofmap();
            const int n = mesh.num_cells();
            const int m = mesh.num_edges();

            std::vector<int> edge2dof;
            edge2dof.reserve(m);

            Cell* dolfin_cell;

            for (int i = 0; i < n; ++i) {
                dolfin_cell = new Cell(mesh, i);
                std::vector<int> cellDOF = dofmap_u.cell_dofs(i);
                const uint* edgeVALUES = dolfin_cell->entities(1);
                for (int j = 0; j < cellDOF.size(); ++j) {
                    edge2dof[edgeVALUES[j]]=cellDOF[j];
                }
            }

            Edge* dolfin_edge;
            int k = -1;
            for (int i = 0; i < m; ++i) {
                k = k + 1;
                dolfin_edge = new Edge(mesh, i);
                dataX[k] =Gx[dolfin_edge->index()]/2;
                dataY[k] =Gy[dolfin_edge->index()]/2;
                dataZ[k] =Gz[dolfin_edge->index()]/2;
                k = k + 1;
                dataX[k] =Gx[dolfin_edge->index()]/2;
                dataY[k] =Gy[dolfin_edge->index()]/2;
                dataZ[k] =Gz[dolfin_edge->index()]/2;
            }
        }





 void ProlongationGradsecond(const Mesh& mesh, double * dataX, double * dataY, double * dataZ, double * data, double * row, double * column ) {
        const int m = mesh.num_edges();
        const int dim = mesh.geometry().dim();

        std::vector<double> tangent;
        std::vector<double> Ntangent;
        tangent.reserve(dim);
        Ntangent.reserve(dim);

        Edge* dolfin_edge;

        Vertex* dolfin_vertex0;
        Vertex* dolfin_vertex1;

        int k = -1;
        for (int i = 0; i < m; ++i) {
            k = k + 1;
            dolfin_edge = new Edge(mesh, i);
            const uint* edgeVERTICES = dolfin_edge->entities(0);
            dolfin_vertex0 = new Vertex(mesh, edgeVERTICES[0]);
            dolfin_vertex1 = new Vertex(mesh,edgeVERTICES[1]);
            // std::cout << "edge " << i << " vertices " << edgeVERTICES[0] << " "  << edgeVERTICES[1] << std::endl;
            tangent[0] = (dolfin_vertex1->x(0)-dolfin_vertex0->x(0));
            tangent[1] = (dolfin_vertex1->x(1)-dolfin_vertex0->x(1));
            if (dim == 3) {
                tangent[2] = dolfin_vertex1->x(2)-dolfin_vertex0->x(2);
            }

            for (int j = 0; j < dim; ++j) {
                if (dim == 3) {
                    if (tangent[0] == 0 && tangent[1] == 0 && tangent[2] == 0){
                        Ntangent[j] =0;
                    }
                    else {
                        Ntangent[j] = tangent[j]/(sqrt(pow(tangent[0],2.0)+pow(tangent[1],2.0)+pow(tangent[2],2.0)));
                    }
                }
                else {

                    if (tangent[0] == 0 && tangent[1] == 0){
                        Ntangent[j] =0;
                    }
                    else {
                        Ntangent[j] = tangent[j]/(sqrt(pow(tangent[0],2.0)+pow(tangent[1],2.0)));
                    }
                }
            }


            // std::cout << len << " , " << dolfin_edge->length() << std::endl;
            dataX[k]=0.5*dolfin_edge->length()*Ntangent[0];
            dataY[k]=0.5*dolfin_edge->length()*Ntangent[1];

            if (dim == 3) {
                dataZ[k]=0.5*dolfin_edge->length()*Ntangent[2];
            }
            data[k] = 1;
            row[k] = i;
            column[k] = edgeVERTICES[1];
            // std::cout << dataX[k] << std::endl;
            k = k + 1;
            dataX[k]=0.5*dolfin_edge->length()*Ntangent[0];
            dataY[k]=0.5*dolfin_edge->length()*Ntangent[1];

            if (dim == 3) {
                dataZ[k]=0.5*dolfin_edge->length()*Ntangent[2];
            }
            data[k] = -1;
            row[k] = i;
            column[k] = edgeVERTICES[0];
            // std::cout << dataX[k] << std::endl;
        }



    }





    void ProlongationGradBoundary(const Mesh& mesh, double * EdgeBoundary, double * dataX, double * dataY, double * dataZ, double * data, double * row, double * column ) {
        const int m = mesh.num_edges();
        const int dim = mesh.geometry().dim();

        std::vector<double> tangent;
        std::vector<double> Ntangent;
        tangent.reserve(dim);
        Ntangent.reserve(dim);

        Edge* dolfin_edge;

        Vertex* dolfin_vertex0;
        Vertex* dolfin_vertex1;
        int kk = 0;
        int k = -1;
        for (int i = 0; i < m; ++i) {
            if (i == EdgeBoundary[kk]) {
                 // std::cout << "Boundary Edge found " << std::endl;
                 kk = kk+1;
            }
            else {
                k = k + 1;
                dolfin_edge = new Edge(mesh, i);
                const uint* edgeVERTICES = dolfin_edge->entities(0);
                dolfin_vertex0 = new Vertex(mesh, edgeVERTICES[0]);
                dolfin_vertex1 = new Vertex(mesh,edgeVERTICES[1]);
                // std::cout << "edge " << i << " vertices " << edgeVERTICES[0] << " "  << edgeVERTICES[1] << std::endl;
                tangent[0] = (dolfin_vertex1->x(0)-dolfin_vertex0->x(0));
                tangent[1] = (dolfin_vertex1->x(1)-dolfin_vertex0->x(1));
                if (dim == 3) {
                    tangent[2] = dolfin_vertex1->x(2)-dolfin_vertex0->x(2);
                }

                for (int j = 0; j < dim; ++j) {
                    if (dim == 3) {
                        if (tangent[0] == 0 && tangent[1] == 0 && tangent[2] == 0){
                            Ntangent[j] =0;
                        }
                        else {
                            Ntangent[j] = tangent[j]/(sqrt(pow(tangent[0],2.0)+pow(tangent[1],2.0)+pow(tangent[2],2.0)));
                        }
                    }
                    else {

                        if (tangent[0] == 0 && tangent[1] == 0){
                            Ntangent[j] =0;
                        }
                        else {
                            Ntangent[j] = tangent[j]/(sqrt(pow(tangent[0],2.0)+pow(tangent[1],2.0)));
                        }
                    }
                }


                // std::cout << len << " , " << dolfin_edge->length() << std::endl;
                dataX[k]=0.5*dolfin_edge->length()*Ntangent[0];
                dataY[k]=0.5*dolfin_edge->length()*Ntangent[1];

                if (dim == 3) {
                    dataZ[k]=0.5*dolfin_edge->length()*Ntangent[2];
                }
                data[k] = 1;
                row[k] = i;
                column[k] = edgeVERTICES[1];
                // std::cout << dataX[k] << std::endl;
                k = k + 1;
                dataX[k]=0.5*dolfin_edge->length()*Ntangent[0];
                dataY[k]=0.5*dolfin_edge->length()*Ntangent[1];

                if (dim == 3) {
                    dataZ[k]=0.5*dolfin_edge->length()*Ntangent[2];
                }
                data[k] = -1;
                row[k] = i;
                column[k] = edgeVERTICES[0];
            }
            // std::cout << dataX[k] << std::endl;
        }



    }

    void FaceToEdge(const Mesh& mesh, double * FaceBoundary, double * EdgeBoundary){

        int k = 0;
        int kk = 0;
        for (FaceIterator face(mesh); !face.end(); ++face){
            if (face->index() == FaceBoundary[k]) {
                k=k+1;
                for (EdgeIterator edge(*face); !edge.end(); ++edge){
                    EdgeBoundary[kk] = edge->index();
                    kk = kk+1;
                }
            }
        }
    }

    void FaceToEdgeBoundary(const Mesh& mesh, double * FaceBoundary, int N, double * EdgeBoundary){

        int k = 0;
        int kk = 0;
        Face* face;
        for (int i = 0; i < N; ++i) {
            face = new Face(mesh, FaceBoundary[i]);
            for (EdgeIterator edge(*face); !edge.end(); ++edge){
                EdgeBoundary[kk] = edge->index();
                kk = kk+1;
            }
        }
    }


   void ProlongationGrad(const Mesh& mesh, double * NodalBoundary, double * dataX, double * dataY, double * dataZ, double * data, double * row, double * column ) {
        const int m = mesh.num_vertices();
        const int dim = mesh.geometry().dim();

        std::vector<double> tangent;
        std::vector<double> Ntangent;
        tangent.reserve(dim);
        Ntangent.reserve(dim);

        Edge* dolfin_edge;
        Vertex* dolfin_vertex;
        Vertex* dolfin_vertex0;
        Vertex* dolfin_vertex1;
        int kk = 0;
        int k = -1;



        // for (int i = 0; i < m; ++i) {

        for (VertexIterator vertex(mesh); !vertex.end(); ++vertex){
            if (vertex->index() == NodalBoundary[kk]) {
                 // std::cout << "Boundary Edge found " << std::endl;
                 kk = kk+1;
            }
            else {

                // dolfin_vertex = new Vertex(mesh, i);
                // const uint* dolfin_v = vertex->entities(0);
                // int N = sizeof(dolfin_v);
                // std::cout << vertex->index() << std::endl;
                for (EdgeIterator dolfin_edge(*vertex); !dolfin_edge.end(); ++dolfin_edge){
                    // std::cout << k << std::endl;
                    k = k + 1;

                // }
                // for (int j = 0; j < N; ++j) {
                // for (VertexIterator vertex(mesh); !vertex.end(); ++vertex){
                    // std::cout << dolfin_v[j] << std::endl;
                    // dolfin_edge = new Edge(mesh, dolfin_v[j]);
                    const uint* edgeVERTICES = dolfin_edge->entities(0);
                    dolfin_vertex0 = new Vertex(mesh, edgeVERTICES[0]);
                    dolfin_vertex1 = new Vertex(mesh,edgeVERTICES[1]);
                    tangent[0] = (dolfin_vertex1->x(0)-dolfin_vertex0->x(0));
                    tangent[1] = (dolfin_vertex1->x(1)-dolfin_vertex0->x(1));
                    if (dim == 3) {
                        tangent[2] = dolfin_vertex1->x(2)-dolfin_vertex0->x(2);
                    }

                    for (int j = 0; j < dim; ++j) {
                        if (dim == 3) {
                            if (tangent[0] == 0 && tangent[1] == 0 && tangent[2] == 0){
                                Ntangent[j] =0;
                            }
                            else {
                                Ntangent[j] = tangent[j]/(sqrt(pow(tangent[0],2.0)+pow(tangent[1],2.0)+pow(tangent[2],2.0)));
                            }
                        }
                        else {

                            if (tangent[0] == 0 && tangent[1] == 0){
                                Ntangent[j] =0;
                            }
                            else {
                                Ntangent[j] = tangent[j]/(sqrt(pow(tangent[0],2.0)+pow(tangent[1],2.0)));
                            }
                        }
                    }


                    // std::cout << len << " , " << dolfin_edge->length() << std::endl;
                    dataX[k]=0.5*dolfin_edge->length()*Ntangent[0];
                    dataY[k]=0.5*dolfin_edge->length()*Ntangent[1];

                    if (dim == 3) {
                        dataZ[k]=0.5*dolfin_edge->length()*Ntangent[2];
                    }
                    data[k] = 1;
                    row[k] = dolfin_edge->index();
                    column[k] = edgeVERTICES[1];
                    // std::cout << Ntangent[0] << std::endl;
                    k = k + 1;
                    dataX[k]=0.5*dolfin_edge->length()*Ntangent[0];
                    dataY[k]=0.5*dolfin_edge->length()*Ntangent[1];

                    if (dim == 3) {
                        dataZ[k]=0.5*dolfin_edge->length()*Ntangent[2];
                    }
                    data[k] = -1;
                    row[k] = dolfin_edge->index();
                    column[k] = edgeVERTICES[0];
                    // std::cout << k<< std::endl;
                }
            }

        }



    }


    // void COO_to_PETSc(Mat * A, const double * data, const double * row, const double * column, int n, int m, int N) {
    //     // Mat A;
    //     // MatCreate(MPI_COMM_SELF,&A);
    //     // MatSetSizes(A,n,m,n,m);

    //     for (int i = 0; i < N; ++i) {
    //         // for (int j = 0; j < m; ++j) {
    //             MatSetValue(A, row[i], column[i],  data[i], ADD_VALUES);
    //                 // MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);
    //             // }
    //     }

    //     // MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    //     // MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
    //     // return A;
    // }

}
