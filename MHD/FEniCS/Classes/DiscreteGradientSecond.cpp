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




}
