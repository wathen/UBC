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


namespace dolfin
{
    void gradient(const FunctionSpace& V, const Array<int>& Mapping, double * column, double * row, double * data) {

        const Mesh mesh = *V.mesh();
        const GenericDofMap& dofmap_u = *V.dofmap();
        const int n = mesh.num_cells();
        const int m = mesh.num_edges();

        std::vector<int> edge2dof;
        edge2dof.reserve(m);
        // std::vector<double> XcoordNormal;
        // std::vector<double> YcoordNormal;
        // std::vector<double> ZcoordNormal;
        // XcoordNormal.reserve(m);
        // YcoordNormal.reserve(m);
        // ZcoordNormal.reserve(m);


        Cell* dolfin_cell;
        // std::cout << std::endl;
        for (int i = 0; i < n; ++i) {
            dolfin_cell = new Cell(mesh, i);
            std::vector<int> cellDOF = dofmap_u.cell_dofs(i);
            const uint* edgeVALUES = dolfin_cell->entities(1);
            for (int j = 0; j < cellDOF.size(); ++j) {
                edge2dof[edgeVALUES[j]]=cellDOF[j];
                // XcoordNormal[edgeVALUES[j]] =  dolfin_cell->normal(j,0);
                // YcoordNormal[edgeVALUES[j]] =  dolfin_cell->normal(j,1);
                // // std::cout << XcoordNormal[edgeVALUES[j]] << std::endl;
                // if (cellDOF.size() != 3) {
                //     ZcoordNormal[edgeVALUES[j]] =  dolfin_cell->normal(j,2);
                // }


            }

        }


        Edge* dolfin_edge;
        int k = -1;
        for (int i = 0; i < m; ++i)
        {
            dolfin_edge = new Edge(mesh, i);
            const uint* edgeVERTICES = dolfin_edge->entities(0);

            k = k+1;
            column[k]=edge2dof[dolfin_edge->index()];
            row[k]=Mapping[edgeVERTICES[0]];
            data[k]=-1;
            // if (cellDOF.size() == 3) {
            // Pcolumn[k] = edge2dof[dolfin_edge->index()];
            // Prow[k] = Mapping[edgeVERTICES[0]];
            // PdataX[k] = -0.5*dolfin_edge->length()*YcoordNormal[dolfin_edge->index()];
            // Pcolumn[k] = edge2dof[dolfin_edge->index()];
            // Prow[k] = Mapping[edgeVERTICES[0]];
            // PdataY[k] = 0.5*dolfin_edge->length()*XcoordNormal[dolfin_edge->index()];
            // }
            // else{
            //     std::cout << "3d" << std::endl;
            // }
            k = k+1;
            column[k]=edge2dof[dolfin_edge->index()];
            row[k]=Mapping[edgeVERTICES[1]];
            data[k]=1;

            // Pcolumn[k] = edge2dof[dolfin_edge->index()];
            // Prow[k] = Mapping[edgeVERTICES[0]];
            // PdataX[k] =  -0.5*dolfin_edge->length()*YcoordNormal[dolfin_edge->index()];
            // Pcolumn[k] = edge2dof[dolfin_edge->index()];
            // Prow[k] = Mapping[edgeVERTICES[0]];
            // PdataY[k] = 0.5*dolfin_edge->length()*XcoordNormal[dolfin_edge->index()];

        }

    }

}
