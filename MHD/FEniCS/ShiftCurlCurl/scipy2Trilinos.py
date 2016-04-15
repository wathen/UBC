from PyTrilinos import Epetra
def scipy_csr_matrix2CrsMatrix(sp, comm):
    Ap = sp.indptr
    Aj = sp.indices
    Ax = sp.data
    m = Ap.shape[0]-1
    aMap = Epetra.Map(m, 0, comm)
    # range Map
    arMap=Epetra.Map(sp.shape[0], 0, comm)
    # domain Map
    adMap=Epetra.Map(sp.shape[1], 0, comm)
    aGraph = Epetra.CrsGraph( Epetra.Copy, aMap, 0)
    for ii in range(aMap.NumGlobalElements()):
          i = aMap.GID(ii)
          indy = range(Ap[i],Ap[i+1])
          if (indy != []): 
              aGraph.InsertGlobalIndices(i, Aj[indy])
    aGraph.FillComplete(adMap, arMap)
    A = Epetra.CrsMatrix(Epetra.Copy, aGraph)
    for ii in range(aMap.NumGlobalElements()):
          i = aMap.GID(ii)
          indy = range(Ap[i],Ap[i+1])
	  if (indy != []): 
	      A.SumIntoGlobalValues(i, Ax[indy], Aj[indy])
    A.FillComplete(adMap, arMap)
    return A	      
