import pandas as pd
import numpy as np
print "\n\n   Lagrange convergence"
LagrangeTitles = ["Total DoF","R DoF","Soln Time","R-L2","R-order","R-H1","H1-order"]
LagrangeValues = np.concatenate((np.random.randn(5, 1) ,np.random.randn(5, 1),np.random.randn(5, 1),np.random.randn(5, 1),np.random.randn(5, 1),np.random.randn(5, 1),np.random.randn(5, 1)),axis=1)
LagrangeTable= pd.DataFrame(LagrangeValues, columns = LagrangeTitles)
pd.set_option('precision',3)
print LagrangeTable