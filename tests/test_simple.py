
import py_gelnet
import numpy as np
#from numpy.random import randint, normal

X = np.matrix( np.random.normal(size=(20,50)) )
y = np.random.normal(size=20)


o = np.random.randint(0, 2, [50,50])
A = np.matrix(o & o.transpose()).astype(float) ## Make the matrix symmetric

L = py_gelnet.adj2lapl(A)

model = py_gelnet.gelnet( X, y, 0.1, 1, P = L )

print model
