
import py_gelnet
import numpy as np
#from numpy.random import randint, normal

X = np.matrix( np.random.normal(size=(20,50)) )

model = py_gelnet.gelnet( X, None, 0, 1 )

print(model)
