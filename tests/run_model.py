#!/usr/bin/env python

import sys
import pandas
import py_gelnet
import numpy as np

with open(sys.argv[1] + ".matrix") as handle:
    A = pandas.read_csv(handle, sep="\t", index_col=0)
L = py_gelnet.adj2nlapl(A.as_matrix())

with open(sys.argv[1] + ".features") as handle:
    X = pandas.read_csv(handle, sep="\t", index_col=0)

with open(sys.argv[1] + ".labels") as handle:
    y = pandas.read_csv(handle, sep="\t", index_col=0)

X_in = X.as_matrix().copy(order="fortran")
y_in = y.as_matrix().reshape(y.shape[0])

model = py_gelnet.gelnet( X_in, y_in, 0.1, 1, P = L, max_iter=100 )
print model
