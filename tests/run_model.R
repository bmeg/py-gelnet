
library(gelnet)

A <- as.matrix(read.csv("train.matrix", sep="\t", row.names=1))
y <- as.matrix(read.csv("train.labels", sep="\t", row.names=1))
X <- as.matrix(read.csv("train.features", sep="\t", row.names=1))

L <- adj2nlapl(A)

gelnet( X, y, 0.1, 1, P = L, max.iter=100 )