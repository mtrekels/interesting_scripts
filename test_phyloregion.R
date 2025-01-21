#Test Phyloregion package

library(phyloregion)

library(ape)
library(Matrix)
library(terra)

data(africa)
sparse_comm <- africa$comm

tree <- africa$phylo