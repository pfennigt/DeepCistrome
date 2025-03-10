library(motifStack)
library(rhdf5)
library(MotifDb)
library(ade4)
library(RColorBrewer)

tfs <- read.csv('data/tfs.csv', col.names = 'TF')
pred_motifs <- h5read('data/predictive_motifs.h5', name = "motifs")

motifs <- list()
for (idx in seq(1, dim(tfs)[1])) {
  mat <- pred_motifs[,,idx]
  dim(mat)
  row.names(mat) <- c('A', 'C', 'G', 'T')
  motif <- new("pcm", mat=mat, name=tfs[idx,])
  motifs[[idx]] <- motif
}
names(motifs) <- tfs$TF

hc <- clusterMotifs(motifs)
phylog <- hclust2phylog(hc)
hc$merge
?clusterMotifs

comp <- matalign(motifs, pseudo = 1)
comp
