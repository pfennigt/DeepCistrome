if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(c("GenomicFeatures", "AnnotationDbi"))
BiocManager::install("motifStack")
install.packages('rhdf5')
install.packages("Cairo")

library(motifStack)
library(rhdf5)
library(MotifDb)
library(ade4)
library(RColorBrewer)

tfs <- read.csv('data/WRKY_tnt_tfs.csv', col.names = 'TF')
pred_motifs <- h5read('data/WRKY_tnt_predictive_motifs.h5', name = "motifs")

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
leaves <-names(phylog$leaves)
motifs <- motifs[leaves]
motifSig <- motifSignature(motifs, phylog, cutoffPval = 0.0001, min.freq=1)

sig <- signatures(motifSig)
gpCol <- sigColor(motifSig)

color <- brewer.pal(9, "Set1")

svg(filename = 'results/Figures/WRKY_family_motifs_clustered_tree.svg')
motifPiles(phylog=phylog, pfms=motifs, pfms2=sig,
           col.tree=rep(color, each=5),
           col.leaves=rep(rev(color), each=5),
           col.pfms2=gpCol,
           r.anno=c(0.02, 0.03, 0.04),
           col.anno=list(sample(colors(), 50),
                         sample(colors(), 50),
                         sample(colors(), 50)),
           motifScale="logarithmic",
           plotIndex=TRUE)
dev.off()

