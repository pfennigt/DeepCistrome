library(motifStack)
library(rhdf5)
library(MotifDb)
library(ade4)
library(RColorBrewer)


tfs <- read.csv('data/GBox_tfs.csv', col.names = 'TF')
pred_motifs <- h5read('data/GBox_predictive_motifs.h5', name = "motifs")

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
selcols <- rep(color, each=12)
svg(filename = 'results/Figures/motifs_radial_tree_gbox.svg')
plotMotifStackWithRadialPhylog(phylog=phylog, pfms=sig,
                               circle=6.8,
                               cleaves = 0.3,
                               clabel.leaves = 0.35,
                               col.bg=selcols, col.bg.alpha=0.3,
                               col.leaves=selcols,
                               col.inner.label.circle=gpCol,
                               inner.label.circle.width=0.03,
                               angle=350, circle.motif=11.2,
                               motifScale="logarithmic")
dev.off()

