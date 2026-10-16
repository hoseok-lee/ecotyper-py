# References for restart merging, cophenetic correlation, silhouettes, hclust.
suppressPackageStartupMessages({library(NMF); library(cluster)})
out <- commandArgs(TRUE)[1]; dir.create(out, showWarnings = FALSE, recursive = TRUE)
hexw <- function(m, f) { m <- as.matrix(m); write.table(matrix(sprintf("%a", m), nrow(m)), file.path(out, f), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE) }
rd <- function(f) { x <- as.matrix(read.delim(f, header = FALSE)); matrix(as.numeric(x), nrow(x)) }
X2 <- matrix(sapply(as.matrix(read.delim(file.path(out, "..", "nmf_small", "X2.txt"), header = FALSE, colClasses = "character")), function(s) as.numeric(s)), ncol = 60)
res <- list()
for (r in 2:6) {
  l <- lapply(1:5, function(k) nmf(X2, r, method = "brunet", seed = 1234 + k, .opt = "P1"))
  e <- suppressWarnings(NMF:::NMFfitX(l, .merge = TRUE))
  hexw(consensus(e), sprintf("cons_r%d.txt", r))
  sis <- summary(silhouette(e, what = "samples")); sig <- summary(silhouette(e, what = "features"))
  sichc <- summary(silhouette(e, what = "chc")); sicons <- summary(silhouette(e, what = "consensus"))
  res[[length(res) + 1]] <- c(r, cophcor(e), sis[4][[1]], sig[4][[1]], sichc[4][[1]], sicons[4][[1]], dispersion(e), which.min(sapply(l, deviance)))
  hexw(e@fit@W, sprintf("bestW_r%d.txt", r))
  writeLines(sprintf("%s", as.character(predict(e))), file.path(out, sprintf("predict_r%d.txt", r)))
}
m <- do.call(rbind, res); hexw(m, "rank_metrics.txt")
# hclust with many ties (consensus-like) and complete linkage on euclidean distances
set.seed(3); C <- matrix(sample(0:5, 30 * 30, TRUE) / 5, 30); C <- (C + t(C)) / 2; diag(C) <- 1
hc <- hclust(as.dist(1 - C), method = "average")
write.table(cbind(hc$merge, sprintf("%a", hc$height)), file.path(out, "hc_avg.txt"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
writeLines(as.character(hc$order), file.path(out, "hc_avg_order.txt"))
writeLines(as.character(sapply(2:10, function(k) paste(cutree(hc, k), collapse = ","))), file.path(out, "hc_avg_cutree.txt"))
hexw(C, "C_ties.txt")
set.seed(4); A <- matrix(rnorm(50 * 8), 50)
h2 <- hclust(dist(t(A)))
writeLines(as.character(h2$order), file.path(out, "hc_complete_order.txt")); hexw(A, "A.txt")
hexw(as.matrix(dist(t(A))), "distA.txt")
hexw(cor(A), "corA.txt")
s <- silhouette(cutree(hc, 4), as.dist(1 - C)); writeLines(sprintf("%a", summary(s)$avg.width), file.path(out, "sil_ties.txt"))
