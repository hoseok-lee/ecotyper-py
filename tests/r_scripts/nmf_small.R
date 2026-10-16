# Generates exact (hex) references for NMF unit parity tests.
# Usage: Rscript nmf_small.R <outdir>
suppressPackageStartupMessages(library(NMF))
out <- commandArgs(TRUE)[1]; dir.create(out, showWarnings = FALSE, recursive = TRUE)
hexw <- function(m, f) write.table(matrix(sprintf("%a", m), nrow(m)), file.path(out, f), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
set.seed(7)
V <- matrix(rnorm(40 * 15), 40, 15)
hexw(V, "V_raw.txt")
X <- posneg(V)
hexw(X, "X.txt")
set.seed(123); init <- rnmf(4, X); hexw(init@W, "init_W.txt"); hexw(init@H, "init_H.txt")
for (mi in c(1, 2, 10, 11, 25)) {
  fit <- nmf(X, 4, method = "brunet", seed = 123, .opt = "P1", maxIter = mi)
  hexw(fit@fit@W, sprintf("W_it%d.txt", mi)); hexw(fit@fit@H, sprintf("H_it%d.txt", mi))
}
fit <- nmf(X, 4, method = "brunet", seed = 123, .opt = "P1")
hexw(fit@fit@W, "W_full.txt"); hexw(fit@fit@H, "H_full.txt")
writeLines(c(sprintf("%d", niter(fit)), sprintf("%a", deviance(fit))), file.path(out, "full_info.txt"))
# a larger, discovery-like run
set.seed(11)
V2 <- matrix(rnorm(300 * 60), 300, 60) + outer(rep(1, 300), rep(c(0, 2, -2), each = 20))
X2 <- posneg(t(scale(t(V2))))
hexw(X2, "X2.txt")
for (s in 1235:1237) {
  fit <- nmf(X2, 5, method = "brunet", seed = s, .opt = "P1")
  hexw(fit@fit@W, sprintf("W2_%d.txt", s)); hexw(fit@fit@H, sprintf("H2_%d.txt", s))
  writeLines(c(sprintf("%d", niter(fit)), sprintf("%a", deviance(fit))), file.path(out, sprintf("info2_%d.txt", s)))
}
