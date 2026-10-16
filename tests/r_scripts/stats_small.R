# References for RNG, sample(), t.test, p.adjust, phyper, quantile, mad,
# matrixTests::row_wilcoxon_twosample and write.table number formatting.
# Usage: Rscript stats_small.R <outdir>
suppressPackageStartupMessages(library(matrixTests))
out <- commandArgs(TRUE)[1]; dir.create(out, showWarnings = FALSE, recursive = TRUE)
hexv <- function(x, f) writeLines(sprintf("%a", x), file.path(out, f))
set.seed(42); hexv(runif(1000), "runif_42.txt")
set.seed(1234); writeLines(as.character(sample(1:5000, 2500)), file.path(out, "sample_1234.txt"))
set.seed(1234); writeLines(as.character(unlist(lapply(c(37, 600, 1200), function(n) sample(1:n, min(500, n))))), file.path(out, "sample_seq.txt"))
set.seed(101); m <- matrix(rnorm(60), 6); hexv(as.vector(m), "perm_in.txt"); set.seed(101); hexv(as.vector(t(apply(m, 1, sample))), "perm_out.txt")
set.seed(5); x <- matrix(runif(40 * 30), 40); hexv(as.vector(x), "tt_x.txt")
p <- sapply(1:30, function(j) t.test(x[1:12, j], x[13:40, j])$p.value); hexv(p, "tt_p.txt")
hexv(p.adjust(c(p, NA, p[1:5]), "BH"), "bh.txt")
hexv(1 - phyper(0:20, 25, 75, 20), "phyper.txt")
set.seed(9); y <- rexp(101); hexv(y, "q_x.txt"); hexv(quantile(y, seq(0.1, 1, 0.05), names = FALSE), "q_out.txt"); hexv(c(median(y), mad(y), median(y[-1]), mad(y[-1])), "medmad.txt")
set.seed(3); a <- matrix(round(rnorm(50 * 130), 1), 50); hexv(as.vector(a), "wx_in.txt")
hexv(row_wilcoxon_twosample(a[, 1:60], a[, 61:130])$pvalue, "wx_p.txt")
set.seed(8); v <- c(0.2527597964472405, 1e-300, 123456789012345678, 1/3, 2/3, 1e5, 1e15, 1e-5, 0.1+0.2, 100, 1234.5, -0.000123, exp(rnorm(300, 0, 20)))
hexv(v, "fmt_in.txt"); write.table(data.frame(v = v), file.path(out, "fmt_out.txt"), sep = "\t", row.names = FALSE, col.names = FALSE)
