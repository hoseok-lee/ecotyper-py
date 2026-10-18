# Writes a vector of doubles in binary plus R's write.table text for it.
out <- commandArgs(TRUE)[1]
set.seed(42)
x <- c(log2(1 + 0:20000),
       runif(200000) * 10^sample(-12:12, 200000, TRUE),
       rnorm(200000), 1 / (1:20000), (1:20000) * 0.1, 10^(-30:30), -log2(1 + 0:2000) / 7)
writeBin(x, file.path(out, "fmt_x.bin"))
write.table(data.frame(v = x), file.path(out, "fmt_r.txt"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
