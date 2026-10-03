# hierNet: lasso for hierarchical interactions (Bien, Taylor, Tibshirani),
# strong hierarchy. Takes raw standardized mains; interactions built
# internally. Selection at the CV-min lambda of hierNet.cv over the
# hierNet.path grid; timing includes path + CV (that IS the method's cost).
# Tier-1 score = path entry index. Known slow at large p — the runner may
# cap its p (printed in the report, never silent).
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
X <- as.matrix(read.csv(file.path(dir, "X.csv"), header = FALSE))
Z <- as.matrix(read.csv(file.path(dir, "Z.csv"), header = FALSE))
y <- scan(file.path(dir, "y.csv"), quiet = TRUE)
pairs <- as.matrix(read.csv(file.path(dir, "pairs.csv"), header = FALSE))
p <- ncol(X)
suppressMessages(library(hierNet))
t0 <- proc.time()[["elapsed"]]
path <- hierNet.path(X, y, strong = TRUE, diagonal = FALSE, trace = 0)
# hierNet.cv's internal fold split is balanced (no glinternet-style crash)
# but unseeded; seed it so lamhat is reproducible across runs.
set.seed(1)
cvfit <- hierNet.cv(path, X, y, nfolds = 10, trace = 0)
secs <- proc.time()[["elapsed"]] - t0
nlam <- length(path$lamlist)
l <- which.min(abs(path$lamlist - cvfit$lamhat))
d1 <- p + nrow(pairs)
sel <- numeric(d1); entry <- numeric(d1); cfv <- numeric(d1)
main_path <- path$bp - path$bn  # (p, nlam) — internal standardized scale
for (j in seq_len(p)) {
  nz <- which(abs(main_path[j, ]) > 1e-10)
  if (length(nz)) entry[j] <- nlam - min(nz) + 1
  if (abs(main_path[j, l]) > 1e-10) sel[j] <- 1
}
for (t in seq_len(nrow(pairs))) {
  i <- pairs[t, 1]; j <- pairs[t, 2]
  th_ij <- (path$th[i, j, ] + path$th[j, i, ]) / 2
  nz <- which(abs(th_ij) > 1e-10)
  if (length(nz)) entry[p + t] <- nlam - min(nz) + 1
  if (abs(th_ij[l]) > 1e-10) sel[p + t] <- 1
}
# hierNet's bp/bn/th live on its internal standardized scales and it keeps
# no explicit intercept; but its fitted surface is exactly linear in the
# active effects, so regressing predict() output on [1, Z_active] recovers
# the implied coefficients on the contract scale exactly (n >= |active|+1).
yh <- predict(path, newx = X)[, l]
intercept <- mean(yh)
act <- which(sel > 0)
if (length(act) && length(act) + 1 <= nrow(Z)) {
  fitlm <- lm.fit(cbind(1, Z[, act, drop = FALSE]), yh)
  intercept <- fitlm$coefficients[1]
  cfv[act] <- fitlm$coefficients[-1]
}
res <- data.frame(selected = sel, score = entry, prob = NA, coef = cfv)
write.csv(res, file.path(dir, "result.csv"), row.names = FALSE)
writeLines(format(intercept, digits = 12), file.path(dir, "intercept.txt"))
writeLines(format(secs, digits = 10), file.path(dir, "timing.txt"))
writeLines(paste0("hierNet ", as.character(packageVersion("hierNet"))),
           file.path(dir, "version.txt"))
