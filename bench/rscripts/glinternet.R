# glinternet: hierarchical group lasso with strong hierarchy (Lim & Hastie).
# Takes the raw standardized mains and builds interactions internally; the
# selected set at the CV-min lambda (lambdaHat) is mapped back to canonical
# pair order. Tier-1 score = path entry (largest lambda index at which the
# effect is active; earlier entry = larger score). Coefficients are the
# unconstrained-parameterization values coef() returns, not a refit.
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
X <- as.matrix(read.csv(file.path(dir, "X.csv"), header = FALSE))
y <- scan(file.path(dir, "y.csv"), quiet = TRUE)
pairs <- as.matrix(read.csv(file.path(dir, "pairs.csv"), header = FALSE))
p <- ncol(X)
suppressMessages(library(glinternet))
# glinternet.cv assigns folds via sample(rep(1:nFolds, ceiling(n/nFolds)), n)
# — when nFolds does not divide n, a fold can get <= 1 test rows and the
# package crashes in apply(YtestHat, 2, ...) ("dim(X) must have a positive
# length"). The folds default is glinternet.cv's first RNG draw (verified
# seed-by-seed against a simulation of that expression), so seeding with the
# smallest seed whose simulated draw keeps every fold >= 2 both avoids the
# crash and makes the CV folds deterministic.
n <- length(y)
nfolds <- 10
cv_seed <- NA
for (s in 1:10000) {
  set.seed(s)
  f <- sample(rep(1:nfolds, ceiling(n / nfolds)), n, replace = FALSE)
  if (min(tabulate(f, nfolds)) >= 2) { cv_seed <- s; break }
}
if (is.na(cv_seed)) stop("no fold seed keeps every CV fold >= 2 rows")
t0 <- proc.time()[["elapsed"]]
set.seed(cv_seed)
cv <- glinternet.cv(X, y, numLevels = rep(1, p), nFolds = nfolds)
secs <- proc.time()[["elapsed"]] - t0
fit <- cv$glinternetFit
nlam <- length(fit$lambda)
d1 <- p + nrow(pairs)
entry <- numeric(d1)  # 0 = never active on the path
pair_key <- paste(pairs[, 1], pairs[, 2])
for (l in seq_len(nlam)) {
  act <- fit$activeSet[[l]]
  if (!is.null(act$cont)) {
    for (j in act$cont) if (entry[j] == 0) entry[j] <- nlam - l + 1
  }
  cc <- act$contcont
  if (!is.null(cc) && length(cc)) {
    cc <- matrix(cc, ncol = 2)
    for (r in seq_len(nrow(cc))) {
      t_idx <- match(paste(min(cc[r, ]), max(cc[r, ])), pair_key)
      if (!is.na(t_idx) && entry[p + t_idx] == 0) entry[p + t_idx] <- nlam - l + 1
    }
  }
}
lhat <- which.min(abs(fit$lambda - cv$lambdaHat))
cfs <- coef(fit, lambdaIndex = lhat)[[1]]
sel <- numeric(d1); cfv <- numeric(d1)
if (!is.null(cfs$mainEffects$cont)) {
  for (k in seq_along(cfs$mainEffects$cont)) {
    j <- cfs$mainEffects$cont[k]
    sel[j] <- 1
    cfv[j] <- cfs$mainEffectsCoef$cont[[k]][1]
  }
}
cc <- cfs$interactions$contcont
if (!is.null(cc) && length(cc)) {
  cc <- matrix(cc, ncol = 2)
  for (r in seq_len(nrow(cc))) {
    t_idx <- match(paste(min(cc[r, ]), max(cc[r, ])), pair_key)
    if (!is.na(t_idx)) {
      sel[p + t_idx] <- 1
      cfv[p + t_idx] <- cfs$interactionsCoef$contcont[[r]][1]
    }
  }
}
res <- data.frame(selected = sel, score = entry, prob = NA, coef = cfv)
write.csv(res, file.path(dir, "result.csv"), row.names = FALSE)
intercept <- fit$betahat[[lhat]][1]
writeLines(format(intercept, digits = 12), file.path(dir, "intercept.txt"))
writeLines(format(secs, digits = 10), file.path(dir, "timing.txt"))
writeLines(paste0("glinternet ", as.character(packageVersion("glinternet"))),
           file.path(dir, "version.txt"))
