# SuSiE on the expanded design (L=20, estimate_prior_variance).
# Caveat carried into the report: SuSiE has no heredity notion and treats
# interaction columns as ordinary correlated predictors — its PIPs target a
# different prior and are benchmarked as baseline PIP quality only.
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
Z <- as.matrix(read.csv(file.path(dir, "Z.csv"), header = FALSE))
y <- scan(file.path(dir, "y.csv"), quiet = TRUE)
suppressMessages(library(susieR))
t0 <- proc.time()[["elapsed"]]
fit <- susie(Z, y, L = 20, estimate_prior_variance = TRUE)
secs <- proc.time()[["elapsed"]] - t0
pip <- susie_get_pip(fit)
b <- coef(fit)  # intercept first, then d-1 effects
res <- data.frame(selected = as.integer(pip > 0.5), score = pip,
                  prob = pip, coef = b[-1])
write.csv(res, file.path(dir, "result.csv"), row.names = FALSE)
writeLines(format(b[1], digits = 12), file.path(dir, "intercept.txt"))
writeLines(format(secs, digits = 10), file.path(dir, "timing.txt"))
writeLines(paste0("susieR ", as.character(packageVersion("susieR"))),
           file.path(dir, "version.txt"))
