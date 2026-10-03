#!/usr/bin/env Rscript
# run_all.R -- run the whole pipeline in order, each script in a fresh R
# session, stopping at the first failure.  Usage:  Rscript run_all.R
if (!file.exists("00_config.R")) stop("run from inside real_designs/")
scripts <- setdiff(sort(list.files(".", pattern = "^0[0-9]_.*[.]R$"), method = "radix"), "00_config.R")
for (s in scripts) {
  cat("\n==================== ", s, " ====================\n", sep = "")
  status <- system2("Rscript", c("--vanilla", s))
  if (status != 0) {
    cat("\n", s, " failed with status ", status, "\n", sep = "")
    quit(status = 1)
  }
}
cat("\nall stages completed\n")
