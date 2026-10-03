#!/usr/bin/env Rscript

source("00_config.R")
tagged <- read_work(file.path(CFG$dir_work, "05_tagged.csv"))
res <- stage_dedup(tagged)
write_csv_atomic(res$deduped, file.path(CFG$dir_work, "06_deduped.csv"))
write_csv_atomic(res$exclusions, file.path(CFG$dir_work, "06_exclusions.csv"))
log_msg("dedup: %d kept, %d removed", nrow(res$deduped), nrow(res$exclusions))
