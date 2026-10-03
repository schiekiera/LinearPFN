#!/usr/bin/env Rscript

source("00_config.R")
res <- stage_frame(CFG)
write_csv_atomic(res$frame, file.path(CFG$dir_work, "02_frame.csv"))
write_csv_atomic(res$exclusions, file.path(CFG$dir_work, "02_exclusions.csv"))
log_msg("frame: %d datasets from %d packages", nrow(res$frame), length(unique(res$frame$package)))
print(table(exclusion_reason = res$exclusions$reason))
