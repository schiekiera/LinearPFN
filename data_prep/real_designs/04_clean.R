#!/usr/bin/env Rscript

source("00_config.R")
frame <- read_work(file.path(CFG$dir_work, "02_frame.csv"))
res <- stage_clean(frame, CFG)
write_csv_atomic(res$realized, file.path(CFG$dir_work, "04_realized.csv"))
write_csv_atomic(res$columns, file.path(CFG$dir_work, "04_column_log.csv"))
write_csv_atomic(res$exclusions, file.path(CFG$dir_work, "04_exclusions.csv"))
log_msg("cleaned %d datasets; admitted %d", nrow(res$realized), sum(res$realized$exclusion_reason == ""))
print(table(exclusion_reason = res$exclusions$reason))
print(table(column_decision = res$columns$reason))
