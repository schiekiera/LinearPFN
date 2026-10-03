#!/usr/bin/env Rscript
# 05_tag.R -- tag the row unit of every admitted dataset and drop pure time series.
#
# Does  : from the RAW table: 'panel' if a unit-like column (name heuristic)
#         has repeated values AND a time-like column with >= 2 values exists;
#         'time' if a time-like column is strictly monotone (one row per
#         period); else 'cross-section' (incl. clustered data without time).  Domain comes
#         from the frozen labels (package domain for core, inferred for mixed).
#         Pure time series are excluded; panel and aggregate data stay, tagged.
#         Tags are for post-hoc slicing only -- sampling never reads them.
# Reads : work/04_realized.csv, data_raw/csv/**
# Writes: work/05_tagged.csv, work/05_exclusions.csv
source("00_config.R")
realized <- read_work(file.path(CFG$dir_work, "04_realized.csv"))
res <- stage_tag(realized, CFG)
write_csv_atomic(res$tagged,     file.path(CFG$dir_work, "05_tagged.csv"))
write_csv_atomic(res$exclusions, file.path(CFG$dir_work, "05_exclusions.csv"))
log_msg("tagged %d datasets; excluded %d pure time series", nrow(res$tagged) + nrow(res$exclusions), nrow(res$exclusions))
print(table(row_unit = res$tagged$row_unit))
