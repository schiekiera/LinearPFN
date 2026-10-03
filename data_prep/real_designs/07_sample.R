#!/usr/bin/env Rscript
# 07_sample.R -- the seeded stratified draw.
#
# Does  : bins every eligible dataset by realized p (3-5, 6-10, 11-20, 21-30;
#         p > 30 -> seeded 30-column subsample, top bin) and realized n
#         (30-199, 200-499, 500-999, 1000-1999; n > 1999 -> seeded row
#         subsample, top bin).  Per cell: one seeded permutation, walked in
#         order, admit or reject (logged), stop at quota_per_cell.  Cells with
#         fewer eligible datasets than the quota are census cells (recorded).
#         Writes the delivered matrices (raw values, not standardized).
# Reads : work/06_deduped.csv, work/03_descriptions.csv, data_raw/csv/**
# Writes: data_clean/<package>__<item_stem>.csv, selected_datasets.csv,
#         work/07_candidates.csv, work/07_draw_log.csv, work/07_cells.csv,
#         work/07_not_drawn.csv, work/07_exclusions.csv
source("00_config.R")
deduped  <- read_work(file.path(CFG$dir_work, "06_deduped.csv"))
licenses <- load_licenses(CFG)

stale <- list.files(CFG$dir_clean, pattern = "[.]csv$", full.names = TRUE)
if (length(stale)) { log_msg("removing %d previously delivered files from %s/", length(stale), CFG$dir_clean); unlink(stale) }

res <- stage_sample(deduped, licenses, CFG, out_dir = ".")
write_csv_atomic(res$selected,   CFG$file_selected)
write_csv_atomic(res$candidates, file.path(CFG$dir_work, "07_candidates.csv"))
write_csv_atomic(res$draw_log,   file.path(CFG$dir_work, "07_draw_log.csv"))
write_csv_atomic(res$cells,      file.path(CFG$dir_work, "07_cells.csv"))
write_csv_atomic(res$not_drawn,  file.path(CFG$dir_work, "07_not_drawn.csv"))
write_csv_atomic(res$exclusions, file.path(CFG$dir_work, "07_exclusions.csv"))
log_msg("selected %d datasets (%d census cells, %d rejections during the walk)",
        nrow(res$selected), sum(res$cells$census), nrow(res$exclusions))
print(res$cells)
