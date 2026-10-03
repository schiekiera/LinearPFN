#!/usr/bin/env Rscript

source("00_config.R")
CFG$min_minority_rows <- 10L # the pool's extra rule; the stratified draw keeps 0
cands <- read_work(file.path(CFG$dir_work, "07_candidates.csv"))
selected <- read_work(CFG$file_selected)
licenses <- load_licenses(CFG)
cands <- cands[is.na(cands$exclusion_reason) | cands$exclusion_reason == "", ]
out_dir <- "data_clean_pool"
dir_ensure(out_dir)
stale <- list.files(out_dir, pattern = "[.]csv$", full.names = TRUE)
if (length(stale)) {
  log_msg("removing %d previously delivered files from %s/", length(stale), out_dir)
  unlink(stale)
}

rows <- list()
failed <- list()
dropped_cols <- list()
for (i in seq_len(nrow(cands))) {
  row <- cands[i, ]
  key <- paste(row$package, row$item, sep = "/")
  res <- clean_dataset(read_raw_csv(row$raw_file), CFG)
  if (is.null(res$X)) {
    failed[[length(failed) + 1]] <- data.frame(package = row$package, item = row$item, reason = "reclean_failed", detail = res$exclusion$reason)
    next
  }
  d <- deliver_matrix(res$X, key, CFG)
  # bins are re-assigned from the DELIVERED matrix (the near-constant rule may
  # lower p); a design below the size minima leaves the pool with a logged reason
  p_bin <- bin_label(ncol(d$X), CFG$p_bins)
  n_bin <- bin_label(nrow(d$X), CFG$n_bins)
  if (is.na(p_bin) || is.na(n_bin) || ncol(d$X) < CFG$min_p || nrow(d$X) < CFG$min_n) {
    failed[[length(failed) + 1]] <- data.frame(
      package = row$package, item = row$item, reason = "below_minimum_after_rule",
      detail = sprintf("n=%d p=%d", nrow(d$X), ncol(d$X))
    )
    next
  }
  dropped_cols[[length(dropped_cols) + 1]] <- data.frame(
    package = row$package, item = row$item,
    n_dropped = sum(res$columns$reason == "near_constant") + length(d$dropped_constant),
    columns = paste(c(res$columns$column[res$columns$reason == "near_constant"], d$dropped_constant), collapse = "|")
  )
  fname <- clean_name(row$package, row$item_stem)
  fpath <- file.path(out_dir, fname)
  write_csv_atomic(d$X, fpath)
  lic <- licenses[row$package]
  if (is.null(lic) || is.na(lic) || lic == "") lic <- "unknown"
  sel_key <- paste(selected$package, selected$item, sep = "/")
  rows[[length(rows) + 1]] <- data.frame(
    package = row$package, item = row$item, title = row$title, csv_url = row$csv_url,
    doc_url = row$doc_url, license = unname(lic), tier = row$tier, domain = row$domain,
    row_unit = row$row_unit, n_index = row$rows_index, p_index = row$p_model_index,
    n_raw = row$n_raw, n_realized = row$n_realized, p_realized = row$p_realized,
    n_delivered = nrow(d$X), p_delivered = ncol(d$X), max_abs_r = round(max_abs_cor_of(d$X), 6),
    n_bin = n_bin, p_bin = p_bin, cell = cell_id(p_bin, n_bin),
    row_subsampled = d$row_subsampled, col_subsampled = d$col_subsampled,
    columns = paste(names(d$X), collapse = "|"),
    clean_file = file.path(out_dir, fname), clean_sha256 = sha256_file(fpath),
    selected = key %in% sel_key, stringsAsFactors = FALSE
  )
}
elig <- rbind_list(rows)
elig <- elig[order_c(match(elig$cell, cell_order(CFG)), elig$package, elig$item), ]
rownames(elig) <- NULL
write_csv_atomic(elig, "eligible_designs.csv")
dc <- rbind_list(dropped_cols)
dc <- dc[dc$n_dropped > 0, ]
write_csv_atomic(dc, file.path(CFG$dir_work, "10_near_constant_columns.csv"))
write_csv_atomic(
  if (length(failed)) rbind_list(failed) else data.frame(package = character(0), item = character(0), reason = character(0), detail = character(0)),
  file.path(CFG$dir_work, "10_excluded.csv")
)
log_msg("near-constant rule (min_minority_rows = %d): %d designs lost columns, %d designs excluded", CFG$min_minority_rows, nrow(dc), length(failed))
# byte-for-byte check against the stratified draw's delivered files
m <- match(paste(selected$package, selected$item, sep = "/"), paste(elig$package, elig$item, sep = "/"))
same <- selected$clean_sha256 == elig$clean_sha256[m]
log_msg(
  "delivered %d designs (%d failed); %d of %d selected designs reproduced byte for byte",
  nrow(elig), length(failed), sum(same, na.rm = TRUE), nrow(selected)
)
log_msg("(the near-constant rule is OFF for the stratified draw, so a selected design that lost a column differs here by design)")
