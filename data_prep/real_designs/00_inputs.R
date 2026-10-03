#!/usr/bin/env Rscript

source("00_config.R")

## 1. package labels + include -------------------------------------------------
pl_path <- file.path(CFG$dir_inputs, "package_labels.csv")
if (!file.exists(pl_path)) stop("hand-curated input missing: ", pl_path)
pl <- read_package_labels(pl_path)
rule <- include_rule(pl, CFG)
if ("include" %in% names(pl)) {
  bad <- which(pl$include != rule)
  if (length(bad)) {
    stop(
      "inputs/package_labels.csv: include disagrees with the rule for ",
      paste(pl$package[bad], collapse = ", "),
      "; change CFG$inputs_include_exceptions, not the file"
    )
  }
  log_msg(
    "include column verified against the rule: %d of %d packages included",
    sum(rule == "include"), nrow(pl)
  )
} else {
  pl$include <- rule
  write_csv_atomic(pl, pl_path)
  log_msg(
    "include column added to %s: %d of %d packages included", pl_path,
    sum(rule == "include"), nrow(pl)
  )
}
exc <- CFG$inputs_include_exceptions
for (i in seq_len(nrow(exc))) log_msg("  exception: %s -> %s (%s)", exc$package[i], exc$include[i], exc$reason[i])

## 2. index snapshot -----------------------------------------------------------
idx <- file.path(CFG$dir_index, "datasets.csv")
idx_sha <- paste0(idx, ".sha256")
idx_date <- file.path(CFG$dir_index, "fetch_date.txt")
if (!file.exists(idx)) {
  log_msg("fetching index %s", CFG$index_url)
  if (download_cached(CFG$index_url, idx, CFG) != "downloaded") stop("index download failed")
  writeLines(sha256_file(idx), idx_sha)
  writeLines(format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), idx_date)
  log_msg("index snapshot saved, sha256 %s", readLines(idx_sha))
} else {
  if (!file.exists(idx_sha)) stop("snapshot present without ", idx_sha, "; restore both from git")
  if (sha256_file(idx) != readLines(idx_sha)) stop("frozen index snapshot changed (sha256 mismatch)")
  log_msg("index snapshot present (fetched %s), sha256 verified", readLines(idx_date)[1])
}

## 3 + 4. merge, flags, descriptions ------------------------------------------
res <- build_inputs(read_index(CFG), pl, CFG)
write_csv_atomic(res$datasets, file.path(CFG$dir_work, "00_datasets.csv"))
write_csv_atomic(res$versions, file.path(CFG$dir_work, "00_package_versions.csv"))
cat(res$report, sep = "\n")
