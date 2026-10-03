#!/usr/bin/env Rscript

source("00_config.R")
W <- function(f) file.path(CFG$dir_work, f)

ex <- consolidate_exclusions(CFG$dir_work)
write_csv_atomic(ex, CFG$file_exclusions)
log_msg("exclusions.csv: %d rows", nrow(ex))

writeLines(c(R.version.string, capture.output(sessionInfo())), CFG$file_session)

sel <- read_work(CFG$file_selected)
hashes <- read_work(W("03_hashes.csv"))
desc <- read_work(W("03_descriptions.csv"))
vers <- read_work(W("00_package_versions.csv"))
inputs <- c(file.path(CFG$dir_inputs, "package_labels.csv"), W("00_datasets.csv"), W("00_package_versions.csv"))
row <- function(section, key, value) data.frame(section = section, key = key, value = as.character(value), stringsAsFactors = FALSE)
cf <- cfg_flat(CFG)
man <- rbind(
  row("pipeline", "generated_on", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  row("pipeline", "n_selected", nrow(sel)),
  row("config", cf$key, cf$value),
  row("index", "url", CFG$index_url),
  row("index", "fetch_date", readLines(file.path(CFG$dir_index, "fetch_date.txt"))[1]),
  row("index", "sha256", readLines(file.path(CFG$dir_index, "datasets.csv.sha256"))[1]),
  row("index", "n_rows", length(readLines(file.path(CFG$dir_index, "datasets.csv"))) - 1),
  row("inputs", inputs, vapply(inputs, sha256_file, "")),
  row("package_version", vers$package, vers$version),
  row("raw_csv", hashes$raw_file[hashes$sha256 != ""], hashes$sha256[hashes$sha256 != ""]),
  row("description", desc$desc_file[desc$sha256 != ""], desc$sha256[desc$sha256 != ""]),
  row("clean", sel$clean_file, sel$clean_sha256),
  row("output", CFG$file_selected, sha256_file(CFG$file_selected)),
  row("output", CFG$file_exclusions, sha256_file(CFG$file_exclusions)),
  row("r", "version", R.version.string),
  row("r", "platform", R.version$platform),
  row("r", "locale", Sys.getlocale("LC_COLLATE")),
  row("r", "sessionInfo_sha256", sha256_file(CFG$file_session))
)
write_csv_atomic(man, CFG$file_manifest)
log_msg("MANIFEST.csv: %d rows", nrow(man))

collog <- read_work(W("04_column_log.csv"))
cells <- read_work(W("07_cells.csv"))
md <- c(
  "# Audit sheet: selected datasets",
  "",
  "One block per selected dataset, in cell order. For each, open the doc URL and",
  "check (1) the dataset is real social-science / psychology data (inclusion is",
  "package-level, so off-topic tables inside core packages reach this list);",
  "(2) the admitted columns are genuine numeric or 0/1 variables;",
  "(3) nothing dropped should have been kept. Record verdicts outside this file.",
  "", sprintf("Generated %s from `%s` (%d datasets).", format(Sys.Date()), CFG$file_selected, nrow(sel)), ""
)
for (cl in cell_order(CFG)) {
  s <- sel[sel$cell == cl, ]
  cs <- cells[cells$cell == cl, ]
  md <- c(md, sprintf(
    "## Cell %s  (eligible %d, selected %d%s)", cl, cs$n_eligible, cs$n_admitted,
    if (isTRUE(cs$census)) ", CENSUS" else ""
  ), "")
  for (i in seq_len(nrow(s))) {
    r <- s[i, ]
    dropped <- collog[collog$package == r$package & collog$item == r$item & collog$decision == "dropped", ]
    md <- c(
      md,
      sprintf("### %s/%s -- %s", r$package, r$item, r$title),
      sprintf("- doc: <%s>", r$doc_url),
      sprintf(
        "- delivered %d x %d (realized %d x %d, raw %d rows); tier %s, domain %s, row unit %s, license %s%s%s",
        r$n_delivered, r$p_delivered, r$n_realized, r$p_realized, r$n_raw, r$tier, r$domain, r$row_unit, r$license,
        if (isTRUE(r$row_subsampled)) ", ROW-SUBSAMPLED" else "",
        if (isTRUE(r$col_subsampled)) ", COL-SUBSAMPLED" else ""
      ),
      sprintf("- admitted columns: %s", gsub("|", ", ", r$columns, fixed = TRUE)),
      sprintf("- dropped columns: %s", if (nrow(dropped)) paste0(dropped$column, " (", dropped$reason, ")", collapse = ", ") else "none"),
      ""
    )
  }
}
writeLines(md, CFG$file_audit)
log_msg("audit sheet: %s", CFG$file_audit)
