#!/usr/bin/env Rscript

source("00_config.R")
frame <- read_work(file.path(CFG$dir_work, "02_frame.csv"))

status <- character(nrow(frame))
dest <- raw_path(frame$package, frame$item_stem, CFG)
for (i in seq_len(nrow(frame))) {
  status[i] <- download_cached(frame$csv_url[i], dest[i], CFG)
  if (i %% 50 == 0) log_msg("%d / %d  (%s)", i, nrow(frame), paste(names(table(status[1:i])), table(status[1:i]), collapse = ", "))
}
hashes <- data.frame(
  package = frame$package, item = frame$item, raw_file = dest, status = status,
  sha256 = ifelse(status == "failed", "", vapply(dest, function(p) if (file.exists(p)) sha256_file(p) else "", "")),
  stringsAsFactors = FALSE
)
write_csv_atomic(hashes, file.path(CFG$dir_work, "03_hashes.csv"))
failed <- status == "failed"
write_csv_atomic(
  excl_rows(
    frame$package[failed], frame$item[failed], "download", "download_failed",
    frame$csv_url[failed]
  ),
  file.path(CFG$dir_work, "03_exclusions.csv")
)
log_msg("csv files: %s", paste(names(table(status)), table(status), collapse = ", "))

pkgs <- sort_c(unique(frame$package))
rows <- list()
for (pkg in pkgs) {
  f <- file.path(CFG$dir_desc, paste0(pkg, ".DESCRIPTION"))
  marker <- paste0(f, ".missing")
  if (file.exists(marker)) {
    st <- "missing"
  } else {
    st <- download_cached(sprintf(CFG$cran_desc_url_tpl, pkg), f, CFG)
    if (st == "failed") {
      dir_ensure(dirname(marker))
      writeLines(format(Sys.Date()), marker)
      st <- "missing"
    }
  }
  lic <- if (file.exists(f)) {
    tryCatch(as.character(read.dcf(f, fields = "License")[1, 1]),
      error = function(e) "unknown"
    )
  } else {
    "unknown"
  }
  if (is.na(lic)) lic <- "unknown"
  rows[[pkg]] <- data.frame(
    package = pkg, status = st, desc_file = if (file.exists(f)) f else "",
    sha256 = if (file.exists(f)) sha256_file(f) else "",
    license = lic, stringsAsFactors = FALSE
  )
}
desc <- do.call(rbind, rows)
rownames(desc) <- NULL
write_csv_atomic(desc, file.path(CFG$dir_work, "03_descriptions.csv"))
log_msg("DESCRIPTION files: %s", paste(names(table(desc$status)), table(desc$status), collapse = ", "))
