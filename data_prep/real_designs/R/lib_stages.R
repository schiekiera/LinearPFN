# R/lib_stages.R -- the pipeline stages as functions of (tables, cfg), shared
# by the numbered scripts and by 09_verify.R, which re-runs them in memory.

stage_frame <- function(cfg) build_frame(load_inputs(cfg)$datasets, cfg)

# One row per frame dataset whose raw CSV is cached: cleaning stats,
# admitted column list, column-set hash, exclusion reason ("" if admitted).
stage_clean <- function(frame, cfg, quiet = FALSE) {
  rows <- vector("list", nrow(frame))
  logs <- vector("list", nrow(frame))
  excl <- vector("list", nrow(frame))
  for (i in seq_len(nrow(frame))) {
    fr <- frame[i, ]
    f <- raw_path(fr$package, fr$item_stem, cfg)
    if (!file.exists(f)) next                     # download failure: logged at stage 03
    df <- tryCatch(read_raw_csv(f), error = function(e) e)
    st <- list(n_raw = NA, n_cols_raw = NA, n_admitted_cols = NA, n_complete = NA,
               row_loss_frac = NA, n_realized = NA, p_realized = NA, max_abs_r = NA)
    cols <- character(0)
    if (inherits(df, "error")) {
      reason <- "parse_error"
      detail <- gsub("[\r\n]+", " ", conditionMessage(df))
    } else {
      res <- clean_dataset(df, cfg)
      st <- res$stats
      if (is.null(res$exclusion)) {
        reason <- ""; detail <- ""; cols <- names(res$X)
      } else {
        reason <- res$exclusion$reason; detail <- res$exclusion$detail
      }
      if (nrow(res$columns) > 0) {
        logs[[i]] <- cbind(package = fr$package, item = fr$item, res$columns,
                           stringsAsFactors = FALSE)
      }
    }
    rows[[i]] <- cbind(fr, data.frame(
      raw_file = f, n_raw = st$n_raw, n_cols_raw = st$n_cols_raw,
      n_admitted_cols = st$n_admitted_cols, n_complete = st$n_complete,
      row_loss_frac = st$row_loss_frac, n_realized = st$n_realized,
      p_realized = st$p_realized, max_abs_r = st$max_abs_r,
      rows_match_index = if (is.na(st$n_raw)) NA else st$n_raw == fr$rows_index,
      columns = paste(cols, collapse = "|"),
      colset_sha256 = if (length(cols)) colset_hash(res$type_admitted) else "",
      exclusion_reason = reason, exclusion_detail = detail, stringsAsFactors = FALSE))
    if (reason != "") excl[[i]] <- excl_rows(fr$package, fr$item, "clean", reason, detail)
    if (!quiet && i %% 100 == 0) log_msg("cleaned %d / %d", i, nrow(frame))
  }
  realized <- rbind_list(rows)
  rownames(realized) <- NULL
  list(realized = realized, columns = rbind_list(logs),
       exclusions = rbind_list(excl) %||% excl_rows())
}

# Row-unit tag from the raw table; pure time series are excluded here.
stage_tag <- function(realized, cfg, quiet = FALSE) {
  ok <- realized[realized$exclusion_reason == "", ]
  rownames(ok) <- NULL
  ru <- character(nrow(ok)); dt <- character(nrow(ok))
  for (i in seq_len(nrow(ok))) {
    t <- tag_row_unit(read_raw_csv(ok$raw_file[i]), cfg)
    ru[i] <- t$row_unit; dt[i] <- t$detail
    if (!quiet && i %% 100 == 0) log_msg("tagged %d / %d", i, nrow(ok))
  }
  ok$row_unit <- ru
  ok$row_unit_detail <- dt
  is_ts <- ok$row_unit == "time"
  excl <- excl_rows(ok$package[is_ts], ok$item[is_ts], "tag", "pure_time_series", dt[is_ts])
  tagged <- ok[!is_ts, ]
  rownames(tagged) <- NULL
  list(tagged = tagged, exclusions = excl)
}

stage_dedup <- function(tagged) {
  d <- dedup_frame(tagged)
  what <- ifelse(d$removed$reason == "duplicate_title", "same title as ", "same admitted column set as ")
  excl <- excl_rows(d$removed$package, d$removed$item, "dedup", d$removed$reason,
                    paste0(what, d$removed$duplicate_of), d$removed$duplicate_of)
  list(deduped = d$kept, exclusions = excl)
}

# Licenses: named character vector package -> license string.
load_licenses <- function(cfg) {
  p <- file.path(cfg$dir_work, "03_descriptions.csv")
  lic <- character(0)
  if (file.exists(p)) {
    d <- read_work(p)
    lic <- stats::setNames(d$license, d$package)
  }
  for (k in names(cfg$license_overrides)) lic[k] <- cfg$license_overrides[[k]]
  lic
}

# The draw plus delivery of the selected matrices into <out_dir>/data_clean/.
stage_sample <- function(deduped, licenses, cfg, out_dir = ".", quiet = FALSE) {
  cands <- assign_bins(deduped, cfg)
  clean_dir <- file.path(out_dir, cfg$dir_clean)
  dir_ensure(clean_dir)
  build_fn <- function(row) {
    key <- paste(row$package, row$item, sep = "/")
    res <- clean_dataset(read_raw_csv(row$raw_file), cfg)
    if (is.null(res$X)) {
      return(list(ok = FALSE, reason = "reclean_failed", detail = res$exclusion$reason))
    }
    d <- deliver_matrix(res$X, key, cfg)
    p_bin <- bin_label(ncol(d$X), cfg$p_bins)
    detail <- sprintf("n_delivered=%d p_delivered=%d dropped_constant=%s",
                      nrow(d$X), ncol(d$X), paste(d$dropped_constant, collapse = "|"))
    if (is.na(p_bin) || p_bin != row$p_bin || nrow(d$X) < cfg$min_n) {
      return(list(ok = FALSE, reason = "post_subsample_bin_shift", detail = detail))
    }
    fname <- clean_name(row$package, row$item_stem)
    fpath <- file.path(clean_dir, fname)
    write_csv_atomic(d$X, fpath)
    lic <- licenses[row$package]
    if (is.null(lic) || is.na(lic) || lic == "") lic <- "unknown"
    sel <- data.frame(
      package = row$package, item = row$item, title = row$title, csv_url = row$csv_url,
      doc_url = row$doc_url, license = unname(lic), tier = row$tier, domain = row$domain,
      row_unit = row$row_unit, n_index = row$rows_index, p_index = row$p_model_index,
      n_raw = row$n_raw, n_realized = row$n_realized, p_realized = row$p_realized,
      n_delivered = nrow(d$X), p_delivered = ncol(d$X), max_abs_r = round(max_abs_cor_of(d$X), 6),
      n_bin = row$n_bin, p_bin = row$p_bin,
      cell = row$cell, census = NA, row_subsampled = d$row_subsampled,
      col_subsampled = d$col_subsampled, columns = paste(names(d$X), collapse = "|"),
      clean_file = file.path(cfg$dir_clean, fname), clean_sha256 = sha256_file(fpath),
      stringsAsFactors = FALSE)
    if (!quiet) log_msg("  admit %-12s %s (%d x %d)", row$cell, key, nrow(d$X), ncol(d$X))
    list(ok = TRUE, selected = sel)
  }
  dr <- draw_cells(cands, cfg, build_fn)
  sel <- dr$selected
  sel$census <- dr$cells$census[match(sel$cell, dr$cells$cell)]
  sel <- sel[order_c(match(sel$cell, cell_order(cfg)), sel$package, sel$item), ]
  rownames(sel) <- NULL
  rej <- dr$draw_log[dr$draw_log$decision == "reject", ]
  excl <- excl_rows(rej$package, rej$item, "draw", rej$reason, rej$detail)
  list(selected = sel, draw_log = dr$draw_log, cells = dr$cells,
       not_drawn = dr$not_drawn %||% data.frame(package = character(0), item = character(0),
                                                cell = character(0), position = integer(0)),
       exclusions = excl, candidates = cands)
}

# The exclusion files written by stages 02..07, in stage order.
EXCL_FILES <- c("02_exclusions.csv", "03_exclusions.csv", "04_exclusions.csv",
                "05_exclusions.csv", "06_exclusions.csv", "07_exclusions.csv")
STAGE_ORDER <- c("frame", "download", "clean", "tag", "dedup", "draw")

consolidate_exclusions <- function(work_dir) {
  parts <- lapply(EXCL_FILES, function(f) {
    p <- file.path(work_dir, f)
    if (file.exists(p)) read_work(p) else NULL
  })
  ex <- rbind_list(parts) %||% excl_rows()
  ex <- ex[order_c(match(ex$stage, STAGE_ORDER), ex$package, ex$item), ]
  rownames(ex) <- NULL
  ex
}
