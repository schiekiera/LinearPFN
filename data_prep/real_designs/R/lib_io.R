# R/lib_io.R -- hashing, cached downloads, CSV I/O, logging.  Base R only.

sha256_file <- function(path) unname(tools::sha256sum(path))

sha256_string <- function(s) unname(tools::sha256sum(bytes = charToRaw(enc2utf8(s))))

log_msg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "|", sprintf(...), "\n")

dir_ensure <- function(d) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
  invisible(d)
}

# Locale-independent ordering (C collation, byte order for strings) so results
# match on any machine.  Strings are marked UTF-8 first: radix refuses
# native strings of unknown encoding, which is what read.csv returns for
# non-ASCII values.
.utf8 <- function(x) if (is.character(x)) enc2utf8(x) else x
sort_c  <- function(x) sort(.utf8(x), method = "radix")
order_c <- function(...) do.call(order, c(lapply(list(...), .utf8), method = "radix"))

# Atomic CSV write: to <path>.tmp, then rename.
write_csv_atomic <- function(df, path) {
  dir_ensure(dirname(path))
  tmp <- paste0(path, ".tmp")
  utils::write.csv(df, tmp, row.names = FALSE, fileEncoding = "UTF-8")
  if (!file.rename(tmp, path)) stop("could not rename ", tmp, " to ", path)
  invisible(path)
}

# Work tables are written by us; read them back with stable types.
WORK_NUM <- c("Rows", "Cols", "n_binary", "n_character", "n_factor", "n_logical", "n_numeric",
              "rows_index", "cols_index", "p_model_index", "n_raw", "n_cols_raw",
              "n_admitted_cols", "n_complete", "row_loss_frac", "n_realized", "p_realized",
              "n_delivered", "p_delivered", "max_abs_r", "position", "n_eligible", "n_admitted",
              "n_rejected", "n_not_drawn")
WORK_LGL <- c("override_applied", "rows_match_index", "row_subsampled", "col_subsampled", "census")

read_work <- function(path) {
  df <- utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE,
                        colClasses = "character", na.strings = character(0),
                        fileEncoding = "UTF-8")
  for (k in intersect(names(df), WORK_NUM)) df[[k]] <- suppressWarnings(as.numeric(df[[k]]))
  for (k in intersect(names(df), WORK_LGL)) df[[k]] <- as.logical(df[[k]])
  df
}

# Raw Rdatasets CSV.  No fileEncoding argument on purpose: with it, a single
# invalid byte makes read.csv silently truncate the table.  Duplicate header
# names are made unique so columns can be addressed by name.
read_raw_csv <- function(path) {
  df <- utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE,
                        na.strings = c("NA", ""), comment.char = "")
  names(df) <- make.unique(names(df), sep = "_dup")
  df
}

# Cached, polite, resumable download.  Returns "cached", "downloaded" or "failed".
download_cached <- function(url, dest, cfg) {
  if (file.exists(dest) && file.size(dest) > 0) return("cached")
  dir_ensure(dirname(dest))
  tmp <- paste0(dest, ".part")
  for (attempt in seq_len(cfg$download_retries)) {
    ok <- tryCatch({
      status <- suppressWarnings(utils::download.file(
        url, tmp, method = "libcurl", quiet = TRUE, mode = "wb",
        headers = c(`User-Agent` = cfg$user_agent)))
      status == 0 && file.exists(tmp) && file.size(tmp) > 0
    }, error = function(e) FALSE)
    Sys.sleep(cfg$download_sleep_sec)
    if (ok) {
      file.rename(tmp, dest)
      return("downloaded")
    }
    if (file.exists(tmp)) unlink(tmp)
    if (attempt < cfg$download_retries) Sys.sleep(2^attempt)
  }
  "failed"
}

# File-system-safe stem for a dataset item (taken from the index's CSV URL).
item_stem_from_url <- function(url) gsub("[^A-Za-z0-9._-]", "_", sub("\\.csv$", "", basename(url)))

raw_path   <- function(package, item_stem, cfg) file.path(cfg$dir_csv, package, paste0(item_stem, ".csv"))
clean_name <- function(package, item_stem) paste0(package, "__", item_stem, ".csv")

# Exclusion-log rows: one per dropped dataset, machine-readable reason.
excl_rows <- function(package = character(0), item = character(0), stage = character(0),
                      reason = character(0), detail = "", duplicate_of = "") {
  n <- length(package)
  data.frame(package = package, item = item, stage = rep_len(stage, n),
             reason = rep_len(reason, n), detail = rep_len(detail, n),
             duplicate_of = rep_len(duplicate_of, n), stringsAsFactors = FALSE)
}

rbind_list <- function(lst) {
  lst <- lst[!vapply(lst, is.null, logical(1))]
  if (length(lst) == 0) return(NULL)
  do.call(rbind, lst)
}

# Flatten the config list to key/value rows for MANIFEST.csv.
cfg_flat <- function(cfg) {
  rows <- lapply(names(cfg), function(k) {
    v <- cfg[[k]]
    v <- if (is.list(v)) paste(do.call(paste, c(v, sep = ":")), collapse = "|")
         else paste(v, collapse = "|")
    data.frame(key = k, value = v, stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
}
