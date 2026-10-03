# R/lib_frame.R -- the sampling frame: the flagged dataset table from
# 00_inputs.R plus the mechanical index-level prefilters.

read_index <- function(cfg) {
  utils::read.csv(file.path(cfg$dir_index, "datasets.csv"), stringsAsFactors = FALSE,
                  check.names = FALSE, na.strings = "NA", fileEncoding = "UTF-8")
}

load_inputs <- function(cfg) {
  list(datasets = read_work(file.path(cfg$dir_work, "00_datasets.csv")))
}

# Returns list(frame = <kept datasets>, exclusions = <one row per dropped index row>).
build_frame <- function(datasets, cfg) {
  m <- datasets
  m$p_model <- pmin(m$n_numeric + m$n_binary + m$n_logical, m$Cols)
  m <- m[order_c(m$Package, m$Item), ]
  rownames(m) <- NULL

  reason <- rep("", nrow(m))
  detail <- rep("", nrow(m))
  set <- function(cond, r, d) {
    cond <- cond & reason == ""
    reason[cond] <<- r
    detail[cond] <<- d[cond]
  }
  set(m$package_exclude == "exclude", "package_excluded", m$package_exclude_reason)
  set(grepl("^word=", m$dataset_exclude_reason), "description_keyword",
      paste0(m$dataset_exclude_reason, "; ", m$description))
  set(grepl("^class=", m$dataset_exclude_reason), "object_class",
      paste0(m$dataset_exclude_reason, " (", m$r_class, ")"))
  set(m$Rows < cfg$min_index_rows, "index_rows_lt_min", paste0("Rows=", m$Rows))
  set(m$p_model < cfg$min_index_p_model, "index_p_model_lt_min", paste0("p_model=", m$p_model))

  keep <- reason == ""
  frame <- data.frame(
    package     = m$Package[keep],
    item        = m$Item[keep],
    item_stem   = item_stem_from_url(m$CSV[keep]),
    title       = m$Title[keep],
    description = m$description[keep],
    r_class     = m$r_class[keep],
    csv_url     = m$CSV[keep],
    doc_url     = m$Doc[keep],
    tier        = m$tier[keep],
    package_domain = m$domain[keep],
    domain      = m$domain[keep],
    rows_index    = m$Rows[keep],
    cols_index    = m$Cols[keep],
    p_model_index = m$p_model[keep],
    stringsAsFactors = FALSE)
  if (anyDuplicated(paste(frame$package, frame$item_stem))) {
    stop("non-unique (package, item_stem) in frame")
  }
  excl <- excl_rows(m$Package[!keep], m$Item[!keep], "frame", reason[!keep], detail[!keep])
  list(frame = frame, exclusions = excl)
}
