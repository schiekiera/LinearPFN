# R/lib_tag.R -- best-effort row-unit tag from the RAW table: panel / time /
# cross-section.  Used for post-hoc slicing and to exclude pure time series;
# never used for stratification.

# Numeric time value for a raw column, or NULL when it does not look like time:
# numbers as they are; ISO / common dates; "1970 Q1", "1991 Jul", "Jul 1991",
# "1984-85" (season -> first year), "1991-07" (year-month).
parse_time_values <- function(x) {
  if (is.numeric(x)) return(as.numeric(x))
  if (!is.character(x)) return(NULL)
  v <- suppressWarnings(as.numeric(x))
  if (any(!is.na(v))) return(v)
  s <- trimws(x)
  nn <- s[!is.na(s)]
  if (length(nn) == 0) return(NULL)
  mon <- function(m) match(tolower(m), tolower(month.abb))
  if (all(grepl("^[0-9]{4} ?Q[1-4]$", nn))) {
    return(as.numeric(substr(s, 1, 4)) + (as.numeric(sub("^[0-9]{4} ?Q", "", s)) - 1) / 4)
  }
  if (all(grepl("^[0-9]{4}[- ][A-Za-z]{3}$", nn))) {
    return(as.numeric(substr(s, 1, 4)) + (mon(substr(s, 6, 8)) - 1) / 12)
  }
  if (all(grepl("^[A-Za-z]{3} [0-9]{4}$", nn))) {
    return(as.numeric(substr(s, 5, 8)) + (mon(substr(s, 1, 3)) - 1) / 12)
  }
  if (all(grepl("^[0-9]{4}[-/][0-9]{2}$", nn))) {
    mm <- as.numeric(substr(s, 6, 7))
    yy <- as.numeric(substr(s, 1, 4))
    return(if (all(mm[!is.na(mm)] <= 12)) yy + (mm - 1) / 12 else yy)
  }
  if (all(grepl("^[0-9]{4}[-/][0-9]{4}$", nn))) return(as.numeric(substr(s, 1, 4)))
  d <- tryCatch(as.Date(s, tryFormats = c("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%m/%d/%Y")),
                error = function(e) NULL)
  if (is.null(d) || all(is.na(d))) return(NULL)
  as.numeric(d)
}

is_strict_monotone <- function(v) {
  v <- v[!is.na(v)]
  if (length(v) < 2) return(FALSE)
  d <- diff(v)
  all(d > 0) || all(d < 0)
}

# list(row_unit, detail)
#   panel         : a unit-like column has repeated values AND a time-like
#                   column with >= 2 distinct values exists (unit x time rows)
#   time          : a time-like column is strictly monotone over all rows
#                   (one row per period, no cross-sectional unit)
#   cross-section : everything else (incl. clustered data without a time axis)
tag_row_unit <- function(df, cfg) {
  nms <- names(df)
  low <- tolower(trimws(nms))
  time_cols <- which(col_is_time_name(nms, cfg))
  time_ok <- vapply(time_cols, function(j) {
    v <- parse_time_values(df[[j]])
    !is.null(v) && length(unique(v[!is.na(v)])) >= 2
  }, logical(1))
  time_cols <- time_cols[time_ok]
  unit_cols <- which(col_is_unit_name(nms, cfg) & low != "rownames")
  if (length(time_cols) > 0) {
    for (j in unit_cols) {
      x <- df[[j]]
      x <- x[!is.na(x)]
      if (length(x) > 1 && anyDuplicated(x)) {
        return(list(row_unit = "panel",
                    detail = paste0("unit_col=", nms[j], ";time_col=", nms[time_cols[1]])))
      }
    }
  }
  for (j in time_cols) {
    if (is_strict_monotone(parse_time_values(df[[j]]))) {
      return(list(row_unit = "time", detail = paste0("time_col=", nms[j])))
    }
  }
  list(row_unit = "cross-section", detail = "")
}
