# R/lib_clean.R -- column admission and complete-case cleaning of one raw table.
# Pure functions: (data.frame, cfg) -> admitted numeric matrix + decision log.

col_is_id_name   <- function(nm, cfg) grepl(cfg$id_name_regex,   tolower(trimws(nm)), perl = TRUE)
col_is_unit_name <- function(nm, cfg) grepl(cfg$unit_name_regex, tolower(trimws(nm)), perl = TRUE)
col_is_time_name <- function(nm, cfg) grepl(cfg$time_name_regex, tolower(trimws(nm)), perl = TRUE)

# A numeric column of consecutive integers, distinct and monotone along the
# rows, is a row index (1..n, or a sort key).  Ranks stored in another order
# are kept: they are variables, not indices.
is_row_index <- function(x) {
  if (!is.numeric(x) || length(x) < 2 || anyNA(x)) return(FALSE)
  if (any(x != round(x))) return(FALSE)
  if (anyDuplicated(x)) return(FALSE)
  if ((max(x) - min(x)) != (length(x) - 1)) return(FALSE)
  d <- diff(x)
  all(d > 0) || all(d < 0)
}

# One column -> list(values = numeric or NULL, reason, detail).
admit_column <- function(x, nm, cfg) {
  out <- function(values, reason, detail = "") list(values = values, reason = reason, detail = detail)
  if (col_is_id_name(nm, cfg)) return(out(NULL, "id_name"))
  if (all(is.na(x))) return(out(NULL, "all_na"))
  if (is.numeric(x)) {
    if (is_row_index(x)) return(out(NULL, "row_index_values"))
    nn <- x[!is.na(x)]
    if (all(nn %in% c(0, 1))) return(out(as.numeric(x), "binary_numeric"))
    return(out(as.numeric(x), "numeric"))
  }
  if (is.logical(x)) return(out(as.numeric(x), "binary_logical", "FALSE=0,TRUE=1"))
  if (is.character(x) || is.factor(x)) {
    x <- as.character(x)
    lv <- sort_c(unique(x[!is.na(x)]))
    if (length(lv) == 2) {
      return(out(as.numeric(x == lv[2]), "binary_character",
                 iconv(paste0(lv[1], "=0,", lv[2], "=1"), "UTF-8", "UTF-8", sub = "byte")))
    }
    if (length(lv) == 1) return(out(NULL, "constant", paste0("single level: ", lv[1])))
    return(out(NULL, "categorical_gt2", paste0("levels=", length(lv))))
  }
  out(NULL, "unsupported_type", class(x)[1])
}

# Returns list(X, columns, stats, exclusion).  X is NULL when excluded.
clean_dataset <- function(df, cfg) {
  n_raw <- nrow(df)
  nms <- names(df)
  vals <- list()
  type_admitted <- character(0)
  decisions <- vector("list", length(nms))
  for (j in seq_along(nms)) {
    r <- admit_column(df[[j]], nms[j], cfg)
    if (!is.null(r$values)) type_admitted <- c(type_admitted, nms[j])
    if (!is.null(r$values)) {
      na_frac <- mean(is.na(r$values))
      if (na_frac > cfg$max_col_na_frac) {
        r <- list(values = NULL, reason = "na_frac_gt_max",
                  detail = sprintf("na_frac=%.3f", na_frac))
      }
    }
    decisions[[j]] <- data.frame(
      column = nms[j], decision = if (is.null(r$values)) "dropped" else "admitted",
      reason = r$reason, detail = r$detail, stringsAsFactors = FALSE)
    if (!is.null(r$values)) vals[[nms[j]]] <- r$values
  }
  columns <- do.call(rbind, decisions)
  if (is.null(columns)) columns <- data.frame(column = character(0), decision = character(0),
                                              reason = character(0), detail = character(0))

  X <- if (length(vals)) data.frame(vals, check.names = FALSE) else data.frame(row.names = seq_len(n_raw))
  cc <- stats::complete.cases(X)
  n_complete <- sum(cc)
  row_loss <- if (n_raw > 0) 1 - n_complete / n_raw else 1
  X <- X[cc, , drop = FALSE]
  rownames(X) <- NULL

  const <- vapply(X, function(v) length(unique(v)) <= 1, logical(1))
  if (any(const)) {
    hit <- columns$column %in% names(X)[const]
    columns$decision[hit] <- "dropped"
    columns$reason[hit] <- "constant"
    columns$detail[hit] <- "constant after complete-case"
    X <- X[, !const, drop = FALSE]
  }

  # Near-constant columns (optional): a column whose rows OUTSIDE its
  # modal value number fewer than min_minority_rows is dropped like a constant
  # one. A cross-validation fold of such a column is constant, which is where
  # hierNet's R code aborts; and a level carried by one or two rows is not a
  # predictor anyone would keep. Off (0) for the frozen stratified draw.
  mmr <- cfg$min_minority_rows %||% 0L
  if (mmr > 0 && ncol(X) > 0) {
    minority <- vapply(X, minority_rows, numeric(1))
    near <- minority < mmr
    if (any(near)) {
      hit <- columns$column %in% names(X)[near]
      columns$decision[hit] <- "dropped"
      columns$reason[hit] <- "near_constant"
      columns$detail[hit] <- sprintf("minority_rows=%d < %d", as.integer(minority[near]), mmr)[match(columns$column[hit], names(X)[near])]
      X <- X[, !near, drop = FALSE]
    }
  }

  # Collinear columns: walk in source order, drop a column whose |r| with any
  # already-kept column reaches max_abs_cor (raw variables precede their
  # transforms in the source, so the raw one survives).
  max_abs_r <- 0
  if (ncol(X) >= 2) {
    Rm <- suppressWarnings(abs(stats::cor(X)))
    Rm[is.na(Rm)] <- 0
    keep <- integer(0)
    for (j in seq_len(ncol(X))) {
      if (length(keep)) {
        r <- Rm[j, keep]
        if (any(r >= cfg$max_abs_cor)) {
          k <- keep[which.max(r)]
          hit <- match(names(X)[j], columns$column)
          columns$decision[hit] <- "dropped"
          columns$reason[hit] <- "collinear"
          columns$detail[hit] <- sprintf("collinear_with=%s;abs_r=%.4f", names(X)[k], max(r))
          next
        }
      }
      keep <- c(keep, j)
    }
    if (length(keep) >= 2) {
      Rk <- Rm[keep, keep]
      diag(Rk) <- 0
      max_abs_r <- max(Rk)
    }
    X <- X[, keep, drop = FALSE]
  }

  stats <- list(n_raw = n_raw, n_cols_raw = length(nms), n_admitted_cols = length(vals),
                n_complete = n_complete, row_loss_frac = row_loss,
                n_realized = nrow(X), p_realized = ncol(X), max_abs_r = max_abs_r)
  exclusion <- NULL
  if (row_loss > cfg$max_row_loss_frac) {
    exclusion <- list(reason = "row_loss_gt_max", detail = sprintf("row_loss_frac=%.3f", row_loss))
  } else if (ncol(X) < cfg$min_p) {
    exclusion <- list(reason = "realized_p_lt_min", detail = paste0("p=", ncol(X)))
  } else if (nrow(X) < cfg$min_n) {
    exclusion <- list(reason = "realized_n_lt_min", detail = paste0("n=", nrow(X)))
  }
  # type_admitted: the columns that passed the identifier and type rules, before
  # any data-dependent drop (NA fraction, constant, collinear).  It is the
  # deduplication key: the same table cleaned into different realized sets
  # (a train/test split, a copy with a constant column) must still be caught.
  list(X = if (is.null(exclusion)) X else NULL, columns = columns, stats = stats,
       exclusion = exclusion, type_admitted = type_admitted)
}

# rows outside the modal value of a column
minority_rows <- function(v) { tb <- table(v); length(v) - max(tb) }

colset_hash <- function(colnames) sha256_string(paste(sort_c(colnames), collapse = "\n"))

# Largest off-diagonal |r| of a numeric matrix (0 when fewer than 2 columns).
max_abs_cor_of <- function(X) {
  if (ncol(X) < 2) return(0)
  R <- suppressWarnings(abs(stats::cor(X)))
  R[is.na(R)] <- 0
  diag(R) <- 0
  max(R)
}
