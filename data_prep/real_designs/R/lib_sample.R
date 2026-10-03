# R/lib_sample.R -- bins, deduplication, seeded subsampling and the per-cell
# seeded draw.  Pure functions; file I/O is injected through build_fn.

bin_label <- function(v, bins) {
  lab <- rep(NA_character_, length(v))
  for (k in seq_along(bins$label)) lab[v >= bins$lo[k] & v <= bins$hi[k]] <- bins$label[k]
  lab
}

cell_id <- function(p_bin, n_bin) paste0("p", p_bin, "_n", n_bin)

# Canonical cell order: p bins outer, n bins inner.
cell_order <- function(cfg) {
  as.vector(t(outer(cfg$p_bins$label, cfg$n_bins$label, cell_id)))
}

assign_bins <- function(df, cfg) {
  df$row_subsampled <- df$n_realized > cfg$n_cap
  df$col_subsampled <- df$p_realized > cfg$p_cap
  df$n_bin <- bin_label(pmin(df$n_realized, cfg$n_cap), cfg$n_bins)
  df$p_bin <- bin_label(pmin(df$p_realized, cfg$p_cap), cfg$p_bins)
  df$cell  <- cell_id(df$p_bin, df$n_bin)
  df
}

# Two keys, applied in turn; within a key keep the largest n, then the core
# tier, then alphabetical (package, item):
#   1. the type-admitted column-name set (same table, train/test splits);
#   2. the normalized title (lower-case alphanumerics), only for groups that
#      span more than one package (the same source shipped under different
#      column names, e.g. Ecdat/Crime and plm/Crime).
dedup_frame <- function(df) {
  pass <- function(d, key, reason, multi_pkg_only = FALSE) {
    o <- order_c(key, -d$n_realized, d$tier != "core", d$package, d$item)
    d <- d[o, ]; key <- key[o]
    first <- match(key, key)
    dup <- seq_len(nrow(d)) != first & key != ""
    if (multi_pkg_only) {
      n_pkg <- vapply(key, function(k) length(unique(d$package[key == k])), integer(1))
      dup <- dup & n_pkg > 1
    }
    removed <- d[dup, ]
    removed$duplicate_of <- paste(d$package, d$item, sep = "/")[first[dup]]
    removed$reason <- rep(reason, nrow(removed))
    list(kept = d[!dup, ], removed = removed)
  }
  p1 <- pass(df, df$colset_sha256, "duplicate_columns")
  k1 <- p1$kept
  p2 <- pass(k1, gsub("[^a-z0-9]", "", tolower(k1$title)), "duplicate_title", multi_pkg_only = TRUE)
  kept <- p2$kept[order_c(p2$kept$package, p2$kept$item), ]
  rownames(kept) <- NULL
  removed <- rbind(p1$removed, p2$removed)
  rownames(removed) <- NULL
  list(kept = kept, removed = removed)
}

set_rng <- function(cfg, seed) {
  RNGkind(cfg$rng_kind[1], cfg$rng_kind[2], cfg$rng_kind[3])
  set.seed(seed)
}

# Per-dataset seed: global seed + the first 7 hex digits of sha256(key).
# Independent of the draw order, so a dataset's subsample never changes when
# other datasets enter or leave the frame.
dataset_seed <- function(key, cfg) {
  (cfg$seed + strtoi(substr(sha256_string(key), 1, 7), 16L)) %% 2147483647L
}

# Seeded column subsample (p > p_cap), then seeded row subsample (n > n_cap),
# then re-drop columns that became constant.
deliver_matrix <- function(X, key, cfg) {
  set_rng(cfg, dataset_seed(key, cfg))
  col_sub <- ncol(X) > cfg$p_cap
  if (col_sub) X <- X[, sort(sample.int(ncol(X), cfg$p_cap)), drop = FALSE]
  row_sub <- nrow(X) > cfg$n_cap
  if (row_sub) X <- X[sort(sample.int(nrow(X), cfg$n_cap)), , drop = FALSE]
  rownames(X) <- NULL
  const <- vapply(X, function(v) length(unique(v)) <= 1, logical(1))
  mmr <- cfg$min_minority_rows %||% 0L
  if (mmr > 0) const <- const | (vapply(X, minority_rows, numeric(1)) < mmr)  # near-constant after subsampling
  list(X = X[, !const, drop = FALSE], row_subsampled = row_sub, col_subsampled = col_sub,
       dropped_constant = names(X)[const])
}

# The draw.  cands: eligible datasets (post dedup) with package, item, cell.
# build_fn(row) -> list(ok, reason, detail, ...) ; extra fields are kept.
# All 16 permutations are drawn first under the global seed, then walked, so
# per-dataset subsampling seeds (set inside build_fn) cannot touch the draw.
draw_cells <- function(cands, cfg, build_fn) {
  cells <- cell_order(cfg)
  set_rng(cfg, cfg$seed)
  perms <- list()
  for (cl in cells) {
    e <- cands[cands$cell == cl, ]
    e <- e[order_c(e$package, e$item), ]
    if (nrow(e) > 1) e <- e[sample.int(nrow(e)), ]
    rownames(e) <- NULL
    perms[[cl]] <- e
  }
  log <- list(); selected <- list(); not_drawn <- list(); summary <- list()
  for (cl in cells) {
    e <- perms[[cl]]
    n_adm <- 0L; n_rej <- 0L; pos <- 0L
    while (pos < nrow(e) && n_adm < cfg$quota_per_cell) {
      pos <- pos + 1L
      row <- e[pos, ]
      res <- build_fn(row)
      if (isTRUE(res$ok)) {
        n_adm <- n_adm + 1L
        selected[[length(selected) + 1]] <- res$selected
      } else {
        n_rej <- n_rej + 1L
      }
      log[[length(log) + 1]] <- data.frame(
        cell = cl, position = pos, package = row$package, item = row$item,
        decision = if (isTRUE(res$ok)) "admit" else "reject",
        reason = if (isTRUE(res$ok)) "" else res$reason,
        detail = if (is.null(res$detail)) "" else res$detail, stringsAsFactors = FALSE)
    }
    if (pos < nrow(e)) {
      nd <- e[(pos + 1):nrow(e), c("package", "item", "cell"), drop = FALSE]
      nd$position <- (pos + 1):nrow(e)
      not_drawn[[length(not_drawn) + 1]] <- nd
    }
    summary[[length(summary) + 1]] <- data.frame(
      cell = cl, n_eligible = nrow(e), n_admitted = n_adm, n_rejected = n_rej,
      n_not_drawn = nrow(e) - pos, census = n_adm < cfg$quota_per_cell,
      stringsAsFactors = FALSE)
  }
  list(selected = rbind_list(selected), draw_log = rbind_list(log),
       not_drawn = rbind_list(not_drawn), cells = do.call(rbind, summary))
}
