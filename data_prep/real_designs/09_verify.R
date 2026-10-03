#!/usr/bin/env Rscript


source("00_config.R")
fails <- 0L
check <- function(ok, what) {
  if (isTRUE(ok)) {
    cat("  ok   ", what, "\n")
  } else {
    cat("  FAIL ", what, "\n")
    fails <<- fails + 1L
  }
}

cat("[a] self-checks on synthetic tables\n")
set.seed(1)
n <- 40
syn <- data.frame(
  rownames = 1:n, id = n:1, x = rnorm(n), flag = rep(c("yes", "no"), n / 2),
  grp = rep(c("a", "b", "c", "d"), n / 4), const = 5, y = c(NA, NA, rnorm(n - 2)),
  serial_no = 100 + 1:n, ok = rnorm(n) > 0, check.names = FALSE,
  stringsAsFactors = FALSE
)
cs <- clean_dataset(syn, CFG)
check(identical(names(cs$X), c("x", "flag", "y", "ok")), "admits numeric, 2-level character, logical; drops id/const/3+levels")
check(cs$stats$n_realized == n - 2 && cs$stats$p_realized == 4, "complete-case n and realized p")
check(all(cs$X$flag[seq(1, n - 2, 2)] == 1), "2-level character coded sorted levels: no=0, yes=1")
check(cs$columns$reason[cs$columns$column == "grp"] == "categorical_gt2", "categorical_gt2 logged")
check(cs$columns$reason[cs$columns$column == "serial_no"] == "id_name", "serial_no matched by id regex")
rowidx <- data.frame(v = 5:(n + 4), x = rnorm(n), z = rnorm(n), w = rnorm(n))
check(clean_dataset(rowidx, CFG)$columns$reason[1] == "row_index_values", "consecutive-integer column in row order dropped as row index")
ranks <- data.frame(r = sample(n), x = rnorm(n), z = rnorm(n))
check(clean_dataset(ranks, CFG)$columns$reason[1] == "numeric", "consecutive integers NOT in row order kept (ranks are variables)")
sparse <- data.frame(x = rnorm(n), z = rnorm(n), w = rnorm(n), s = c(rnorm(10), rep(NA, n - 10)))
cs2 <- clean_dataset(sparse, CFG)
check(is.null(cs2$exclusion) && cs2$stats$n_realized == n, "sparse column dropped before complete-case, dataset survives")
half <- data.frame(x = c(rep(NA, 15), rnorm(n - 15)), z = c(rnorm(15), rep(NA, 15), rnorm(n - 30)), w = rnorm(n))
check(clean_dataset(half, CFG)$exclusion$reason == "row_loss_gt_max", "row loss above threshold (NA spread over columns) excludes")
coll <- data.frame(x = rnorm(n))
coll$x2 <- 3 * coll$x + 1
coll$z <- rnorm(n)
coll$w <- coll$z + rnorm(n)
cc <- clean_dataset(coll, CFG)
check(identical(names(cc$X), c("x", "z", "w")) && cc$columns$reason[cc$columns$column == "x2"] == "collinear", "linear transform dropped as collinear, earlier column kept")
check(cc$stats$max_abs_r < CFG$max_abs_cor && cc$stats$max_abs_r > 0.5, "max_abs_r recorded below the threshold")
check(tag_row_unit(data.frame(firm = rep(1:4, each = 5), year = rep(1:5, 4), v = 1), CFG)$row_unit == "panel", "panel tag")
check(tag_row_unit(data.frame(year = 1950:1989, gdp = rnorm(40)), CFG)$row_unit == "time", "time tag")
check(tag_row_unit(data.frame(rownames = 1:n, state = paste0("s", 1:n), year = 2000, v = 1), CFG)$row_unit == "cross-section", "cross-section tag (constant year, unique units)")
check(tag_row_unit(data.frame(region = rep(c("n", "s"), n / 2), wage = rnorm(n)), CFG)$row_unit == "cross-section", "repeated unit noun without a time axis stays cross-section")
check(tag_row_unit(data.frame(time = c("1990-01-01", "1990-02-01", "1990-03-01"), v = 1:3), CFG)$row_unit == "time", "date string time tag")
check(tag_row_unit(data.frame(Quarter = c("1970 Q1", "1970 Q2", "1970 Q3", "1970 Q4", "1971 Q1"), v = 1:5), CFG)$row_unit == "time", "quarter string time tag")
check(tag_row_unit(data.frame(Month = c("1991 Jul", "1991 Aug", "1991 Sep"), v = 1:3), CFG)$row_unit == "time", "year-month string time tag")
dd <- data.frame(
  package = c("b", "a", "a", "c", "d", "d"), item = c("t", "t_train", "t_test", "crime", "crime", "other"),
  tier = c("core", "mixed", "mixed", "core", "core", "core"), n_realized = c(100, 300, 100, 630, 630, 50),
  colset_sha256 = c("h", "h", "h", "k1", "k2", "k3"), title = c("T", "T", "T", "Crime in NC", "Crime in N.C.", "Crime in N.C."),
  stringsAsFactors = FALSE
)
d <- dedup_frame(dd)
check(all(d$kept$item %in% c("t_train", "crime")) && nrow(d$kept) == 2 && sum(d$removed$reason == "duplicate_columns") == 2, "dedup key 1 keeps largest n, logs duplicate_of")
check(sum(d$removed$reason == "duplicate_title") == 2 && all(d$removed$duplicate_of[d$removed$reason == "duplicate_title"] == "c/crime"), "dedup key 2: same normalized title across packages")
cands <- data.frame(
  package = rep("p", 8), item = paste0("i", 1:8), cell = c(rep(cell_order(CFG)[1], 6), rep(cell_order(CFG)[2], 2)),
  stringsAsFactors = FALSE
)
bf <- function(row) if (row$item == "i3") list(ok = FALSE, reason = "test", detail = "") else list(ok = TRUE, selected = data.frame(item = row$item))
r1 <- draw_cells(cands, CFG, bf)
r2 <- draw_cells(cands, CFG, bf)
check(identical(r1$selected, r2$selected) && identical(r1$draw_log, r2$draw_log), "draw is deterministic")
check(sum(r1$cells$n_admitted[1]) == 4 && !r1$cells$census[1] && r1$cells$census[2] && r1$cells$n_admitted[2] == 2, "quota and census")
check(all(r1$draw_log$decision[r1$draw_log$item == "i3"] == "reject"), "rejection logged, walk continues")
check(nrow(r1$not_drawn) == 8 - nrow(r1$draw_log), "not-drawn accounting")
X <- as.data.frame(matrix(rnorm(2500 * 40), 2500, 40))
names(X) <- paste0("c", 1:40)
dv <- deliver_matrix(X, "k", CFG)
dv2 <- deliver_matrix(X, "k", CFG)
check(nrow(dv$X) == CFG$n_cap && ncol(dv$X) == CFG$p_cap && identical(dv$X, dv2$X), "subsampling caps and is seeded per dataset")

cat("[b] re-run inputs -> frame -> clean -> tag -> dedup -> draw from cached inputs\n")
vdir <- file.path(CFG$dir_work, "verify")
unlink(vdir, recursive = TRUE)
dir_ensure(vdir)
pl <- read_package_labels(file.path(CFG$dir_inputs, "package_labels.csv"))
check(identical(pl$include, include_rule(pl, CFG)), "package_labels.csv include column matches the rule")
inp <- build_inputs(read_index(CFG), pl, CFG)
vin <- file.path(vdir, "00_datasets.csv")
write_csv_atomic(inp$datasets, vin)
v0 <- read_work(file.path(CFG$dir_work, "00_package_versions.csv"))
vdiff <- v0$package[v0$version != inp$versions$version[match(v0$package, inp$versions$package)]]
if (length(vdiff)) cat("  note: installed versions differ from the recorded ones for", paste(vdiff, collapse = ", "), "\n")
check(sha256_file(vin) == sha256_file(file.path(CFG$dir_work, "00_datasets.csv")), "work/00_datasets.csv byte-identical when re-derived from package_labels.csv + index")
fr <- stage_frame(CFG)
frame0 <- read_work(file.path(CFG$dir_work, "02_frame.csv"))
check(identical(paste(fr$frame$package, fr$frame$item), paste(frame0$package, frame0$item)), sprintf("frame identical (%d datasets)", nrow(fr$frame)))
cl <- stage_clean(fr$frame, CFG, quiet = TRUE)
tg <- stage_tag(cl$realized, CFG, quiet = TRUE)
dd <- stage_dedup(tg$tagged)
sm <- stage_sample(dd$deduped, load_licenses(CFG), CFG, out_dir = vdir, quiet = TRUE)
vsel <- file.path(vdir, CFG$file_selected)
write_csv_atomic(sm$selected, vsel)
check(sha256_file(vsel) == sha256_file(CFG$file_selected), "selected_datasets.csv byte-identical on re-run")
sel <- read_work(CFG$file_selected)
same <- vapply(seq_len(nrow(sel)), function(i) {
  a <- sel$clean_file[i]
  b <- file.path(vdir, sel$clean_file[i])
  file.exists(a) && file.exists(b) && sha256_file(a) == sha256_file(b) && sha256_file(a) == sel$clean_sha256[i]
}, logical(1))
check(all(same), sprintf("all %d delivered matrices byte-identical and match recorded sha256", nrow(sel)))
vex <- rbind_list(list(fr$exclusions, read_work(file.path(CFG$dir_work, "03_exclusions.csv")), cl$exclusions, tg$exclusions, dd$exclusions, sm$exclusions))
vex <- vex[order_c(match(vex$stage, STAGE_ORDER), vex$package, vex$item), ]
ex0 <- read_work(CFG$file_exclusions)
check(nrow(vex) == nrow(ex0) && setequal(paste(vex$package, vex$item, vex$reason), paste(ex0$package, ex0$item, ex0$reason)), sprintf("exclusion log identical (%d rows)", nrow(ex0)))
idx_n <- length(readLines(file.path(CFG$dir_index, "datasets.csv"))) - 1
nd <- nrow(sm$not_drawn)
check(idx_n == nrow(sel) + nrow(ex0) + nd, sprintf("accounting: %d index = %d selected + %d excluded + %d eligible-not-drawn", idx_n, nrow(sel), nrow(ex0), nd))

cat("[c] cached downloads vs MANIFEST.csv\n")
man <- read_work(CFG$file_manifest)
raw <- man[man$section %in% c("raw_csv", "description"), ]
ok_hash <- vapply(seq_len(nrow(raw)), function(i) file.exists(raw$key[i]) && sha256_file(raw$key[i]) == raw$value[i], logical(1))
check(all(ok_hash), sprintf("%d cached files match MANIFEST sha256", nrow(raw)))
check(man$value[man$section == "index" & man$key == "sha256"] == sha256_file(file.path(CFG$dir_index, "datasets.csv")), "index snapshot matches MANIFEST sha256")
check(all(vapply(seq_len(nrow(sel)), function(i) sha256_file(sel$clean_file[i]) == sel$clean_sha256[i], logical(1))), "data_clean/ files match selected_datasets.csv sha256")

cat("[d] summary\n")
cells <- read_work(file.path(CFG$dir_work, "07_cells.csv"))
tab <- matrix(sprintf("%d/%d", cells$n_admitted, cells$n_eligible),
  nrow = length(CFG$p_bins$label), byrow = TRUE,
  dimnames = list(p = CFG$p_bins$label, n = CFG$n_bins$label)
)
cat("selected / eligible per cell (p rows x n columns):\n")
print(tab, quote = FALSE)
cat(sprintf("census cells: %s\n", if (any(cells$census)) paste(cells$cell[cells$census], collapse = ", ") else "none"))
cat("exclusions by stage and reason:\n")
print(table(stage = factor(ex0$stage, STAGE_ORDER), reason = ex0$reason))
cat("selected: row unit x tier\n")
print(table(row_unit = sel$row_unit, tier = sel$tier))
cat(sprintf("selected: %d row-subsampled, %d column-subsampled\n", sum(sel$row_subsampled), sum(sel$col_subsampled)))

if (fails > 0) {
  cat(sprintf("\nVERIFY FAILED: %d check(s)\n", fails))
  quit(status = 1)
}
cat("\nVERIFY PASSED\n")
