# R/lib_inputs.R -- derive the pipeline inputs from the one hand-curated file,
# inputs/package_labels.csv.  Pure functions of (tables, cfg); the network
# fetch of the index stays in 00_inputs.R.

# Step 1: the include column.  "include" if tier == "core", else "exclude",
# overridden by cfg$inputs_include_exceptions.
include_rule <- function(package_labels, cfg) {
  exc <- cfg$inputs_include_exceptions
  unknown <- setdiff(exc$package, package_labels$package)
  if (length(unknown)) {
    stop("include exception names a package not in package_labels.csv: ",
         paste(unknown, collapse = ", "))
  }
  rule <- ifelse(package_labels$tier == "core", "include", "exclude")
  hit <- match(package_labels$package, exc$package)
  rule[!is.na(hit)] <- exc$include[hit[!is.na(hit)]]
  rule
}

# Steps 2-4 given the index snapshot and the labelled packages.
# desc_fn(package) -> data.frame(Item, description) from the installed package.
# Returns list(datasets, versions, report).
build_inputs <- function(index, package_labels, cfg, desc_fn = local_descriptions) {
  pl <- package_labels
  if (!"include" %in% names(pl)) pl$include <- include_rule(pl, cfg)
  exc <- cfg$inputs_include_exceptions

  m <- merge(index, pl[, c("package", "tier", "domain", "include")],
             by.x = "Package", by.y = "package", all.x = TRUE, sort = FALSE)
  m <- m[order_c(m$Package, m$Item), ]
  rownames(m) <- NULL
  labelled <- !is.na(m$include)
  included <- labelled & m$include == "include"
  m$package_exclude <- ifelse(included, "", "exclude")
  reason <- rep("package not in package_labels.csv", nrow(m))
  reason[labelled] <- paste0("tier=", m$tier[labelled])
  ex_hit <- match(m$Package, exc$package)
  reason[!is.na(ex_hit)] <- paste0("exception: ", exc$reason[ex_hit[!is.na(ex_hit)]])
  reason[included] <- ""
  m$package_exclude_reason <- reason
  for (k in c("tier", "domain", "include")) m[[k]][is.na(m[[k]])] <- ""

  inc_pkgs <- sort_c(pl$package[pl$include == "include"])
  desc <- list(); vers <- list()
  for (p in inc_pkgs) {
    d <- desc_fn(p)
    desc[[p]] <- data.frame(Package = p, Item = d$Item, description = d$description,
                            r_class = d$r_class, stringsAsFactors = FALSE)
    vers[[p]] <- data.frame(package = p, version = d$version, n_local_datasets = nrow(d),
                            n_index_datasets = sum(m$Package == p), stringsAsFactors = FALSE)
  }
  desc <- do.call(rbind, desc); rownames(desc) <- NULL
  vers <- do.call(rbind, vers); rownames(vers) <- NULL
  key_m <- paste(m$Package, m$Item, sep = "/")
  key_d <- paste(desc$Package, desc$Item, sep = "/")
  hit_d <- match(key_m, key_d)
  loc <- desc$description[hit_d]
  m$description_source <- ifelse(!is.na(loc), "installed package",
                                 ifelse(included, "index title (no local match)", "index title"))
  m$description <- ifelse(is.na(loc), m$Title, loc)
  m$r_class <- ifelse(is.na(hit_d), "", desc$r_class[hit_d])

  pat <- paste0("\\b(", paste(cfg$inputs_exclude_words, collapse = "|"), ")\\b")
  hitw <- regexpr(pat, m$description, ignore.case = TRUE, perl = TRUE)
  cls_hit <- vapply(strsplit(m$r_class, "/", fixed = TRUE),
                    function(cl) { z <- intersect(cl, cfg$inputs_exclude_classes); if (length(z)) z[1] else "" }, "")
  m$dataset_exclude_reason <- ""
  m$dataset_exclude_reason[cls_hit != ""] <- paste0("class=", cls_hit[cls_hit != ""])
  m$dataset_exclude_reason[hitw > 0] <- paste0("word=", tolower(regmatches(m$description, hitw)))
  m$dataset_exclude <- ifelse(m$dataset_exclude_reason != "", "exclude", "")

  nomatch <- included & m$description_source != "installed package"
  mism <- included & m$description_source == "installed package" & m$description != m$Title
  kw <- which(included & grepl("^word=", m$dataset_exclude_reason))
  kc <- which(included & grepl("^class=", m$dataset_exclude_reason))
  report <- c(
    sprintf("index rows %d; in included packages %d (%d packages)", nrow(m), sum(included), length(inc_pkgs)),
    if (any(nomatch)) sprintf("  no local match (index title used): %s", paste(key_m[nomatch], collapse = ", ")),
    if (length(setdiff(key_d, key_m))) sprintf("  local datasets absent from the index snapshot: %s", paste(setdiff(key_d, key_m), collapse = ", ")),
    if (any(mism)) sprintf("  local title differs from index title for %d datasets: %s", sum(mism), paste(key_m[mism], collapse = ", ")),
    sprintf("dataset_exclude by description keyword, within included packages: %d", length(kw)),
    sprintf("  %-34s %-18s %s", key_m[kw], m$dataset_exclude_reason[kw], m$description[kw]),
    sprintf("dataset_exclude by object class, within included packages: %d (%s)", length(kc),
            paste(names(table(m$dataset_exclude_reason[kc])), table(m$dataset_exclude_reason[kc]), collapse = ", ")),
    sprintf("frame candidates (both flags empty): %d", sum(included & m$dataset_exclude == "")))
  list(datasets = m, versions = vers, report = report)
}

# Descriptions and object classes from the installed package:
# utils::data(package = p)$results gives Item and Title; each object is then
# loaded to read its class (an Item "obj (file)" is object obj in data file file).
local_descriptions <- function(p) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop("included package not installed locally: ", p, "\n  install.packages(\"", p, "\")")
  }
  r <- utils::data(package = p)$results
  r <- r[!duplicated(r[, "Item"]), , drop = FALSE]
  cls <- vapply(r[, "Item"], function(item) {
    m <- regmatches(item, regexec("^(.*) \\((.*)\\)$", item))[[1]]
    obj <- if (length(m)) m[2] else item
    file <- if (length(m)) m[3] else item
    e <- new.env()
    tryCatch({
      suppressWarnings(utils::data(list = file, package = p, envir = e))
      if (exists(obj, envir = e, inherits = FALSE)) paste(class(get(obj, envir = e)), collapse = "/") else "not_found"
    }, error = function(err) "error")
  }, "")
  data.frame(Item = r[, "Item"], description = r[, "Title"], r_class = unname(cls),
             version = as.character(utils::packageVersion(p)), stringsAsFactors = FALSE)
}

read_package_labels <- function(path) {
  pl <- utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE,
                        na.strings = character(0), fileEncoding = "UTF-8")
  for (k in c("package", "tier", "domain")) if (!k %in% names(pl)) stop(path, " lacks column ", k)
  if (anyDuplicated(pl$package)) stop("duplicate package rows in ", path)
  pl
}
