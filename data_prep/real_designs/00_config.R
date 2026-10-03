# 00_config.R -- single source of truth for the real_designs pipeline.
#
# Every tunable lives here: the global seed and RNG kind, the source URLs,
# the mechanical admission thresholds, the stratification bins, the quota
# and the folder layout.  Every numbered script sources this file first.
# Changing any value changes the list of selected datasets; the values in
# force are copied verbatim into MANIFEST.csv by 08_manifest.R.
#
# Requirements: R >= 4.5.0 (tools::sha256sum).  No add-on packages.
# Run every script from inside real_designs/ (all paths are relative to it).

if (getRversion() < "4.5.0") {
  stop("real_designs needs R >= 4.5.0 (tools::sha256sum); found ", getRversion())
}
if (!file.exists("00_config.R")) {
  stop("run the scripts from inside real_designs/ (setwd / cd there first)")
}

CFG <- list(
  # ---- reproducibility -----------------------------------------------------
  seed = 20260904L,
  rng_kind = c("Mersenne-Twister", "Inversion", "Rejection"),

  # ---- sources -------------------------------------------------------------
  index_url = "https://vincentarelbundock.github.io/Rdatasets/datasets.csv",
  cran_desc_url_tpl = "https://cran.r-project.org/web/packages/%s/DESCRIPTION",
  user_agent = "LinearPFN-real_designs/1.0 (academic benchmark construction; base R download.file)",
  download_sleep_sec = 0.2,
  download_retries = 3L,

  # ---- inputs (00_inputs.R): the only hand input is inputs/package_labels.csv ----
  # include = "include" if tier == "core" else "exclude", with these exceptions
  inputs_include_exceptions = data.frame(
    package = "itsadug", include = "exclude",
    reason = "time-series (EEG) data only", stringsAsFactors = FALSE
  ),
  # a dataset whose description contains one of these words (whole word,
  # case-insensitive) gets dataset_exclude = "exclude"
  inputs_exclude_words = c("hypothetical", "simulated", "synthetic", "fake", "artificial", "fictional"),
  # a dataset whose R object (in the installed package) has one of these classes
  # gets dataset_exclude = "exclude": time-series classes export without a time
  # column, contingency tables have cells as rows
  inputs_exclude_classes = c("ts", "mts", "zoo", "xts", "its", "timeSeries", "table"),

  # ---- frame: index-level mechanical prefilters -----------------------------
  min_index_rows = 30L,
  min_index_p_model = 3L, # p_model = min(n_numeric + n_binary + n_logical, Cols)

  # ---- cleaning: dataset-level mechanical rules ----------------------------
  min_n = 30L, # realized rows after complete-case
  min_p = 3L, # realized admitted columns
  max_row_loss_frac = 0.5, # complete-case may drop at most this fraction of rows
  max_col_na_frac = 0.5, # columns with more NA than this are dropped first
  min_minority_rows = 0L, # >0: drop a column with fewer rows outside its modal value (10_deliver_eligible.R sets 10L)
  max_abs_cor = 0.95, # a column with |r| >= this to an EARLIER kept column is
  # dropped (same measurement twice: transforms, re-codings)
  # column names (lower-cased, trimmed) matching this are identifiers -> dropped
  id_name_regex = paste0(
    "^(rownames|row|rows|rowid|rownum|rownumber|row_number|id|ids|idx|index|indices|",
    "obs|observation|case|caseid|casenum|caseno|serial|serialno|key|name|names|fullname|",
    "firstname|first_name|lastname|last_name|subject|subj|subjid|respondent|participant|",
    "pid|sid|uid|hhid|record|recordid|nr|code)$",
    "|(^|[._-])(id|ids)$|[._-](no|nr|num|number|code)$"
  ),
  # names of cross-sectional units (used ONLY for the row-unit tag)
  unit_name_regex = paste0(
    "^(country|nation|state|province|county|city|region|district|firm|company|plant|",
    "school|class|classroom|teacher|student|person|individual|subject|participant|unit|",
    "group|cluster|site|family|household|village|hospital|clinic|team|player|patient|",
    "store|branch|worker|employee|respondent|panel|id|idcode|ident)$",
    "|(^|[._-])(id|ids)$"
  ),
  # names of time variables (used ONLY for the row-unit tag)
  time_name_regex = paste0(
    "^(time|times|year|yr|yrs|years|date|dates|datetime|timestamp|period|periods|",
    "quarter|qtr|month|months|week|weeks|day|days|wave|season|t|trend)$",
    "|(^|[._-])(year|date|time)$"
  ),
  # 2-level character/factor/logical columns are coded 0/1: sorted levels, first -> 0

  # ---- stratification ------------------------------------------------------
  p_bins = list(
    label = c("3-5", "6-10", "11-20", "21-30"),
    lo = c(3L, 6L, 11L, 21L),
    hi = c(5L, 10L, 20L, 30L)
  ),
  n_bins = list(
    label = c("30-199", "200-499", "500-999", "1000-1999"),
    lo = c(30L, 200L, 500L, 1000L),
    hi = c(199L, 499L, 999L, 1999L)
  ),
  n_cap = 1999L, # n > n_cap  -> seeded row subsample to n_cap, top n bin
  p_cap = 30L, # p > p_cap  -> seeded column subsample to p_cap, top p bin
  quota_per_cell = 4L,

  # ---- layout (relative to real_designs/) ------------------------------------
  dir_inputs = "inputs",
  dir_index = "data_raw/index",
  dir_csv = "data_raw/csv",
  dir_desc = "data_raw/description",
  dir_work = "work",
  dir_clean = "data_clean",
  file_selected = "selected_datasets.csv",
  file_exclusions = "exclusions.csv",
  file_manifest = "MANIFEST.csv",
  file_session = "MANIFEST_sessionInfo.txt",
  file_audit = "audit_sheet.md"
)

# Known licenses for packages whose DESCRIPTION is not served by CRAN
# (base-R packages).  Everything else is read from the cached DESCRIPTION
# or reported as "unknown".
CFG$license_overrides <- c(datasets = "Part of R (GPL-2 | GPL-3)")

# Load the shared functions (R/lib_*.R); scripts only source this file.
for (.f in sort(list.files("R", pattern = "[.]R$", full.names = TRUE), method = "radix")) source(.f)
rm(.f)
