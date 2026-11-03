# ============================================================
# ASSESSING THE GAP — POOLED DATASET VERIFICATION
# ============================================================
# Purpose:
#   Verify the structure, completeness, and internal
#   consistency of the processed BRFSS 2014–2024 dataset.
#
# Dataset:
#   analysis_pooled_2014_2024.rds
#
# Expected:
#   4,790,742 respondents
#   25 variables
#   11 survey years (2014–2024)
#
#   This script uses the pre-processed pooled dataset.
# ============================================================


# ============================================================
# 01 — INITIALIZE PROJECT UTILITIES
# ============================================================

# Load shared configuration, validation, and output functions.
source(
    file.path(
        "R",
        "01_utilities.R"
    )
)

# Verify the required R package is installed.
check_packages(
    c("data.table")
)

# Load data.table without printing startup messages.
suppressPackageStartupMessages(
    library(data.table)
)

# Load the previously constructed pooled BRFSS dataset.
d <- load_data()


# ============================================================
# 02 — VERIFY DATASET DIMENSIONS
# ============================================================

# Confirm the dataset contains the expected respondent count.
if (nrow(d) != 4790742L) {
    
    warning(
        "Row count differs from the previously ",
        "verified 4,790,742."
    )
}

# Confirm the dataset contains the expected number of variables.
if (ncol(d) != 25L) {
    
    warning(
        "Column count differs from the previously ",
        "verified 25."
    )
}


# ============================================================
# 03 — SUMMARIZE ANNUAL DATA COMPLETENESS
# ============================================================

# Calculate annual respondent counts and data availability.
#
# Measures:
#   respondents             = Total respondents per year
#   aian_alone              = AI/AN-alone respondents
#   comparison              = Other identified single-race
#   multiracial_or_unknown  = Excluded race classifications
#   valid_fmd               = Respondents with valid FMD values
#   missing_fmd             = Missing FMD responses
#   complete_model          = Complete predictor/outcome records

annual <- d[
    ,
    .(
        respondents = .N,
        
        aian_alone = sum(
            AIAN_ALONE %in% TRUE
        ),
        
        comparison = sum(
            AIAN_ALONE %in% FALSE
        ),
        
        multiracial_or_unknown = sum(
            is.na(AIAN_ALONE)
        ),
        
        valid_fmd = sum(
            FMD %in% c(0, 1)
        ),
        
        missing_fmd = sum(
            is.na(FMD)
        ),
        
        complete_model = sum(
            COMPLETE_MODEL %in% TRUE
        )
    ),
    by = YEAR
][order(YEAR)]


# ============================================================
# 04 — VALIDATE ANNUAL RECORD COUNTS
# ============================================================

# Confirm that all 11 study years are represented.
if (nrow(annual) != length(STUDY_YEARS)) {
    
    stop(
        "One or more annual files are absent ",
        "from the pooled data."
    )
}

# Verify that annual respondent counts reconcile
# with the full pooled dataset.
if (sum(annual$respondents) != nrow(d)) {
    
    stop(
        "The annual row totals do not reconcile."
    )
}


# ============================================================
# 05 — DISPLAY AND SAVE ANNUAL SUMMARY
# ============================================================

cat("\n[ANNUAL DATASET AUDIT]\n")

print(annual)

# Save the annual summary for research documentation.
save_csv(
    annual,
    "01_annual_input_audit.csv"
)


# ============================================================
# 06 — VERIFY FREQUENT MENTAL DISTRESS CODING
# ============================================================

# Reconstruct the FMD outcome independently using
# the original BRFSS MENTHLTH response codes.
#
# BRFSS response coding:
#   1–30 = Number of mentally unhealthy days
#   88   = None (equivalent to zero days)
#   77   = Don't know / Not sure
#   99   = Refused
#
# FMD definition:
#   1  = 14–30 mentally unhealthy days
#   0  = 0–13 mentally unhealthy days
#   NA = Missing, refused, or unknown response

# Convert original responses into mentally unhealthy days.
expected_days <- ifelse(
    d$MENTHLTH_RAW %in% 1:30,
    d$MENTHLTH_RAW,
    ifelse(
        d$MENTHLTH_RAW == 88,
        0,
        NA_real_
    )
)

# Independently reconstruct the binary FMD outcome.
expected_fmd <- ifelse(
    is.na(expected_days),
    NA_integer_,
    as.integer(expected_days >= 14)
)

# Count discrepancies between the reconstructed outcome
# and the FMD variable stored in the pooled dataset.
fmd_mismatch <- sum(
    !(is.na(expected_fmd) & is.na(d$FMD)) &
        (
            is.na(expected_fmd) |
                is.na(d$FMD) |
                expected_fmd != d$FMD
        ),
    na.rm = TRUE
)


# ============================================================
# 07 — VERIFY AI/AN-ALONE CLASSIFICATION
# ============================================================

# Reconstruct the AI/AN-alone indicator from the
# harmonized race-alone variable.
#
# Expected classification:
#   Code 3     = AI/AN alone
#   Codes 1–6  = Identified single-race categories
#   Other      = Excluded / unclassified
#
# This check does not identify multiracial respondents
# who may report AI/AN alongside another race.

race_expected <- ifelse(
    d$RACE_ALONE_RAW %in% 1:6,
    d$RACE_ALONE_RAW == 3,
    NA
)

# Count inconsistencies between the reconstructed
# classification and the saved AIAN_ALONE indicator.
race_mismatch <- sum(
    !(is.na(race_expected) & is.na(d$AIAN_ALONE)) &
        (
            is.na(race_expected) |
                is.na(d$AIAN_ALONE) |
                race_expected != d$AIAN_ALONE
        ),
    na.rm = TRUE
)


# ============================================================
# 08 — VERIFY STORED POOLED WEIGHTS
# ============================================================

# Confirm that POOL_WEIGHT matches the historical
# weighting formula used when constructing the dataset.
#
# Formula:
#   POOL_WEIGHT = RAW_WEIGHT × (annual_n / pooled_n)
#
# This is an internal consistency check.
# It does not establish that the formula is the uniquely
# appropriate method for multi-year BRFSS inference.

# Calculate respondent counts for each survey year.
n_by_year <- table(
    as.character(d$YEAR)
)

# Calculate each year's proportion of the pooled sample.
ratio <- as.numeric(
    n_by_year[
        as.character(d$YEAR)
    ]
) / nrow(d)

# Compare the stored weights against the expected formula.
weight_error <- max(
    abs(
        d$POOL_WEIGHT -
            d$RAW_WEIGHT * ratio
    ),
    na.rm = TRUE
)


# ============================================================
# 09 — COMPILE INTERNAL CONSISTENCY RESULTS
# ============================================================

# Create a summary table of discrepancies detected
# by the three independent checks.

checks <- data.frame(
    check = c(
        "FMD coding from MENTHLTH_RAW",
        "AI/AN-alone race recoding",
        "Stored historical pooled-weight formula"
    ),
    
    discrepancies = c(
        fmd_mismatch,
        race_mismatch,
        as.integer(weight_error > 1e-6)
    ),
    
    notes = c(
        "Independent raw-code rule (88=zero, 77/99=missing)",
        "Checks mapped race-alone flag, not race combinations",
        paste(
            "Maximum absolute numeric discrepancy:",
            signif(weight_error, 6)
        )
    )
)


# ============================================================
# 10 — DISPLAY AND SAVE VALIDATION RESULTS
# ============================================================

cat("\n[INTERNAL CONSISTENCY CHECKS]\n")

print(
    checks,
    row.names = FALSE
)

save_csv(
    checks,
    "01_internal_consistency.csv"
)


# ============================================================
# 11 — VALIDATE CODING RESULTS
# ============================================================

# Stop analysis if the FMD or race classifications
# do not match their expected internal coding rules.
#
# Investigate discrepancies before fitting models.

if (
    fmd_mismatch > 0 ||
    race_mismatch > 0
) {
    
    stop(
        "An internal coding check failed. ",
        "Inspect the audit before modeling."
    )
}


# ============================================================
# 12 — CONFIRM METADATA AVAILABILITY
# ============================================================

# Check whether the dataset construction metadata
# is available for documentation and reproducibility.

if (file.exists(META_FILE)) {
    
    message(
        "[METADATA] Pooled metadata file is present."
    )
}


# ============================================================
# 13 — FINAL VERIFICATION SUMMARY
# ============================================================

cat(
    "\n[COMPLETE] Provided pooled dataset audited.\n"
)

cat(
    "[LIMITATION] Original XPT extraction and 2024 ",
    "CDC derived-category discrepancies require ",
    "a separate source-level audit.\n",
    sep = ""
)


# ============================================================
# 14 — RELEASE MEMORY
# ============================================================

# Remove the large pooled dataset from the workspace.
# This frees memory before subsequent analyses.

rm(d)
# Request garbage collection for unused memory.
gc(
    verbose = FALSE
)
