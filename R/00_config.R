# ============================================================
# ASSESSING THE GAP — SHARED SETTINGS
# ============================================================
# Purpose:
#   Define project paths, research variables, model parameters,
#   and reproducibility settings used throughout the analysis.
#
# Usage:
#   Source this file from the RStudio project root.
#
# Note:
#   Changing analytical settings may invalidate existing
#   outputs and checkpoints. Archive or clear affected results
#   before rerunning dependent scripts.
# ============================================================


# ============================================================
# 01 — PROJECT DIRECTORIES
# ============================================================

# Identify the current working directory as the project root.
PROJECT_ROOT <- normalizePath(
    getwd(),
    mustWork = TRUE
)

# Define the path to the processed BRFSS dataset.
# This file contains the pooled 2014–2024 analytical data.
DATA_FILE <- file.path(
    PROJECT_ROOT,
    "data",
    "processed",
    "analysis_pooled_2014_2024.rds"
)

# Define the path to the dataset metadata.
# Metadata documents the pooled dataset's construction,
# source files, processing configuration, and record count.
META_FILE <- file.path(
    PROJECT_ROOT,
    "data",
    "processed",
    "analysis_pooled_2014_2024.meta.rds"
)

# Define directories for analysis outputs.
OUT_DIR <- file.path(
    PROJECT_ROOT,
    "outputs"
)

TABLE_DIR <- file.path(
    OUT_DIR,
    "tables"
)

FIGURE_DIR <- file.path(
    OUT_DIR,
    "figures"
)

MODEL_DIR <- file.path(
    OUT_DIR,
    "models"
)

CHECKPOINT_DIR <- file.path(
    OUT_DIR,
    "checkpoints"
)

# Create output directories if they do not already exist.
for (folder in c(
    OUT_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR
)) {
    dir.create(
        folder,
        recursive = TRUE,
        showWarnings = FALSE
    )
}


# ============================================================
# 02 — STUDY POPULATION AND PERIODS
# ============================================================

# Define the 11 BRFSS survey years included in the study.
STUDY_YEARS <- 2014:2024

# Define the two periods used for temporal comparisons.
PERIOD_LEVELS <- c(
    "Pre-2020",
    "Post-2020"
)

# Define the two primary population groups.
#
# AI/AN alone:
#   Respondents classified as American Indian or
#   Alaska Native alone.
#
# Other identified single-race:
#   Respondents classified in other single-race categories.
#
# Multiracial and unknown-race respondents are excluded
# from the primary two-group comparison.
GROUP_LEVELS <- c(
    "AI/AN alone",
    "Other identified single-race"
)


# ============================================================
# 03 — PREDICTOR VARIABLES
# ============================================================

# Define the demographic, socioeconomic, physical-health,
# and temporal predictors used in the classification models.
FEATURES <- c(
    "PHYSHLTH_DAYS",
    "GENHLTH_CLEAN",
    "AGE_BAND",
    "SEX_CLEAN",
    "EDU_CLEAN",
    "INCOME_BAND",
    "PERIOD"
)


# ============================================================
# 04 — MODEL FORMULAS
# ============================================================

# Variable definitions:
#
# FMD:
#   Frequent mental distress, defined as 14 or more
#   mentally unhealthy days in the previous 30 days.
#
# PHYSHLTH_DAYS:
#   Number of physically unhealthy days.
#
# GENHLTH_CLEAN:
#   Self-rated general health.
#
# AGE_BAND:
#   Respondent age category.
#
# SEX_CLEAN:
#   Sex recorded in BRFSS.
#
# EDU_CLEAN:
#   Educational attainment.
#
# INCOME_BAND:
#   Household income category.
#
# PERIOD:
#   Pre-2020 or post-2020 survey period.


# ------------------------------------------------------------
# 04A — OUTCOME AND PREDICTOR FORMULA
# ------------------------------------------------------------

# Define FMD as the binary outcome and the seven variables
# used to estimate the probability of frequent mental distress.
#
# Used for logistic regression and related analyses.

MODEL_FORMULA <- FMD ~
    PHYSHLTH_DAYS +
    GENHLTH_CLEAN +
    AGE_BAND +
    SEX_CLEAN +
    EDU_CLEAN +
    INCOME_BAND +
    PERIOD


# ------------------------------------------------------------
# 04B — PREDICTOR-ONLY FORMULA
# ------------------------------------------------------------

# Define the same predictors without the FMD outcome.
#
# Used with model.matrix() to convert categorical variables
# into numerical indicator columns for machine learning.
#
# This is not the complex-survey sampling design formula.

DESIGN_FORMULA <- ~
    PHYSHLTH_DAYS +
    GENHLTH_CLEAN +
    AGE_BAND +
    SEX_CLEAN +
    EDU_CLEAN +
    INCOME_BAND +
    PERIOD


# ============================================================
# 05 — REPRODUCIBILITY AND DATA SPLITTING
# ============================================================

# Set the random seed to make random sampling and model
# initialization reproducible.
SEED <- 42L

# Reserve 20% of eligible respondents for held-out testing.
TEST_FRACTION <- 0.20

# Limit the comparison-group training sample to 90,000 rows.
OTHER_TRAIN_CAP <- 90000L

# Limit comparison-group test samples to 60,000 rows per year.
OTHER_TEST_CAP_PER_YEAR <- 60000L

# Use three folds for model cross-validation.
CV_FOLDS <- 3L

# Limit the PCA sample to 12,000 observations.
PCA_MAX_ROWS <- 12000L


# ============================================================
# 06 — RANDOM FOREST CONFIGURATION
# ============================================================

# Set the number of decision trees in the random forest.
RF_TREES <- 80L

# Limit parallel processing to two threads.
RF_THREADS <- 2L

# Define candidate numbers of predictors considered
# at each tree split during model tuning.
RF_MTRY_GRID <- c(
    2L,
    4L
)


# ============================================================
# 07 — LASSO CONFIGURATION
# ============================================================

# Define candidate regularization strengths (lambda).
#
# Smaller values apply less coefficient shrinkage.
# Larger values apply more coefficient shrinkage.
#
# The selected lambda is determined during model evaluation.

LAMBDA_GRID <- 10 ^ seq(
    -3.5,
    -1,
    length.out = 7
)


# ============================================================
# 08 — CLASSIFICATION THRESHOLD
# ============================================================

# Define the probability cutoff used to classify FMD.
#
# Predicted probability >= 0.20:
#   Classify as FMD present (1).
#
# Predicted probability < 0.20:
#   Classify as FMD absent (0).
#
# Threshold selection affects sensitivity and specificity
# but does not change AUROC.

CLASSIFICATION_THRESHOLD <- 0.20


# ============================================================
# 09 — SURVEY WEIGHT CONFIGURATION
# ============================================================

# Use the stored pooled weights for weighted analyses.
#
# POOL_WEIGHT reproduces the weighting approach used
# when the historical BRFSS dataset was constructed.
#
# Alternative multi-year weighting methods should be
# examined through sensitivity analyses.
#
# This setting does not establish that the historical
# formula is uniquely appropriate for population inference.

PRIMARY_WEIGHT <- "POOL_WEIGHT"


# ============================================================
# 10 — CHECKPOINT AND REPRODUCIBILITY
# ============================================================

# Training, testing, and cross-validation use consistent
# respondent selection rules and random seed settings.
#
# Saved checkpoints allow completed stages to be reused
# without repeating computationally expensive operations.
#
# If predictor definitions, sample selection, model parameters,
# or weighting strategies change, archive or clear affected
# checkpoints before rerunning the corresponding analyses.
