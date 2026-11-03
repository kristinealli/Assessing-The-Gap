# ============================================================
# ASSESSING THE GAP — BASIC TESTS
# ============================================================
# Purpose:
#   Verify that the model evaluation functions and feature
#   preparation utilities behave as expected.
#
# Tests:
#   Weighted AUROC
#   Classification Metrics
#   Predictor Normalization
#   Machine-Learning Design Matrix
#
# Usage:
#   Run from the RStudio project root.
#
# Note:
#   Lightweight smoke tests using synthetic data.
#   They do not require loading the pooled BRFSS dataset
#   or fitting any machine-learning models.
# ============================================================


# ============================================================
# 01 — LOAD PROJECT UTILITIES
# ============================================================

# Import shared configuration, data preparation functions,
# and model evaluation metrics.

source(
    file.path(
        "R",
        "01_utilities.R"
    )
)

source(
    file.path(
        "R",
        "02_metrics.R"
    )
)


# ============================================================
# 02 — CREATE SYNTHETIC TEST DATA
# ============================================================

# Create four observations with known binary outcomes.
#
# Outcome:
#   0 = No frequent mental distress
#   1 = Frequent mental distress
#
# Predicted probabilities are deliberately arranged so
# positive cases receive higher scores than negative cases.

y <- c(
    0,
    0,
    1,
    1
)

p <- c(
    0.1,
    0.2,
    0.8,
    0.9
)

# Assign equal weights to all four observations.
w <- c(
    1,
    1,
    1,
    1
)


# ============================================================
# 03 — TEST WEIGHTED AUROC
# ============================================================

# Confirm perfect discrimination.
#
# Expected:
#   AUROC = 1.0
#
# Positive cases have higher predicted probabilities
# than every negative case.

stopifnot(
    abs(
        weighted_auc(
            y,
            p,
            w
        ) - 1
    ) < 1e-10
)

# Confirm completely reversed discrimination.
#
# Expected:
#   AUROC = 0.0
#
# Reversing predicted probabilities places positive
# cases below negative cases in the ranking.

stopifnot(
    abs(
        weighted_auc(
            y,
            1 - p,
            w
        ) - 0
    ) < 1e-10
)


# ============================================================
# 04 — TEST CLASSIFICATION METRICS
# ============================================================

# Evaluate the synthetic predictions using a threshold
# of 0.50 rather than the project's default threshold.
#
# Expected:
#   Accuracy    = 1.0
#   Sensitivity = 1.0
#   Specificity = 1.0
#   Respondents = 4

m <- score_metrics(
    y,
    p,
    w,
    threshold = 0.5
)

# Verify that all four cases were correctly classified.
stopifnot(
    m$accuracy == 1,
    m$sensitivity == 1,
    m$specificity == 1,
    m$n == 4L
)


# ============================================================
# 05 — CREATE SYNTHETIC PREDICTOR DATA
# ============================================================

# Construct a small dataset containing the same
# predictor variables used in the primary models.
#
# The two observations intentionally differ across
# health, demographic, socioeconomic, and period values.
#
# This tests feature preparation without requiring
# the full BRFSS dataset.

toy <- data.frame(
    PHYSHLTH_DAYS = c(
        0,
        10
    ),
    
    GENHLTH_CLEAN = c(
        1,
        5
    ),
    
    AGE_BAND = c(
        "18-34",
        "65+"
    ),
    
    SEX_CLEAN = c(
        "Female",
        "Male"
    ),
    
    EDU_CLEAN = c(
        "HS",
        "College grad"
    ),
    
    INCOME_BAND = c(
        "<25k",
        "75k+"
    ),
    
    PERIOD = c(
        "Pre-2020",
        "Post-2020"
    )
)


# ============================================================
# 06 — TEST PREDICTOR NORMALIZATION
# ============================================================

# Apply consistent factor levels and numeric data types.
#
# This ensures categorical variables use the expected
# definitions before generating a design matrix.

toy <- normalize_features(
    toy
)


# ============================================================
# 07 — TEST MACHINE-LEARNING DESIGN MATRIX
# ============================================================

# Convert the prepared predictor variables into a
# numeric design matrix.
#
# Categorical predictors are encoded into indicator
# columns using the shared DESIGN_FORMULA.

X <- make_matrix(
    toy
)

# Verify that:
#   The matrix contains two observations.
#   No missing values appear in the resulting matrix.

stopifnot(
    nrow(X) == 2,
    !anyNA(X)
)


# ============================================================
# 08 — REPORT TEST RESULTS
# ============================================================

# If all assertions succeed, report successful completion.
#
# stopifnot() automatically stops execution if any
# condition evaluates to FALSE.

cat(
    "[PASS] Basic metric and design-matrix smoke tests.\n"
)

