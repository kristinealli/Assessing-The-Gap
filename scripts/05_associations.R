
# ============================================================
# ASSESSING THE GAP — EXPLORATORY ADJUSTED ASSOCIATIONS
# ============================================================
# Purpose:
#   Examine relationships between frequent mental distress
#   (FMD) and demographic, socioeconomic, and health factors.
#
# Research Question:
#   Which characteristics are associated with FMD, and how
#   do selected associations differ across population groups
#   and survey periods?
#
# Method:
#   Weighted multivariable logistic regression with
#   interaction terms and model-standardized predictions.
#
# Limitations:
#   Results describe associations, not causation.
#   BRFSS design-based standard errors, confidence intervals,
#   and statistical significance tests are not calculated.
#
# This script produces: 
# Saved weighted logistic regression model
#   outputs/models/associations_exploratory.rds	
#
# Model coefficients and odds ratios
#   outputs/tables/05_association_coefficients.csv
#
# Adjusted FMD predictions for each group and period
#   outputs/tables/05_adjusted_predicted_probabilities.csv
# ============================================================


# ============================================================
# 01 — LOAD PROJECT UTILITIES
# ============================================================

# Import shared configuration, weighting, and output functions.
source(
    file.path(
        "R",
        "01_utilities.R"
    )
)

# Verify the required package is installed.
check_packages(
    c("data.table")
)

# Load the prepared training dataset.
training <- readRDS(
    file.path(
        CHECKPOINT_DIR,
        "training_sample.rds"
    )
)


# ============================================================
# 02 — DEFINE STUDY POPULATIONS
# ============================================================

# Assign readable population labels.
#
# The comparison group is the reference category:
#   1. Other identified single-race
#   2. AI/AN alone

training$cohort <- factor(
    two_group(
        training$AIAN_ALONE
    ),
    levels = rev(
        GROUP_LEVELS
    )
)


# ============================================================
# 03 — DEFINE REGRESSION MODEL
# ============================================================

# Specify the binary FMD outcome and predictor variables.
#
# Interaction terms:
#   cohort * PERIOD:
#     Differences in period associations by population group.
#
#   PHYSHLTH_DAYS * cohort:
#     Differences in physical-health associations by group.
#
#   GENHLTH_CLEAN * PERIOD:
#     Differences in general-health associations by period.
#
# Additional predictors:
#   Age, sex, education, and household income.
#
# In R, A * B includes A, B, and their interaction A:B.

formula <- FMD ~
    cohort * PERIOD +
    PHYSHLTH_DAYS * cohort +
    GENHLTH_CLEAN * PERIOD +
    AGE_BAND +
    SEX_CLEAN +
    EDU_CLEAN +
    INCOME_BAND


# ============================================================
# 04 — PREPARE MODEL WEIGHTS
# ============================================================

# Normalize the computationally adjusted training weights.
#
# Normalization preserves relative respondent weights
# while rescaling the average weight to 1.

training$MODEL_WEIGHT <- model_weight(
    training
)


# ============================================================
# 05 — FIT WEIGHTED LOGISTIC REGRESSION
# ============================================================

# Fit the exploratory association model.
#
# quasibinomial():
#   Models binary FMD outcomes using a logit link.
#
# MODEL_WEIGHT:
#   Applies normalized training weights.
#
# maxit = 35:
#   Maximum number of fitting iterations.
#
# This is not a full complex-survey regression model.

fit <- stats::glm(
    formula = formula,
    data = training,
    weights = MODEL_WEIGHT,
    family = stats::quasibinomial(),
    control = stats::glm.control(
        maxit = 35
    )
)


# ============================================================
# 06 — VALIDATE MODEL CONVERGENCE
# ============================================================

# Stop execution if the regression fails to converge.
if (!isTRUE(fit$converged)) {
    
    stop(
        "Association model did not converge."
    )
}


# ============================================================
# 07 — EXTRACT REGRESSION COEFFICIENTS
# ============================================================

# Create a summary of fitted model coefficients.
#
# term:
#   Predictor or interaction term.
#
# log_odds:
#   Estimated coefficient on the log-odds scale.
#
# odds_ratio:
#   Exponentiated coefficient (exp(beta)).
#
# uncertainty:
#   Documents the absence of survey-adjusted
#   uncertainty estimates.
#
# Interaction terms must be interpreted alongside
# their associated main effects.

coefficient_table <- data.frame(
    term = names(
        stats::coef(fit)
    ),
    
    log_odds = unname(
        stats::coef(fit)
    ),
    
    odds_ratio = exp(
        unname(
            stats::coef(fit)
        )
    ),
    
    uncertainty = paste(
        "Descriptive only:",
        "survey-adjusted SEs not estimated"
    )
)


# ============================================================
# 08 — INITIALIZE STANDARDIZED PREDICTIONS
# ============================================================

# Store adjusted predicted probabilities for each
# population-group and study-period combination.
#
# Model standardization applies the fitted regression
# to a common distribution of the other predictors.
#
# These results are model-based probabilities,
# not directly observed FMD prevalence estimates.

standardized <- list()


# ============================================================
# 09 — CALCULATE STANDARDIZED PREDICTIONS
# ============================================================

# Generate predictions for four combinations:
#   Other single-race, Pre-2020.
#   Other single-race, Post-2020.
#   AI/AN alone, Pre-2020.
#   AI/AN alone, Post-2020.
#
# For each combination, the original distributions of
# age, sex, education, income, and health predictors
# are preserved.


# ------------------------------------------------------------
# 09A — ITERATE THROUGH POPULATION GROUPS
# ------------------------------------------------------------

for (cohort in levels(training$cohort)) {
    
    
    # --------------------------------------------------------
    # 09B — ITERATE THROUGH SURVEY PERIODS
    # --------------------------------------------------------
    
    for (period in PERIOD_LEVELS) {
        
        # Create a copy of the training dataset.
        synthetic <- training
        
        
        # ----------------------------------------------------
        # 09C — ASSIGN THE POPULATION GROUP
        # ----------------------------------------------------
        
        # Set every synthetic respondent to the current
        # population group while retaining factor levels.
        synthetic$cohort <- factor(
            cohort,
            levels = levels(
                training$cohort
            )
        )
        
        
        # ----------------------------------------------------
        # 09D — ASSIGN THE SURVEY PERIOD
        # ----------------------------------------------------
        
        # Set every synthetic respondent to the current
        # study period while retaining factor levels.
        synthetic$PERIOD <- factor(
            period,
            levels = PERIOD_LEVELS
        )
        
        
        # ----------------------------------------------------
        # 09E — GENERATE PREDICTED PROBABILITIES
        # ----------------------------------------------------
        
        # Predict each synthetic respondent's probability
        # of experiencing frequent mental distress.
        #
        # type = "response" returns probabilities
        # rather than log-odds.
        
        p <- stats::predict(
            fit,
            newdata = synthetic,
            type = "response"
        )
        
        
        # ----------------------------------------------------
        # 09F — CALCULATE WEIGHTED MEAN PREDICTION
        # ----------------------------------------------------
        
        # Calculate the standardized mean probability.
        #
        # Multiply by 100 to express the estimate
        # as a percentage.
        #
        # Store the result for the current group and period.
        
        standardized[[
            length(standardized) + 1L
        ]] <- data.frame(
            cohort = cohort,
            
            study_period = period,
            
            adjusted_prediction_pct = 100 *
                stats::weighted.mean(
                    p,
                    training$MODEL_WEIGHT
                )
        )
    }
}


# ============================================================
# 10 — SAVE FITTED MODEL
# ============================================================

# Save the exploratory regression model as an RDS file
# for future inspection and reproducibility.

save_checkpoint(
    fit,
    file.path(
        MODEL_DIR,
        "associations_exploratory.rds"
    )
)


# ============================================================
# 11 — SAVE REGRESSION COEFFICIENTS
# ============================================================

# Export estimated log-odds and odds ratios.
save_csv(
    coefficient_table,
    "05_association_coefficients.csv"
)


# ============================================================
# 12 — SAVE STANDARDIZED PREDICTIONS
# ============================================================

# Combine all group-period predictions into one table.
# Export the model-adjusted FMD probabilities.

save_csv(
    data.table::rbindlist(
        standardized
    ),
    "05_adjusted_predicted_probabilities.csv"
)


# ============================================================
# 13 — REPORT INTERPRETATION AND LIMITATIONS
# ============================================================

# Clarify that coefficients represent associations,
# not evidence of causal relationships.

cat(
    "[INTERPRETATION] Coefficients represent ",
    "conditional associations.\n",
    sep = ""
)

# Clarify that full survey-design uncertainty
# calculations have not been performed.

cat(
    "[CAUTION] No BRFSS design-based standard errors; ",
    "do not infer\n",
    "causal changes or statistically significant ",
    "subgroup differences.\n",
    sep = ""
)


# ============================================================
# 14 — RELEASE MEMORY
# ============================================================

# Remove datasets no longer required after saving results.
rm(
    training,
    synthetic
)


# ============================================================
# 15 — REPORT COMPLETION
# ============================================================

message(
    "[COMPLETE] Exploratory associations saved."
)
