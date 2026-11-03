
# ============================================================
# ASSESSING THE GAP — CONDITIONAL BOOTSTRAP ANALYSIS
# ============================================================
# Purpose:
#   Estimate the variability of model discrimination
#   and subgroup performance differences using
#   respondent-level bootstrap resampling.
#
# Research Objective:
#   Assess the stability of weighted AUROC estimates
#   for AI/AN-alone and other identified single-race
#   respondents across the two study periods.
#
# Models:
#   Logistic Regression
#   LASSO Logistic Regression
#   Random Forest
#
# Method:
#   Resample held-out respondents with replacement
#   within each population-by-period group.
#
#   Recalculate weighted AUROC for each replicate.
#   Estimate percentile-based uncertainty intervals.
#   Compare changes in subgroup AUROC gaps over time.
#
# Bootstrap Configuration:
#   100 replicates per model family.
#   Fixed model predictions.
#   Stratified resampling by population and period.
#
# Important:
#   This is a conditional respondent-level bootstrap.
#
#   Models are not retrained during resampling.
#   BRFSS clustering and stratification are not
#   incorporated into the bootstrap procedure.
#
#   Results are exploratory and must not be labeled
#   national complex-survey design-based confidence
#   intervals or interpreted as causal effects.
#
# This script produces:
#
# Conditional bootstrap AUROC intervals
#   outputs/tables/10_conditional_bootstrap_auc.csv
#
# Conditional bootstrap intervals for changes in AUROC gaps
#   outputs/tables/10_conditional_bootstrap_gap_change.csv
#
# Logistic regression bootstrap replicates
#   outputs/checkpoints/bootstrap_logistic.rds
#
# LASSO regression bootstrap replicates
#   outputs/checkpoints/bootstrap_lasso.rds
#
# Random forest bootstrap replicates
#   outputs/checkpoints/bootstrap_forest.rds
# ============================================================


# ============================================================
# 01 — LOAD PROJECT UTILITIES
# ============================================================

# Import shared configuration, evaluation metrics,
# checkpoint management, and output functions.

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

# Verify that data.table is installed.
check_packages(
    c("data.table")
)

# Load data.table without displaying startup messages.
suppressPackageStartupMessages(
    library(data.table)
)


# ============================================================
# 02 — DEFINE BOOTSTRAP CONFIGURATION
# ============================================================

# Specify the number of bootstrap replicates.
#
# Each replicate resamples respondents with replacement
# within the four study population-by-period groups.
#
# The default of 100 replicates is exploratory.
# More replicates would provide more stable percentile
# estimates but require additional computation.

N_BOOT <- 100L


# ============================================================
# 03 — IDENTIFY HELD-OUT PREDICTION FILES
# ============================================================

# Construct the paths to the previously generated
# held-out predictions for all 11 study years.
#
# Each file contains observed FMD outcomes, survey
# weights, population classifications, and predictions
# from all three classification models.

files <- file.path(
    CHECKPOINT_DIR,
    sprintf(
        "heldout_predictions_%d.rds",
        STUDY_YEARS
    )
)


# ============================================================
# 04 — LOAD HELD-OUT EVALUATION DATA
# ============================================================

# Read and combine the annual held-out prediction files.
#
# All three models use the same held-out respondents.
# The saved predictions remain fixed during bootstrapping.

test <- data.table::rbindlist(
    lapply(
        files,
        readRDS
    )
)


# ============================================================
# 05 — DEFINE STUDY PERIODS AND POPULATIONS
# ============================================================

# ------------------------------------------------------------
# 05A — ASSIGN STUDY PERIODS
# ------------------------------------------------------------

# Classify survey years into:
#
# Pre-2020:
#   2014–2019
#
# Post-2020:
#   2020–2024

test[
    ,
    study_period := ifelse(
        YEAR < 2020,
        "Pre-2020",
        "Post-2020"
    )
]


# ------------------------------------------------------------
# 05B — ASSIGN POPULATION GROUPS
# ------------------------------------------------------------

# Convert the AI/AN-alone indicator into the two
# established study population labels.
#
# AI/AN alone:
#   Respondents identifying as AI/AN alone.
#
# Other identified single-race:
#   Respondents classified in other single-race groups.

test[
    ,
    cohort := two_group(
        AIAN_ALONE
    )
]


# ============================================================
# 06 — DEFINE BOOTSTRAP SAMPLING GROUPS
# ============================================================

# Divide held-out observations into four groups:
#
#   AI/AN alone, Pre-2020
#   AI/AN alone, Post-2020
#   Other single-race, Pre-2020
#   Other single-race, Post-2020
#
# Each group is resampled independently.
#
# This preserves the original number of observations
# within each group during every bootstrap replicate.

all_groups <- split(
    seq_len(
        nrow(test)
    ),
    interaction(
        test$study_period,
        test$cohort,
        drop = TRUE
    )
)


# ============================================================
# 07 — VALIDATE BOOTSTRAP GROUPS
# ============================================================

# Confirm that all four expected population-by-period
# combinations contain observations.
#
# Stop execution if any comparison group is missing.

if (length(all_groups) != 4L) {
    
    stop(
        "Expected all four group-by-period test cells."
    )
}


# ============================================================
# 08 — INITIALIZE BOOTSTRAP RESULTS
# ============================================================

# Create a list to store bootstrap replicate results
# for each classification model family.

family_results <- list()


# ============================================================
# 09 — PERFORM CONDITIONAL BOOTSTRAP
# ============================================================

# Perform respondent-level bootstrap resampling
# separately for each model family.
#
# Every bootstrap replicate:
#   Resamples observations with replacement.
#   Retains the original group sample size.
#   Reuses the previously generated predictions.
#   Recalculates weighted AUROC.
#
# Completed model-specific checkpoints are reused.


# ------------------------------------------------------------
# 09A — ITERATE THROUGH MODEL FAMILIES
# ------------------------------------------------------------

for (family in c(
    "logistic",
    "lasso",
    "forest"
)) {
    
    
    # --------------------------------------------------------
    # 09B — DEFINE BOOTSTRAP CHECKPOINT
    # --------------------------------------------------------
    
    # Construct the destination for the current
    # model family's bootstrap results.
    #
    # Example:
    #   bootstrap_logistic.rds
    
    destination <- file.path(
        CHECKPOINT_DIR,
        paste0(
            "bootstrap_",
            family,
            ".rds"
        )
    )
    
    
    # --------------------------------------------------------
    # 09C — REUSE EXISTING BOOTSTRAP RESULTS
    # --------------------------------------------------------
    
    # Skip resampling if a completed checkpoint exists.
    #
    # Existing checkpoints are assumed to correspond
    # to the current predictions and bootstrap settings.
    
    if (file.exists(destination)) {
        
        message(
            "[SKIP] Existing conditional bootstrap for ",
            family
        )
        
        # Load the previously saved bootstrap estimates.
        family_results[[family]] <- readRDS(
            destination
        )
        
        next
    }
    
    
    # --------------------------------------------------------
    # 09D — INITIALIZE REPRODUCIBLE RESAMPLING
    # --------------------------------------------------------
    
    # Set the random seed for reproducibility.
    #
    # The same seed is used for each model family.
    # Because the models share the same held-out sample,
    # this supports corresponding bootstrap draws
    # across model families.
    
    set.seed(
        SEED + 900L
    )
    
    # Create a list for storing replicate results.
    batches <- list()
    
    
    # ========================================================
    # 10 — GENERATE BOOTSTRAP REPLICATES
    # ========================================================
    
    # Repeat the bootstrap procedure N_BOOT times.
    #
    # Each iteration generates one resampled evaluation
    # dataset for every population-by-period group.
    
    for (b in seq_len(N_BOOT)) {
        
        
        # ----------------------------------------------------
        # 10A — INITIALIZE REPLICATE RESULTS
        # ----------------------------------------------------
        
        # Store AUROC estimates for the four
        # population-by-period groups.
        
        vals <- list()
        
        
        # ----------------------------------------------------
        # 10B — ITERATE THROUGH POPULATION GROUPS
        # ----------------------------------------------------
        
        for (name in names(all_groups)) {
            
            # Retrieve original respondent positions
            # for the current population and period.
            
            ix <- all_groups[[name]]
            
            
            # ------------------------------------------------
            # 10C — RESAMPLE WITH REPLACEMENT
            # ------------------------------------------------
            
            # Randomly select respondents with replacement.
            #
            # The resampled group contains the same number
            # of observations as the original group.
            #
            # Some respondents may appear multiple times,
            # while others may not appear in the replicate.
            
            drawn <- ix[
                sample.int(
                    length(ix),
                    length(ix),
                    replace = TRUE
                )
            ]
            
            
            # ------------------------------------------------
            # 10D — EXTRACT RESAMPLED OBSERVATIONS
            # ------------------------------------------------
            
            # Retrieve the resampled respondent records,
            # including fixed predictions and weights.
            
            z <- test[
                drawn
            ]
            
            
            # ------------------------------------------------
            # 10E — CALCULATE WEIGHTED AUROC
            # ------------------------------------------------
            
            # Calculate AUROC for the current bootstrap
            # replicate, population group, and period.
            #
            # The fitted model is not updated or retrained.
            
            vals[[name]] <- data.frame(
                replicate = b,
                
                study_period = z$study_period[1],
                
                cohort = z$cohort[1],
                
                auroc = weighted_auc(
                    z$FMD,
                    z[[
                        paste0(
                            "pred_",
                            family
                        )
                    ]],
                    z$ML_WEIGHT
                )
            )
        }
        
        
        # ----------------------------------------------------
        # 10F — COMBINE REPLICATE RESULTS
        # ----------------------------------------------------
        
        # Combine the four group-specific AUROC
        # estimates for the current replicate.
        
        batches[[b]] <- do.call(
            rbind,
            vals
        )
        
        
        # ----------------------------------------------------
        # 10G — REPORT BOOTSTRAP PROGRESS
        # ----------------------------------------------------
        
        # Print a progress message after every
        # 20 completed bootstrap replicates.
        
        if (b %% 20L == 0) {
            
            message(
                "[BOOT] ",
                family,
                ": ",
                b,
                "/",
                N_BOOT
            )
        }
    }
    
    
    # ========================================================
    # 11 — COMBINE BOOTSTRAP REPLICATES
    # ========================================================
    
    # Combine results from all bootstrap replicates.
    #
    # Expected structure:
    #   One row per replicate and population-period group.
    
    values <- do.call(
        rbind,
        batches
    )
    
    
    # ========================================================
    # 12 — SAVE MODEL BOOTSTRAP CHECKPOINT
    # ========================================================
    
    # Save the completed replicate estimates.
    #
    # This allows future runs to reuse the results
    # without repeating the bootstrap procedure.
    
    save_checkpoint(
        values,
        destination
    )
    
    
    # --------------------------------------------------------
    # 12A — STORE MODEL BOOTSTRAP RESULTS
    # --------------------------------------------------------
    
    # Retain the model's bootstrap results for
    # subsequent interval calculations.
    
    family_results[[family]] <- values
}


# ============================================================
# 13 — INITIALIZE UNCERTAINTY SUMMARIES
# ============================================================

# Create separate lists for:
#
# intervals:
#   AUROC uncertainty intervals for individual groups.
#
# gap_changes:
#   Uncertainty intervals for the pre/post change
#   in the AI/AN-minus-comparison AUROC gap.

intervals <- list()

gap_changes <- list()


# ============================================================
# 14 — CALCULATE BOOTSTRAP AUROC INTERVALS
# ============================================================

# Summarize the distribution of AUROC estimates
# across bootstrap replicates.
#
# Method:
#   Use the 2.5th and 97.5th percentiles.
#
# These are conditional bootstrap percentile intervals.
#
# They do not account for model retraining or
# the full complex BRFSS sampling design.


# ------------------------------------------------------------
# 14A — ITERATE THROUGH MODEL FAMILIES
# ------------------------------------------------------------

for (family in names(family_results)) {
    
    # Retrieve the current model's bootstrap results.
    b <- family_results[[family]]
    
    
    # --------------------------------------------------------
    # 14B — ITERATE THROUGH STUDY PERIODS
    # --------------------------------------------------------
    
    for (period in PERIOD_LEVELS) {
        
        
        # ----------------------------------------------------
        # 14C — ITERATE THROUGH POPULATION GROUPS
        # ----------------------------------------------------
        
        for (group in GROUP_LEVELS) {
            
            # Select AUROC estimates for the current
            # model, period, and population group.
            
            z <- b$auroc[
                b$study_period == period &
                    b$cohort == group
            ]
            
            
            # ------------------------------------------------
            # 14D — CALCULATE PERCENTILE INTERVAL
            # ------------------------------------------------
            
            # Estimate lower and upper percentile bounds.
            #
            # Lower:
            #   2.5th percentile.
            #
            # Upper:
            #   97.5th percentile.
            #
            # The interval reflects variability under
            # the defined respondent-resampling procedure.
            
            intervals[[
                length(intervals) + 1L
            ]] <- data.frame(
                family = family,
                
                study_period = period,
                
                cohort = group,
                
                lower_95 = unname(
                    stats::quantile(
                        z,
                        0.025,
                        na.rm = TRUE
                    )
                ),
                
                upper_95 = unname(
                    stats::quantile(
                        z,
                        0.975,
                        na.rm = TRUE
                    )
                ),
                
                interval_type = paste(
                    "conditional respondent bootstrap",
                    "(not survey-design CI)"
                )
            )
        }
    }
    
    
    # ========================================================
    # 15 — RESHAPE BOOTSTRAP AUROC RESULTS
    # ========================================================
    
    # Reshape model-specific bootstrap estimates so
    # each replicate occupies one row.
    #
    # Columns represent the four population-by-period
    # combinations.
    #
    # This allows the subgroup performance gap
    # to be calculated separately for each replicate.
    
    wide <- data.table::dcast(
        data.table::as.data.table(
            b
        ),
        
        replicate ~ study_period + cohort,
        
        value.var = "auroc"
    )
    
    
    # ========================================================
    # 16 — IDENTIFY GROUP-PERIOD COLUMNS
    # ========================================================
    
    # Define a helper function to construct
    # the expected column names created by dcast().
    #
    # Example:
    #   Pre-2020_AI/AN alone
    
    colname <- function(period, group) {
        
        return(
            paste(
                period,
                group,
                sep = "_"
            )
        )
    }
    
    
    # --------------------------------------------------------
    # 16A — DEFINE REQUIRED COMPARISON COLUMNS
    # --------------------------------------------------------
    
    # The four columns are ordered as:
    #
    #   1. Pre-2020, AI/AN alone.
    #   2. Pre-2020, other identified single-race.
    #   3. Post-2020, AI/AN alone.
    #   4. Post-2020, other identified single-race.
    
    needed <- c(
        colname(
            "Pre-2020",
            GROUP_LEVELS[1]
        ),
        
        colname(
            "Pre-2020",
            GROUP_LEVELS[2]
        ),
        
        colname(
            "Post-2020",
            GROUP_LEVELS[1]
        ),
        
        colname(
            "Post-2020",
            GROUP_LEVELS[2]
        )
    )
    
    
    # --------------------------------------------------------
    # 16B — VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------
    
    # Confirm that every population-by-period
    # combination is represented in the reshaped data.
    
    if (!all(needed %in% names(wide))) {
        
        stop(
            "Bootstrap group-period columns not found."
        )
    }
    
    
    # ========================================================
    # 17 — CALCULATE CHANGE IN AUROC GAP
    # ========================================================
    
    # Calculate subgroup AUROC differences within
    # each study period.
    #
    # Subgroup gap:
    #   AI/AN-alone AUROC minus comparison-group AUROC.
    #
    # Change in gap:
    #   Post-2020 subgroup gap minus Pre-2020 subgroup gap.
    #
    # Negative values indicate the subgroup difference
    # became more negative after 2020.
    #
    # This is a descriptive temporal comparison,
    # not evidence of a causal effect of 2020.
    
    shift <- (
        wide[[needed[3]]] -
            wide[[needed[4]]]
    ) - (
        wide[[needed[1]]] -
            wide[[needed[2]]]
    )
    
    
    # ========================================================
    # 18 — ESTIMATE UNCERTAINTY IN GAP CHANGES
    # ========================================================
    
    # Calculate the percentile interval for the
    # change in subgroup AUROC gap.
    #
    # Each bootstrap replicate contributes one
    # paired difference-in-gaps estimate.
    #
    # The resulting interval is conditional on
    # the fixed predictions and evaluation dataset.
    
    gap_changes[[family]] <- data.frame(
        family = family,
        
        lower_95 = unname(
            stats::quantile(
                shift,
                0.025,
                na.rm = TRUE
            )
        ),
        
        upper_95 = unname(
            stats::quantile(
                shift,
                0.975,
                na.rm = TRUE
            )
        ),
        
        replicates = length(
            shift
        ),
        
        interval_type = paste(
            "conditional respondent bootstrap",
            "(not survey-design CI)"
        )
    )
}


# ============================================================
# 19 — SAVE BOOTSTRAP AUROC INTERVALS
# ============================================================

# Combine the model-specific and population-specific
# AUROC percentile intervals into one results table.

save_csv(
    do.call(
        rbind,
        intervals
    ),
    "10_conditional_bootstrap_auc.csv"
)


# ============================================================
# 20 — SAVE BOOTSTRAP GAP-CHANGE INTERVALS
# ============================================================

# Combine uncertainty intervals for the changes
# in subgroup AUROC gaps across the three models.

save_csv(
    do.call(
        rbind,
        gap_changes
    ),
    "10_conditional_bootstrap_gap_change.csv"
)


# ============================================================
# 21 — REPORT METHODOLOGICAL LIMITATIONS
# ============================================================

# Clarify that these are conditional bootstrap results.
#
# Sources of uncertainty not incorporated:
#   Retraining models on new datasets.
#   The BRFSS complex sampling design.
#   Survey clustering and stratification.
#
# Interpret results as exploratory uncertainty estimates.

cat(
    "[CAUTION] Conditional bootstrap only. ",
    "No model-refit or\n",
    "complex-survey design uncertainty is incorporated.\n",
    sep = ""
)


# ============================================================
# 22 — REPORT COMPLETION
# ============================================================

message(
    "[COMPLETE] Conditional bootstrap results saved."
)
