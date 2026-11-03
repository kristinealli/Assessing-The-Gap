
# ============================================================
# ASSESSING THE GAP — HELD-OUT MODEL PREDICTIONS
# ============================================================
# Purpose:
#   Generate predictions for independent held-out BRFSS
#   respondents using the three final classification models.
#
# Models:
#   Logistic Regression
#   LASSO Logistic Regression
#   Random Forest
#
# Dataset:
#   BRFSS 2014–2024
#
# Method:
#   Load previously trained models and the held-out split.
#   Identify eligible held-out respondents for each year.
#   Apply computational subsampling where necessary.
#   Generate FMD probabilities using all three models.
#   Save a separate prediction checkpoint for each year.
#
# Sampling:
#   All eligible AI/AN-alone held-out respondents are retained.
#
#   Comparison-group observations are limited to 60,000
#   per year for computational feasibility.
#
# Weighting:
#   Evaluation weights adjust for held-out selection
#   probabilities and computational subsampling.
#
# Reproducibility:
#   Existing yearly prediction checkpoints are reused.
#   All three models evaluate the same respondents.
#
# Important:
#   These weights support descriptive model evaluation.
#   They do not independently establish nationally valid
#   complex-survey confidence intervals.
# ============================================================


# ============================================================
# 01 — LOAD PROJECT UTILITIES
# ============================================================

# Import shared configuration, data preparation functions,
# model prediction methods, and checkpoint utilities.

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

# Verify that the required modeling packages are installed.
check_packages()

# Load data.table without displaying startup messages.
suppressPackageStartupMessages(
    library(data.table)
)


# ============================================================
# 02 — LOAD FINAL CLASSIFICATION MODELS
# ============================================================

# Load the final models fitted using the selected
# cross-validation configurations.
#
# Each model checkpoint contains:
#   Model family identifier.
#   Selected configuration.
#   Fitted model object.
#
# All three models are loaded before scoring begins.

models <- lapply(
    c(
        "logistic",
        "lasso",
        "forest"
    ),
    function(family) {
        
        readRDS(
            file.path(
                MODEL_DIR,
                paste0(
                    family,
                    ".rds"
                )
            )
        )
    }
)

# Assign readable names to the model list.
names(models) <- c(
    "logistic",
    "lasso",
    "forest"
)


# ============================================================
# 03 — LOAD HELD-OUT SPLIT INFORMATION
# ============================================================

# Load the respondent indices selected during the
# original stratified training and testing split.
#
# The checkpoint contains:
#   indices:
#     Original pooled dataset row indices.
#
#   probabilities:
#     Probability of selection into the held-out sample.
#
#   seed:
#     Random seed used during sampling.
#
#   fraction:
#     Requested held-out sample proportion.

split <- readRDS(
    file.path(
        CHECKPOINT_DIR,
        "heldout_split.rds"
    )
)


# ============================================================
# 04 — IDENTIFY YEARS REQUIRING PREDICTIONS
# ============================================================

# Check whether a prediction checkpoint already exists
# for each study year.
#
# Expected filenames:
#   heldout_predictions_2014.rds
#   ...
#   heldout_predictions_2024.rds
#
# Only years without saved prediction files are processed.

missing_years <- STUDY_YEARS[
    !file.exists(
        file.path(
            CHECKPOINT_DIR,
            sprintf(
                "heldout_predictions_%d.rds",
                STUDY_YEARS
            )
        )
    )
]


# ============================================================
# 05 — CHECK FOR COMPLETED PREDICTIONS
# ============================================================

# Skip scoring when every study year already has
# a saved held-out prediction checkpoint.
#
# Otherwise, load the pooled dataset and score
# only the years that are missing.

if (!length(missing_years)) {
    
    message(
        "[SKIP] Every year has saved held-out predictions."
    )
    
} else {
    
    
    # ========================================================
    # 06 — LOAD POOLED BRFSS DATASET
    # ========================================================
    
    # Load the existing 2014–2024 analytical dataset.
    #
    # The pooled dataset is used to retrieve the original
    # respondents identified by the held-out split.
    #
    # No new training/testing split is generated.
    
    d <- load_data()
    
    
    # ========================================================
    # 07 — PROCESS EACH MISSING SURVEY YEAR
    # ========================================================
    
    # Score held-out respondents separately by survey year.
    #
    # Each year receives its own reproducible
    # prediction checkpoint.
    
    for (year in missing_years) {
        
        
        # ----------------------------------------------------
        # 07A — IDENTIFY HELD-OUT RESPONDENTS
        # ----------------------------------------------------
        
        # Identify held-out observations belonging
        # to the current survey year.
        ix_year <- d$YEAR[
            split$indices
        ] == year
        
        # Retrieve their original pooled dataset indices.
        ids <- split$indices[
            ix_year
        ]
        
        # Retrieve their held-out selection probabilities.
        split_pi <- split$probabilities[
            ix_year
        ]
        
        
        # ====================================================
        # 08 — IDENTIFY STUDY POPULATIONS
        # ====================================================
        
        # Separate held-out respondents into the
        # two primary population groups.
        #
        # AI/AN alone:
        #   AIAN_ALONE = TRUE
        #
        # Other identified single-race:
        #   AIAN_ALONE = FALSE
        #
        # These are positions within the current year's
        # held-out respondent index vector.
        
        aian <- which(
            d$AIAN_ALONE[ids] %in% TRUE
        )
        
        other <- which(
            d$AIAN_ALONE[ids] %in% FALSE
        )
        
        
        # ----------------------------------------------------
        # 08A — VALIDATE POPULATION AVAILABILITY
        # ----------------------------------------------------
        
        # Confirm that both study populations are
        # represented in the current year.
        #
        # Both groups are needed for the planned
        # subgroup performance comparisons.
        
        if (
            !length(aian) ||
            !length(other)
        ) {
            
            stop(
                "No members of one comparison group in ",
                year
            )
        }
        
        
        # ====================================================
        # 09 — SUBSAMPLE COMPARISON RESPONDENTS
        # ====================================================
        
        # Retain every eligible AI/AN-alone held-out respondent.
        #
        # Cap the larger comparison group at the configured
        # maximum number of observations per survey year.
        #
        # Default:
        #   OTHER_TEST_CAP_PER_YEAR = 60,000
        
        
        # ----------------------------------------------------
        # 09A — DETERMINE COMPARISON SAMPLE SIZE
        # ----------------------------------------------------
        
        n_other <- min(
            length(other),
            OTHER_TEST_CAP_PER_YEAR
        )
        
        
        # ----------------------------------------------------
        # 09B — SET REPRODUCIBLE RANDOM SEED
        # ----------------------------------------------------
        
        # Combine the project seed with the survey year.
        #
        # This produces a reproducible sample for each year
        # while allowing different selections across years.
        
        set.seed(
            SEED + year
        )
        
        
        # ----------------------------------------------------
        # 09C — SELECT COMPARISON RESPONDENTS
        # ----------------------------------------------------
        
        # Randomly sample comparison respondents only
        # when the eligible number exceeds the cap.
        #
        # Otherwise, retain the entire comparison group.
        
        selected_other <- if (n_other < length(other)) {
            
            sample(
                other,
                n_other
            )
            
        } else {
            
            other
        }
        
        
        # ----------------------------------------------------
        # 09D — COMBINE STUDY POPULATIONS
        # ----------------------------------------------------
        
        # Combine all AI/AN-alone respondents with
        # the selected comparison-group respondents.
        #
        # Sort their positions to preserve the order
        # of the original held-out observations.
        
        local_ids <- sort(
            c(
                aian,
                selected_other
            )
        )
        
        # Retrieve the original pooled dataset indices.
        take <- ids[
            local_ids
        ]
        
        # Retrieve the corresponding split probabilities.
        test_pi <- split_pi[
            local_ids
        ]
        
        
        # ====================================================
        # 10 — CALCULATE EVALUATION INCLUSION PROBABILITIES
        # ====================================================
        
        # Account for computational selection into
        # the yearly evaluation sample.
        #
        # AI/AN alone:
        #   All held-out observations are retained.
        #   Computational selection probability = 1.
        #
        # Other identified single-race:
        #   Selection probability equals the proportion
        #   retained after applying the yearly cap.
        
        evaluation_pi <- ifelse(
            d$AIAN_ALONE[take],
            1,
            n_other / length(other)
        )
        
        
        # ====================================================
        # 11 — PREPARE HELD-OUT EVALUATION DATA
        # ====================================================
        
        # Identify variables required for prediction
        # and subgroup model evaluation.
        #
        # Includes:
        #   Survey year.
        #   AI/AN-alone classification.
        #   Observed FMD outcome.
        #   Pooled survey weight.
        #   All model predictor variables.
        
        columns <- unique(
            c(
                "YEAR",
                "AIAN_ALONE",
                "FMD",
                "POOL_WEIGHT",
                FEATURES
            )
        )
        
        
        # ----------------------------------------------------
        # 11A — EXTRACT HELD-OUT RESPONDENTS
        # ----------------------------------------------------
        
        # Create a data frame containing the selected
        # respondents and their model predictors.
        
        test <- as.data.frame(
            d[
                take,
                ..columns
            ]
        )
        
        
        # ----------------------------------------------------
        # 11B — NORMALIZE PREDICTOR VARIABLES
        # ----------------------------------------------------
        
        # Apply the same factor levels and numeric types
        # used during model training.
        #
        # This ensures consistent predictor encoding.
        
        test <- normalize_features(
            test
        )
        
        
        # ====================================================
        # 12 — CALCULATE EVALUATION WEIGHTS
        # ====================================================
        
        # Adjust each respondent's pooled weight for:
        #   Original held-out sample selection.
        #   Additional computational subsampling.
        #
        # Formula:
        #   ML_WEIGHT = POOL_WEIGHT /
        #               (test_pi * evaluation_pi)
        #
        # This corrects the relative contributions of
        # sampled observations for the specified selection
        # procedures.
        #
        # It does not independently validate the original
        # multi-year survey weighting methodology.
        
        test$ML_WEIGHT <- test$POOL_WEIGHT / (
            test_pi * evaluation_pi
        )
        
        
        # ----------------------------------------------------
        # 12A — VALIDATE EVALUATION WEIGHTS
        # ----------------------------------------------------
        
        # Stop if any corrected evaluation weight
        # is missing, non-finite, zero, or negative.
        
        if (any(!valid_weight(test$ML_WEIGHT))) {
            
            stop(
                "Test sampling-adjusted weights were invalid."
            )
        }
        
        
        # ====================================================
        # 13 — GENERATE PREDICTIONS FROM ALL THREE MODELS
        # ====================================================
        
        # Generate predicted probabilities of FMD using:
        #   Logistic regression.
        #   LASSO logistic regression.
        #   Random forest.
        #
        # Each model scores the exact same test respondents.
        #
        # predict_model() applies the appropriate
        # model-specific prediction procedure.
        #
        # For LASSO, the selected lambda value is used.
        
        for (family in names(models)) {
            
            # Retrieve the saved model and configuration.
            fit <- models[[family]]
            
            # Create a column containing the predicted
            # probability of frequent mental distress.
            #
            # Expected columns:
            #   pred_logistic
            #   pred_lasso
            #   pred_forest
            
            test[[
                paste0(
                    "pred_",
                    family
                )
            ]] <- predict_model(
                fit$model,
                family,
                test,
                fit$config
            )
        }
        
        
        # ====================================================
        # 14 — CONSTRUCT HELD-OUT PREDICTION OUTPUT
        # ====================================================
        
        # Retain only the variables required for
        # subsequent model performance evaluation.
        #
        # YEAR:
        #   Survey year.
        #
        # FMD:
        #   Observed frequent mental distress outcome.
        #
        # AIAN_ALONE:
        #   Population classification.
        #
        # ML_WEIGHT:
        #   Sampling-adjusted evaluation weight.
        #
        # pred_*:
        #   Predicted FMD probabilities from each model.
        
        output <- test[
            ,
            c(
                "YEAR",
                "FMD",
                "AIAN_ALONE",
                "ML_WEIGHT",
                paste0(
                    "pred_",
                    names(models)
                )
            )
        ]
        
        
        # ====================================================
        # 15 — SAVE YEARLY PREDICTION CHECKPOINT
        # ====================================================
        
        # Save the completed evaluation data for
        # the current survey year.
        #
        # Example:
        #   heldout_predictions_2024.rds
        #
        # Saved predictions can be reused without
        # repeating model inference.
        
        save_checkpoint(
            output,
            file.path(
                CHECKPOINT_DIR,
                sprintf(
                    "heldout_predictions_%d.rds",
                    year
                )
            )
        )
        
        
        # ====================================================
        # 16 — REPORT YEARLY SCORING RESULTS
        # ====================================================
        
        # Display the survey year, total number
        # of scored respondents, and AI/AN sample size.
        
        cat(
            "[TEST] ",
            year,
            ": ",
            nrow(output),
            " scored; AI/AN ",
            length(aian),
            "\n",
            sep = ""
        )
        
        
        # ====================================================
        # 17 — RELEASE YEARLY INTERMEDIATE OBJECTS
        # ====================================================
        
        # Remove temporary test and output datasets
        # after the current year's results are saved.
        
        rm(
            test,
            output
        )
        
        # Request garbage collection.
        gc(
            verbose = FALSE
        )
    }
    
    
    # ========================================================
    # 18 — RELEASE POOLED DATASET
    # ========================================================
    
    # Remove the large pooled dataset once all
    # missing survey years have been scored.
    
    rm(d)
    
    gc(
        verbose = FALSE
    )
}


# ============================================================
# 19 — REPORT COMPLETION
# ============================================================

# Confirm that all yearly held-out prediction
# checkpoints are now available.

message(
    "[COMPLETE] Yearly held-out predictions available."
)
