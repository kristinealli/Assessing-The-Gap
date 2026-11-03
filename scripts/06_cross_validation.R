# ============================================================
# ASSESSING THE GAP — COMMON CROSS-VALIDATION
# ============================================================
# Purpose:
#   Compare and select machine-learning model configurations
#   using a common three-fold cross-validation procedure.
#
# Models:
#   Logistic Regression
#   LASSO Logistic Regression
#   Random Forest
#
# Method:
#   Train and evaluate each model using identical
#   cross-validation fold assignments.
#
# Model Selection:
#   Select configurations using mean weighted AUROC
#   across the cross-validation folds.
#
#   Higher AUROC indicates better discrimination.
#   Brier score is used to break AUROC ties.
#
# Important:
#   Cross-validation uses only the training dataset.
#   Independent held-out test data are not used
#   during model configuration selection.
#
#   Completed fold checkpoints are reused to avoid
#   repeating computationally expensive model fitting.
# 
# This script produces: 
#
# Logistic regression CV results
#   outputs/checkpoints/cv_logistic_fold_1.rds
#   outputs/checkpoints/cv_logistic_fold_2.rds
#   outputs/checkpoints/cv_logistic_fold_3.rds
#
# LASSO performance across lambda values
#   outputs/checkpoints/cv_lasso_fold_1.rds
#   outputs/checkpoints/cv_lasso_fold_2.rds
#   outputs/checkpoints/cv_lasso_fold_3.rds
#
# Random forest performance across mtry values
#   outputs/checkpoints/cv_forest_fold_1.rds
#   outputs/checkpoints/cv_forest_fold_2.rds
#   outputs/checkpoints/cv_forest_fold_3.rds
#
# Winning configuration for each model family
#   outputs/selected_cv_configurations.rds
#
# Mean CV AUROC and Brier score for all configurations
#   outputs/06_cv_model_comparison.csv
#
# Selected configuration and performance for each model
#   outputs/06_selected_configurations.csv
# ============================================================


# ============================================================
# 01 — LOAD PROJECT UTILITIES
# ============================================================

# Import shared configuration, model-fitting functions,
# evaluation metrics, and checkpoint management.

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

# Verify that all required modeling packages are installed.
check_packages()


# ============================================================
# 02 — LOAD TRAINING DATA AND CV FOLDS
# ============================================================

# Load the previously prepared model training dataset.
training <- readRDS(
    file.path(
        CHECKPOINT_DIR,
        "training_sample.rds"
    )
)

# Load the common cross-validation fold assignments.
folds <- readRDS(
    file.path(
        CHECKPOINT_DIR,
        "common_cv_folds.rds"
    )
)


# ============================================================
# 03 — VALIDATE CROSS-VALIDATION ASSIGNMENTS
# ============================================================

# Confirm that every training respondent has a
# corresponding cross-validation fold assignment.
#
# The fold vector must contain exactly one entry
# for each observation in the training dataset.

if (length(folds) != nrow(training)) {
    
    stop(
        "Fold vector length mismatch."
    )
}


# ============================================================
# 04 — INITIALIZE MODEL CROSS-VALIDATION
# ============================================================

# Evaluate three classification model families:
#
#   logistic:
#     Weighted logistic regression.
#
#   lasso:
#     L1-penalized logistic regression.
#
#   forest:
#     Weighted random forest classifier.
#
# Every model uses the same CV fold assignments
# to support comparable performance evaluation.

for (family in c(
    "logistic",
    "lasso",
    "forest"
)) {
    
    
    # ========================================================
    # 05 — ITERATE THROUGH CROSS-VALIDATION FOLDS
    # ========================================================
    
    # Each fold is used once for assessment while
    # the remaining folds are used for model training.
    
    for (k in seq_len(CV_FOLDS)) {
        
        
        # ----------------------------------------------------
        # 05A — DEFINE FOLD CHECKPOINT
        # ----------------------------------------------------
        
        # Construct a unique checkpoint filename
        # for the current model family and fold.
        #
        # Example:
        #   cv_logistic_fold_1.rds
        
        checkpoint <- file.path(
            CHECKPOINT_DIR,
            sprintf(
                "cv_%s_fold_%d.rds",
                family,
                k
            )
        )
        
        
        # ----------------------------------------------------
        # 05B — REUSE COMPLETED CHECKPOINTS
        # ----------------------------------------------------
        
        # Skip model fitting if the current fold
        # has already been completed and saved.
        #
        # Existing checkpoints must correspond to
        # the current model and sampling settings.
        
        if (file.exists(checkpoint)) {
            
            message(
                "[SKIP] ",
                basename(checkpoint)
            )
            
            next
        }
        
        
        # ----------------------------------------------------
        # 05C — PREPARE TRAINING AND ASSESSMENT DATA
        # ----------------------------------------------------
        
        # Use all folds except k for model fitting.
        learn <- training[
            folds != k,
            ,
            drop = FALSE
        ]
        
        # Use fold k for independent CV assessment.
        assess <- training[
            folds == k,
            ,
            drop = FALSE
        ]
        
        # Report the current model, fold, and sample sizes.
        message(
            sprintf(
                paste(
                    "[START] CV %s / fold %d:",
                    "training %d, assessing %d"
                ),
                family,
                k,
                nrow(learn),
                nrow(assess)
            )
        )
        
        
        # ----------------------------------------------------
        # 05D — INITIALIZE PERFORMANCE RESULTS
        # ----------------------------------------------------
        
        # Store performance metrics for each model
        # configuration evaluated within this fold.
        scores <- list()
        
        
        # ====================================================
        # 06 — LOGISTIC REGRESSION CROSS-VALIDATION
        # ====================================================
        
        if (family == "logistic") {
            
            # Fit weighted logistic regression.
            #
            # This model uses its default configuration
            # and does not require hyperparameter tuning.
            fit <- fit_model(
                "logistic",
                learn
            )
            
            # Generate predicted FMD probabilities
            # for the current assessment fold.
            p <- predict_model(
                fit,
                "logistic",
                assess
            )
            
            # Calculate weighted classification metrics.
            #
            # Store the configuration label as "default".
            scores[[1]] <- transform(
                score_metrics(
                    assess$FMD,
                    p,
                    assess$ML_WEIGHT
                ),
                config = "default"
            )
            
            
            # ====================================================
            # 07 — LASSO REGRESSION CROSS-VALIDATION
            # ====================================================
            
        } else if (family == "lasso") {
            
            # Fit LASSO logistic regression.
            #
            # LASSO applies L1 regularization to shrink
            # coefficients and perform feature selection.
            #
            # The model is fitted using the predefined
            # lambda grid from the shared configuration.
            fit <- fit_model(
                "lasso",
                learn
            )
            
            
            # ------------------------------------------------
            # 07A — EVALUATE LAMBDA CONFIGURATIONS
            # ------------------------------------------------
            
            # Generate predictions for each lambda value.
            #
            # Smaller lambda:
            #   Less coefficient shrinkage.
            #
            # Larger lambda:
            #   More coefficient shrinkage.
            
            for (lambda in LAMBDA_GRID) {
                
                # Predict FMD probabilities using
                # the current regularization strength.
                p <- predict_model(
                    fit,
                    "lasso",
                    assess,
                    lambda
                )
                
                # Calculate weighted evaluation metrics.
                #
                # Save the lambda value as a character
                # label with ten decimal places.
                scores[[
                    length(scores) + 1L
                ]] <- transform(
                    score_metrics(
                        assess$FMD,
                        p,
                        assess$ML_WEIGHT
                    ),
                    config = sprintf(
                        "%.10f",
                        lambda
                    )
                )
            }
            
            
            # ====================================================
            # 08 — RANDOM FOREST CROSS-VALIDATION
            # ====================================================
            
        } else {
            
            # Evaluate the candidate mtry values.
            #
            # mtry controls the number of predictors
            # considered at each decision-tree split.
            #
            # Candidate values are defined in RF_MTRY_GRID.
            
            
            # ------------------------------------------------
            # 08A — EVALUATE RANDOM FOREST CONFIGURATIONS
            # ------------------------------------------------
            
            for (mtry in RF_MTRY_GRID) {
                
                # Train a random forest with the
                # current mtry configuration.
                fit <- fit_model(
                    "forest",
                    learn,
                    mtry
                )
                
                # Predict FMD probabilities for
                # the assessment observations.
                p <- predict_model(
                    fit,
                    "forest",
                    assess
                )
                
                # Calculate weighted evaluation metrics.
                #
                # Record the current mtry value.
                scores[[
                    length(scores) + 1L
                ]] <- transform(
                    score_metrics(
                        assess$FMD,
                        p,
                        assess$ML_WEIGHT
                    ),
                    config = as.character(
                        mtry
                    )
                )
            }
        }
        
        
        # ====================================================
        # 09 — COMPILE FOLD PERFORMANCE RESULTS
        # ====================================================
        
        # Combine the configuration-specific results
        # into a single data frame.
        results <- do.call(
            rbind,
            scores
        )
        
        # Add the model family identifier.
        results$family <- family
        
        # Add the cross-validation fold number.
        results$fold <- k
        
        
        # ====================================================
        # 10 — SAVE FOLD CHECKPOINT
        # ====================================================
        
        # Save the completed fold performance metrics.
        #
        # This allows the current fold to be skipped
        # during future executions.
        
        save_checkpoint(
            results,
            checkpoint
        )
        
        
        # ====================================================
        # 11 — DISPLAY FOLD PERFORMANCE
        # ====================================================
        
        # Display the most relevant configuration metrics:
        #
        #   family = Model algorithm
        #   fold   = Cross-validation fold
        #   config = Hyperparameter configuration
        #   auroc  = Weighted discrimination
        #   brier  = Weighted probability prediction error
        
        print(
            results[
                ,
                c(
                    "family",
                    "fold",
                    "config",
                    "auroc",
                    "brier"
                )
            ]
        )
        
        
        # ====================================================
        # 12 — RELEASE FOLD MEMORY
        # ====================================================
        
        # Remove temporary training, assessment,
        # model, and prediction objects.
        rm(
            learn,
            assess,
            fit,
            p
        )
        
        # Request garbage collection.
        gc(
            verbose = FALSE
        )
    }
}


# ============================================================
# 13 — IDENTIFY SAVED CROSS-VALIDATION CHECKPOINTS
# ============================================================

# Construct the expected checkpoint filenames
# for every model family and CV fold.
#
# These files contain the evaluation results
# generated during the preceding loops.

files <- unlist(
    lapply(
        c(
            "logistic",
            "lasso",
            "forest"
        ),
        function(family) {
            
            sprintf(
                "cv_%s_fold_%d.rds",
                family,
                seq_len(CV_FOLDS)
            )
        }
    )
)


# ============================================================
# 14 — COMBINE CROSS-VALIDATION RESULTS
# ============================================================

# Read all saved fold checkpoints.
#
# Combine the results into one data frame containing
# every model family, configuration, and fold.

all_results <- do.call(
    rbind,
    lapply(
        file.path(
            CHECKPOINT_DIR,
            files
        ),
        readRDS
    )
)


# ============================================================
# 15 — SUMMARIZE MODEL PERFORMANCE
# ============================================================

# Calculate the mean AUROC and Brier score across
# the cross-validation folds for each configuration.
#
# AUROC:
#   Higher values indicate better discrimination.
#
# Brier Score:
#   Lower values indicate better probability predictions.
#
# Each fold contributes equally to the mean of
# its already-weighted performance estimates.

comparison <- aggregate(
    cbind(
        auroc,
        brier
    ) ~ family + config,
    data = all_results,
    FUN = mean
)


# ------------------------------------------------------------
# 15A — RENAME PERFORMANCE COLUMNS
# ------------------------------------------------------------

# Rename the aggregated metrics to distinguish them
# from individual fold performance estimates.

names(comparison)[
    names(comparison) == "auroc"
] <- "cv_mean_auroc"

names(comparison)[
    names(comparison) == "brier"
] <- "cv_mean_brier"


# ------------------------------------------------------------
# 15B — RANK MODEL CONFIGURATIONS
# ------------------------------------------------------------

# Sort configurations by:
#
#   1. Model family.
#   2. Highest mean weighted AUROC.
#   3. Lowest mean weighted Brier score.
#
# This ordering determines the preferred configuration
# within each classification model family.

comparison <- comparison[
    order(
        comparison$family,
        -comparison$cv_mean_auroc,
        comparison$cv_mean_brier
    ),
]


# ============================================================
# 16 — SELECT THE BEST CONFIGURATION PER MODEL
# ============================================================

# Divide the comparison table by model family.
#
# Select the first row from each group, which contains
# the configuration with the highest mean AUROC.
#
# If AUROC values are tied, the configuration with
# the lower mean Brier score is preferred.

selected <- do.call(
    rbind,
    lapply(
        split(
            comparison,
            comparison$family
        ),
        function(z) {
            
            z[
                1,
                ,
                drop = FALSE
            ]
        }
    )
)

# Remove inherited row names.
rownames(selected) <- NULL


# ============================================================
# 17 — SAVE SELECTED MODEL CONFIGURATIONS
# ============================================================

# Save the winning configurations for use
# when fitting final classification models.

save_checkpoint(
    selected,
    file.path(
        CHECKPOINT_DIR,
        "selected_cv_configurations.rds"
    )
)


# ============================================================
# 18 — EXPORT CROSS-VALIDATION TABLES
# ============================================================

# Export the complete model comparison.
save_csv(
    comparison,
    "06_cv_model_comparison.csv"
)

# Export the selected configuration for each model family.
save_csv(
    selected,
    "06_selected_configurations.csv"
)


# ============================================================
# 19 — REPORT CROSS-VALIDATION RESULTS
# ============================================================

# Display the selected model configurations.
cat(
    "[CV WINNERS] Weighted AUROC selected ",
    "these configurations:\n",
    sep = ""
)

print(
    selected,
    row.names = FALSE
)


# ============================================================
# 20 — DOCUMENT INDEPENDENT HELD-OUT EVALUATION
# ============================================================

# Clarify the distinction between:
#
#   Cross-validation:
#     Selects model configurations using training data.
#
#   Held-out evaluation:
#     Measures final performance on observations
#     not used during model fitting or tuning.
#
# Final reported model performance must come from
# the independent held-out evaluation dataset.

cat(
    "[READ] Cross-validation selects models only; ",
    "the independent\n",
    "held-out data determine final reported ",
    "model performance.\n",
    sep = ""
)
