# ============================================================
# ASSESSING THE GAP — FINAL MODEL FITTING
# ============================================================
# Purpose:
#   Train the three final classification models using
#   configurations selected during cross-validation.
#
# Models:
#   Logistic Regression
#   LASSO Logistic Regression
#   Random Forest
#
# Method:
#   Load the selected configurations from common
#   three-fold cross-validation.
#
#   Fit each model using the same prepared training
#   dataset and computationally adjusted model weights.
#
# Reproducibility:
#   Each completed model is saved as an RDS checkpoint.
#   Existing model files are reused without refitting.
#
# Important:
#   Final model fitting uses only the training dataset.
#   Independent held-out observations are reserved for
#   the subsequent evaluation stages.
#
# This script produces: 
# 
# Final logistic regression model
#   outputs/models/logistic.rds	
#
# Final LASSO logistic regression model
#   outputs/models/lasso.rds
#
# Final random forest model
#   outputs/models/forest.rds	
# ============================================================


# ============================================================
# 01 — LOAD PROJECT UTILITIES
# ============================================================

# Import shared configuration, model-fitting functions,
# validation utilities, and checkpoint management.

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
# 02 — LOAD SELECTED MODEL CONFIGURATIONS
# ============================================================

# Load the configurations selected during the common
# three-fold cross-validation procedure.
#
# Selection is based primarily on mean weighted AUROC.
#
# The selected configurations are stored separately
# for each model family.

selected <- readRDS(
    file.path(
        CHECKPOINT_DIR,
        "selected_cv_configurations.rds"
    )
)


# ============================================================
# 03 — LOAD MODEL TRAINING DATA
# ============================================================

# Load the prepared training dataset.
#
# This dataset contains:
#   Eligible training respondents.
#   Normalized predictor variables.
#   Binary FMD outcome.
#   AI/AN-alone classification.
#   Computationally adjusted training weights.
#
# The held-out evaluation dataset is not loaded.

training <- readRDS(
    file.path(
        CHECKPOINT_DIR,
        "training_sample.rds"
    )
)


# ============================================================
# 04 — INITIALIZE FINAL MODEL FITTING
# ============================================================

# Iterate through the three classification model families.
#
# Each model is fitted to the same training sample
# using its selected cross-validation configuration.
#
# Model families:
#   logistic = Logistic regression
#   lasso    = LASSO logistic regression
#   forest   = Random forest

for (family in c(
    "logistic",
    "lasso",
    "forest"
)) {
    
    
    # ========================================================
    # 05 — DEFINE MODEL CHECKPOINT
    # ========================================================
    
    # Construct the output path for the current model.
    #
    # Expected filenames:
    #   logistic.rds
    #   lasso.rds
    #   forest.rds
    
    destination <- file.path(
        MODEL_DIR,
        paste0(
            family,
            ".rds"
        )
    )
    
    
    # ========================================================
    # 06 — CHECK FOR EXISTING MODEL
    # ========================================================
    
    # Skip fitting if the final model checkpoint
    # already exists.
    #
    # This avoids unnecessary retraining and preserves
    # previously completed model results.
    #
    # Existing checkpoints are assumed to correspond
    # to the current training data and model settings.
    
    if (file.exists(destination)) {
        
        message(
            "[SKIP] Existing final model: ",
            family
        )
        
        next
    }
    
    
    # ========================================================
    # 07 — RETRIEVE SELECTED CONFIGURATION
    # ========================================================
    
    # Find the cross-validated configuration associated
    # with the current model family.
    #
    # Logistic regression:
    #   Default configuration.
    #
    # LASSO regression:
    #   Selected lambda value.
    #
    # Random forest:
    #   Selected mtry value.
    
    config <- selected$config[
        selected$family == family
    ]
    
    
    # ========================================================
    # 08 — VALIDATE CONFIGURATION
    # ========================================================
    
    # Confirm that exactly one configuration
    # was selected for the current model family.
    #
    # Stop execution if the selection is missing
    # or multiple configurations were returned.
    
    if (length(config) != 1L) {
        
        stop(
            "Missing unique cross-validated selection for ",
            family
        )
    }
    
    
    # ========================================================
    # 09 — REPORT MODEL FITTING
    # ========================================================
    
    # Display the model family and training sample size
    # before beginning final model fitting.
    
    cat(
        "[FIT] ",
        family,
        " | n=",
        nrow(training),
        "\n",
        sep = ""
    )
    
    
    # ========================================================
    # 10 — FIT FINAL CLASSIFICATION MODEL
    # ========================================================
    
    # Train the current model using:
    #   The full prepared training sample.
    #   The selected cross-validation configuration.
    #   The weighting and modeling settings specified
    #   in the shared project utilities.
    #
    # fit_model() handles the algorithm-specific
    # training procedure.
    
    model <- fit_model(
        family,
        training,
        config
    )
    
    
    # ========================================================
    # 11 — SAVE FINAL MODEL CHECKPOINT
    # ========================================================
    
    # Save the fitted model together with its family
    # identifier and selected configuration.
    #
    # This allows subsequent evaluation scripts
    # to identify the appropriate prediction method.
    
    save_checkpoint(
        list(
            family = family,
            config = config,
            model = model
        ),
        destination
    )
    
    
    # ========================================================
    # 12 — RELEASE MODEL MEMORY
    # ========================================================
    
    # Remove the fitted model object after saving.
    #
    # The next model is fitted independently using
    # the same training dataset.
    
    rm(model)
    
    # Request garbage collection.
    gc(
        verbose = FALSE
    )
}


# ============================================================
# 13 — REPORT COMPLETION
# ============================================================

# This message appears when all three model families
# have either been fitted or their existing
# checkpoints have been reused.

message(
    "[COMPLETE] All three final models saved."
)
