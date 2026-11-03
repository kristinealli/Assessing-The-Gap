# ============================================================
# ASSESSING THE GAP — RUN ANALYSIS PIPELINE
# ============================================================
# Purpose:
#   Execute the required research analyses in sequence,
#   beginning with the existing pooled BRFSS dataset.
#
# Dataset:
#   analysis_pooled_2014_2024.rds
#
# Usage:
#   1. Open the Assessing_the_Gap.Rproj file in RStudio.
#   2. Open run_all.R.
#   3. Select Source to execute the full analysis pipeline.
#
# Workflow:
#   Verify dataset integrity.
#   Generate descriptive statistics.
#   Prepare training and testing samples.
#   Perform principal component analysis (PCA).
#   Estimate predictor associations.
#   Cross-validate classification models.
#   Fit final machine-learning models.
#   Evaluate held-out predictions.
#   Generate tables, figures, and the research report.
#
#   Completed model and prediction checkpoints are reused
#   by the individual analysis scripts where supported.
#
#   Model fitting and cross-validation may require
#   substantial computational resources.
# ============================================================


# ============================================================
# 01 — RUN PRELIMINARY TESTS
# ============================================================

# Execute lightweight smoke tests before beginning the
# computationally intensive analysis pipeline.
#
# Tests verify:
#   Weighted AUROC calculations.
#   Classification metric calculations.
#   Predictor normalization.
#   Machine-learning design matrix construction.
#
# Execution stops if a test fails.

source(
    "tests/smoke_test.R"
)


# ============================================================
# 02 — DEFINE ANALYSIS PIPELINE
# ============================================================

# List the required analysis scripts in execution order.
#
# Each script depends on the configuration, data products,
# or model outputs created by earlier stages.
#
# The pipeline begins with the existing pooled dataset
# and concludes with the final report and session details.

steps <- c(
    
    # --------------------------------------------------------
    # DATA VERIFICATION AND EXPLORATION
    # --------------------------------------------------------
    
    # Verify the pooled dataset and internal coding rules.
    "scripts/01_audit_pooled_data.R",
    
    # Examine population characteristics, FMD prevalence,
    # temporal patterns, and weighting sensitivity.
    "scripts/02_descriptive_analysis.R",
    
    
    # --------------------------------------------------------
    # MODEL DATA PREPARATION
    # --------------------------------------------------------
    
    # Prepare eligible respondents, training samples,
    # held-out test samples, and model weights.
    "scripts/03_prepare_model_samples.R",
    
    
    # --------------------------------------------------------
    # EXPLORATORY AND ASSOCIATION ANALYSES
    # --------------------------------------------------------
    
    # Perform principal component analysis (PCA).
    "scripts/04_pca.R",
    
    # Estimate associations between study predictors
    # and frequent mental distress.
    "scripts/05_associations.R",
    
    
    # --------------------------------------------------------
    # MACHINE-LEARNING MODEL DEVELOPMENT
    # --------------------------------------------------------
    
    # Compare model configurations using cross-validation.
    "scripts/06_cross_validation.R",
    
    # Fit the final logistic regression, LASSO,
    # and random forest models.
    "scripts/07_fit_final_models.R",
    
    
    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------
    
    # Generate predictions for held-out survey years.
    "scripts/08_score_heldout_years.R",
    
    # Calculate subgroup performance metrics and
    # generate evaluation tables and figures.
    "scripts/09_evaluate_and_visualize.R",
    
    
    # --------------------------------------------------------
    # REPORTING AND REPRODUCIBILITY
    # --------------------------------------------------------
    
    # Render the final research report.
    "scripts/11_render_report.R",
    
    # Save R session information for reproducibility.
    "scripts/12_session_information.R"
)


# ============================================================
# 03 — EXECUTE ANALYSIS SCRIPTS
# ============================================================

# Execute each script sequentially.
#
# source() runs the specified R script.
#
# A separate environment is created for each script,
# with the global environment as its parent.
#
# This helps prevent variables created by one script
# from unintentionally overwriting those in another.
#
# If an unhandled error occurs, execution stops.
# The remaining scripts will not run.

for (step in steps) {
    
    # Display a separator before each analysis stage.
    cat(
        "\n====================================================\n"
    )
    
    # Display the name of the script being executed.
    cat(
        "[RUN] ",
        step,
        "\n",
        sep = ""
    )
    
    cat(
        "====================================================\n"
    )
    
    # Execute the current script in a separate environment.
    source(
        step,
        local = new.env(
            parent = globalenv()
        )
    )
}


# ============================================================
# 04 — REPORT PIPELINE COMPLETION
# ============================================================

# This message appears only if all required scripts
# complete without an unhandled error.

cat(
    "\n[COMPLETE] Required analyses and report generated.\n"
)


# ============================================================
# 05 — OPTIONAL ANALYSIS
# ============================================================

# The conditional bootstrap is not part of the required
# execution pipeline.
#
# It can be run separately to estimate variability in
# model performance conditional on the sampled data.
#
# Bootstrap intervals should not be interpreted as
# full BRFSS complex-survey design-based intervals.

cat(
    "[OPTIONAL] scripts/10_conditional_bootstrap.R\n"
)
