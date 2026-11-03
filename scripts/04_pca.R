# ============================================================
# ASSESSING THE GAP — PRINCIPAL COMPONENT ANALYSIS
# ============================================================
# Purpose:
#   Explore underlying patterns and relationships among
#   demographic, socioeconomic, and health predictors
#   using unsupervised Principal Component Analysis (PCA).
#
# Research Objective:
#   Examine whether combinations of predictor variables
#   reveal patterns across AI/AN-alone and other identified
#   single-race respondents.
#
# Method:
#   Convert categorical predictors into numeric indicators.
#   Standardize predictor variables.
#   Apply PCA to the prepared training sample.
#   Examine explained variance and component loadings.
#   Visualize respondents using the first two components.
#
# Important:
#   PCA is unsupervised. The FMD outcome is excluded
#   from component estimation.
#
#   The PCA sample is selected for computational feasibility
#   and is not nationally weighted or representative.
#
#   PCA identifies variation in predictors, not causal
#   relationships or differences in health outcomes.
#
# This script produces: 
# Saved PCA model and explained variance: 
#   outputs/models/pca.rds
# Predictor contributions to PC1 and PC2:
#   outputs/tables/04_pca_loadings.csv
# Percentage of variance explained by each component:
#   outputs/tables/04_pca_explained_variance.csv
# Scatterplot comparing the two study populations:
#  outputs/figures/04_pca_scores.png
# ============================================================


# ============================================================
# 01 — LOAD PROJECT UTILITIES
# ============================================================

# Import shared configuration, data preparation functions,
# checkpoint management, and figure export utilities.

source(
    file.path(
        "R",
        "01_utilities.R"
    )
)

# Verify that ggplot2 is installed.
check_packages(
    c("ggplot2")
)


# ============================================================
# 02 — LOAD MODEL TRAINING DATA
# ============================================================

# Load the previously prepared training sample.
#
# This dataset contains:
#   Eligible model-training respondents.
#   Standardized predictor variables.
#   Population classification.
#   Model training weights.
#
# The PCA analysis uses predictors only.

training <- readRDS(
    file.path(
        CHECKPOINT_DIR,
        "training_sample.rds"
    )
)


# ============================================================
# 03 — SELECT PCA ANALYSIS SAMPLE
# ============================================================

# Randomly select respondents from the training dataset.
#
# The sample size is limited by PCA_MAX_ROWS, which
# defaults to 12,000 observations.
#
# Sampling is unweighted and intended to reduce
# computational requirements.
#
# The random seed ensures reproducible selection.


# ------------------------------------------------------------
# 03A — SET RANDOM SEED
# ------------------------------------------------------------

set.seed(
    SEED + 11L
)


# ------------------------------------------------------------
# 03B — SELECT RESPONDENTS
# ------------------------------------------------------------

# Randomly select row indices without replacement.
#
# If the training dataset contains fewer observations
# than PCA_MAX_ROWS, all observations are retained.

ids <- sample.int(
    nrow(training),
    min(
        nrow(training),
        PCA_MAX_ROWS
    )
)


# ------------------------------------------------------------
# 03C — EXTRACT THE PCA SAMPLE
# ------------------------------------------------------------

# Retain the selected respondents and all columns.
sampled <- training[
    ids,
    ,
    drop = FALSE
]


# ============================================================
# 04 — CREATE NUMERIC PREDICTOR MATRIX
# ============================================================

# Convert the selected predictors into a numeric matrix.
#
# make_matrix() uses the shared DESIGN_FORMULA to:
#   Select the model predictors.
#   Convert categorical variables into indicator columns.
#   Remove the intercept column.
#   Standardize column names.
#
# The FMD outcome is not included in this matrix.

X <- make_matrix(
    sampled
)


# ============================================================
# 05 — REMOVE CONSTANT PREDICTORS
# ============================================================

# Identify predictor columns with nonzero variation.
#
# PCA requires variables with measurable variation
# because the predictors will be standardized.
#
# Constant columns have a standard deviation of zero
# and cannot be scaled meaningfully.


# ------------------------------------------------------------
# 05A — IDENTIFY VARYING COLUMNS
# ------------------------------------------------------------

varying <- apply(
    X,
    2,
    stats::sd
) > 0


# ------------------------------------------------------------
# 05B — RETAIN VARYING PREDICTORS
# ------------------------------------------------------------

# Remove predictor columns with zero standard deviation.
X <- X[
    ,
    varying,
    drop = FALSE
]


# ------------------------------------------------------------
# 05C — VALIDATE THE PCA MATRIX
# ------------------------------------------------------------

# Confirm that at least two varying predictor columns
# remain available for principal component analysis.

if (ncol(X) < 2L) {
    
    stop(
        "PCA requires at least two varying predictors."
    )
}


# ============================================================
# 06 — FIT PRINCIPAL COMPONENT ANALYSIS
# ============================================================

# Perform PCA on the prepared numeric predictor matrix.
#
# center = TRUE:
#   Subtract each variable's mean.
#
# scale. = TRUE:
#   Divide each variable by its standard deviation.
#
# Standardization prevents predictors with larger
# numerical scales from dominating the analysis.
#
# The resulting components are uncorrelated linear
# combinations of the standardized predictors.

fit <- stats::prcomp(
    X,
    center = TRUE,
    scale. = TRUE
)


# ============================================================
# 07 — CALCULATE EXPLAINED VARIANCE
# ============================================================

# Calculate the percentage of total predictor variation
# explained by each principal component.
#
# fit$sdev contains the standard deviation of each
# principal component.
#
# Squaring these values produces component variances.
#
# The variance proportions are multiplied by 100
# to express explained variance as percentages.

variance <- 100 * fit$sdev^2 / sum(
    fit$sdev^2
)


# ============================================================
# 08 — EXTRACT PRINCIPAL COMPONENT SCORES
# ============================================================

# Extract the first two principal component scores
# for each respondent in the PCA sample.
#
# PC1:
#   Direction explaining the greatest variation.
#
# PC2:
#   Direction explaining the second-greatest variation,
#   orthogonal to PC1.
#
# Population labels are added for visualization only.
# They are not used to estimate the components.

scores <- data.frame(
    PC1 = fit$x[
        ,
        1
    ],
    
    PC2 = fit$x[
        ,
        2
    ],
    
    cohort = two_group(
        sampled$AIAN_ALONE
    )
)


# ============================================================
# 09 — EXTRACT COMPONENT LOADINGS
# ============================================================

# Retrieve the variable loadings for the first two
# principal components.
#
# Loadings describe how strongly each standardized
# predictor contributes to a component.
#
# Positive and negative signs indicate the direction
# of the relationship within the component.
#
# The absolute magnitude helps identify which variables
# contribute most strongly to that component.
#
# Component signs are arbitrary and should not be
# interpreted as inherently favorable or unfavorable.

loadings <- data.frame(
    variable = rownames(
        fit$rotation
    ),
    
    PC1 = fit$rotation[
        ,
        1
    ],
    
    PC2 = fit$rotation[
        ,
        2
    ],
    
    row.names = NULL
)


# ============================================================
# 10 — SAVE PCA MODEL CHECKPOINT
# ============================================================

# Save the fitted PCA model and explained variance.
#
# The checkpoint allows component results to be
# retrieved without recomputing the analysis.

save_checkpoint(
    list(
        model = fit,
        explained_variance = variance
    ),
    file.path(
        MODEL_DIR,
        "pca.rds"
    )
)


# ============================================================
# 11 — SAVE PCA ANALYSIS TABLES
# ============================================================

# ------------------------------------------------------------
# 11A — SAVE COMPONENT LOADINGS
# ------------------------------------------------------------

# Export predictor loadings for PC1 and PC2.
#
# This table can be used to interpret which variables
# contribute to the largest patterns of variation.

save_csv(
    loadings,
    "04_pca_loadings.csv"
)


# ------------------------------------------------------------
# 11B — SAVE EXPLAINED VARIANCE
# ------------------------------------------------------------

# Create a table containing the percentage of
# total variance explained by every component.

explained_variance <- data.frame(
    component = paste0(
        "PC",
        seq_along(variance)
    ),
    
    explained_variance_pct = variance
)

# Export the explained variance table.
save_csv(
    explained_variance,
    "04_pca_explained_variance.csv"
)


# ============================================================
# 12 — VISUALIZE PRINCIPAL COMPONENT SCORES
# ============================================================

# Create a scatterplot of respondents projected onto
# the first two principal components.
#
# X-axis:
#   First principal component (PC1).
#
# Y-axis:
#   Second principal component (PC2).
#
# Color:
#   AI/AN-alone versus other identified single-race.
#
# Interpretation:
#   Respondents positioned near one another have
#   similar combinations of the coded predictors
#   along the displayed component directions.
#
# Overlap between populations does not establish
# equivalent health outcomes or experiences.


# ------------------------------------------------------------
# 12A — CREATE PCA SCATTERPLOT
# ------------------------------------------------------------

p <- ggplot2::ggplot(
    scores,
    ggplot2::aes(
        x = PC1,
        y = PC2,
        color = cohort
    )
) +
    
    # Plot individual respondents with transparency.
    #
    # Transparency reduces visual overplotting when
    # many respondents occupy similar positions.
    ggplot2::geom_point(
        alpha = 0.20,
        size = 0.65
    ) +
    
    # Add chart titles, axes, legend, and interpretation.
    ggplot2::labs(
        title = paste(
            "Patterns among demographic",
            "and health predictors"
        ),
        
        subtitle = sprintf(
            paste(
                "PCA training sample: n=%s;",
                "PC1 %.1f%%, PC2 %.1f%% of variance"
            ),
            format(
                nrow(scores),
                big.mark = ","
            ),
            variance[1],
            variance[2]
        ),
        
        x = "First principal component",
        
        y = "Second principal component",
        
        color = "Respondent group",
        
        caption = paste(
            "Points are respondents.",
            "Nearby points have similar combinations",
            "of coded predictors; overlap does not",
            "prove equal health.",
            "The FMD outcome was excluded from PCA fitting."
        )
    ) +
    
    # Apply consistent plot styling.
    ggplot2::theme_minimal(
        base_size = 12
    )


# ============================================================
# 13 — DISPLAY AND SAVE PCA FIGURE
# ============================================================

# Display the PCA scatterplot in RStudio.
print(p)

# Export the figure as a PNG file.
save_figure(
    p,
    "04_pca_scores.png"
)


# ============================================================
# 14 — RELEASE MEMORY
# ============================================================

# Remove large objects no longer required
# after PCA results have been saved.

rm(
    training,
    sampled,
    X
)


# ============================================================
# 15 — REPORT COMPLETION
# ============================================================

# Confirm successful PCA analysis and output generation.

message(
    "[COMPLETE] PCA and interpretation data saved."
)
